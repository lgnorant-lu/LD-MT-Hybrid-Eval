# @title --- MODULE: inference_runner ---

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .checkpoint import CheckpointStore
    from .schemas import RawInferenceItem, RawInferenceReport, RunMeta, validate_dataset_bundle
except ImportError:
    from checkpoint import CheckpointStore
    from schemas import RawInferenceItem, RawInferenceReport, RunMeta, validate_dataset_bundle


TranslatorFn = Callable[..., Any]


def _is_non_recoverable_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "strict local mode",
        "model dir invalid",
        "local_path_invalid",
        "non-recoverable translategemma config",
        "rope config patch verification failed",
        "rope config hard-patch failed",
        "weights not found",
        "cannot access gated repo",
        "gated repo",
        "download_non_recoverable",
        "repository not found",
        "404",
        "401",
        "403",
        "repo id must be in the form",
        "is not a valid model identifier",
        "unexpected keyword argument",
        "got an unexpected keyword argument",
        "engineargs.__init__",
        "validation error for modelconfig",
        "rope_parameters should have a 'rope_type' key",
        "rope_parameters should have a \"rope_type\" key",
        "pydantic.dev",
        "not supported",
        "architectures",
        "gemma3forcausallm",
        "non-recoverable translator error",
        "non-recoverable batch translator error",
    )
    return any(marker in message for marker in markers)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_llm_wrappers(text: str) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return ""

    # Strip common chat-template leftovers from some instruction-tuned models.
    candidate = candidate.replace("</|im_end|>", "").replace("<|im_end|>", "")
    candidate = candidate.replace("<|im_start|>assistant", "")
    candidate = candidate.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def robust_json_parser(text: str) -> dict[str, Any] | None:
    """Parse the most likely JSON object from noisy LLM output."""
    if not isinstance(text, str):
        return None

    candidate = _strip_llm_wrappers(text)
    if not candidate:
        return None

    def _try_parse(candidate_text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(candidate_text)
        except Exception:
            return None

        if isinstance(parsed, dict):
            return parsed

        # Some models return a JSON string that itself contains a JSON object.
        if isinstance(parsed, str):
            nested = _strip_llm_wrappers(parsed)
            if nested and nested != candidate_text:
                return robust_json_parser(nested)
        return None

    variants: list[str] = [candidate]

    # Recover fragments like '"en": "...", "zh-TW": "..." }' (missing leading '{').
    if re.match(r'^\s*"[^"\n]+"\s*:', candidate):
        variants.append("{" + candidate)

    # Recover trailing garbage after a valid object tail.
    if "}" in candidate and not candidate.rstrip().endswith("}"):
        trimmed = candidate[: candidate.rfind("}") + 1]
        variants.append(trimmed)
        if re.match(r'^\s*"[^"\n]+"\s*:', trimmed):
            variants.append("{" + trimmed)

    # Recover missing right braces in noisy generations.
    for item in list(variants):
        opens = item.count("{")
        closes = item.count("}")
        if opens > closes:
            variants.append(item + ("}" * (opens - closes)))

    deduped_variants: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            deduped_variants.append(item)

    for item in deduped_variants:
        parsed = _try_parse(item)
        if isinstance(parsed, dict):
            return parsed

    fenced_blocks = re.findall(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in reversed(fenced_blocks):
        parsed = _try_parse(block)
        if isinstance(parsed, dict):
            return parsed

    decoder = json.JSONDecoder()
    for source in deduped_variants:
        starts = [m.start() for m in re.finditer(r"\{", source)]
        for start in reversed(starts):
            try:
                parsed, _ = decoder.raw_decode(source[start:])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    return None


def _normalize_hypotheses(payload: Mapping[str, Any] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not isinstance(payload, Mapping):
        return normalized

    for lang, value in payload.items():
        if isinstance(value, Mapping):
            value = value.get("title") or value.get("text") or value.get("content") or ""
        text = str(value).strip()
        if text:
            normalized[str(lang)] = text
    return normalized


def _coverage_count(hypotheses: Mapping[str, Any], target_languages: Sequence[str] | None = None) -> int:
    if not isinstance(hypotheses, Mapping):
        return 0

    if target_languages:
        return sum(1 for language in target_languages if str(hypotheses.get(language, "")).strip())

    return sum(1 for value in hypotheses.values() if str(value).strip())


def _recover_hypotheses_from_payload(
    raw_text: str,
    hypotheses_payload: Mapping[str, Any] | None,
    default_language: str = "en",
) -> tuple[dict[str, str], str]:
    existing = _normalize_hypotheses(hypotheses_payload)

    parsed_from_raw = robust_json_parser(raw_text)
    recovered_from_raw = _normalize_hypotheses(parsed_from_raw)
    if recovered_from_raw:
        if set(recovered_from_raw.keys()) == {default_language}:
            nested = robust_json_parser(recovered_from_raw.get(default_language, ""))
            nested_hypotheses = _normalize_hypotheses(nested)
            if nested_hypotheses:
                return nested_hypotheses, "raw_nested"
        return recovered_from_raw, "raw"

    default_language_value = existing.get(default_language, "")
    if default_language_value:
        parsed_from_default = robust_json_parser(default_language_value)
        recovered_from_default = _normalize_hypotheses(parsed_from_default)
        if recovered_from_default:
            return recovered_from_default, "default_language_fragment"

    return existing, "existing"


def repair_raw_inference_file(
    raw_inference_path: Path,
    target_languages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Repair raw inference hypotheses in place without re-running model inference."""
    path = Path(raw_inference_path)
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return {
            "path": str(path),
            "rows_total": 0,
            "rows_changed": 0,
            "rows_repaired_from_raw": 0,
            "rows_repaired_from_default_language": 0,
            "rows_filled_empty_raw": 0,
            "changed": False,
        }

    targets = [str(language).strip() for language in list(target_languages or []) if str(language).strip()]
    default_language = targets[0] if targets else "en"

    stats = {
        "path": str(path),
        "rows_total": 0,
        "rows_changed": 0,
        "rows_repaired_from_raw": 0,
        "rows_repaired_from_default_language": 0,
        "rows_filled_empty_raw": 0,
    }

    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue

        stats["rows_total"] += 1
        original_hypotheses = _normalize_hypotheses(row.get("hypotheses", {}))
        raw_text = str(row.get("raw_llm_response", "") or "")

        recovered_hypotheses, recovered_from = _recover_hypotheses_from_payload(
            raw_text=raw_text,
            hypotheses_payload=original_hypotheses,
            default_language=default_language,
        )

        before = _coverage_count(original_hypotheses, targets)
        after = _coverage_count(recovered_hypotheses, targets)

        if after > before:
            row["hypotheses"] = recovered_hypotheses
            changed = True
            stats["rows_changed"] += 1

            if recovered_from in {"raw", "raw_nested"}:
                stats["rows_repaired_from_raw"] += 1
            elif recovered_from == "default_language_fragment":
                stats["rows_repaired_from_default_language"] += 1

            if str(row.get("status", "")).upper() != "SUCCESS":
                row["status"] = "SUCCESS"
                row["error_message"] = ""

        if not str(row.get("raw_llm_response", "")).strip() and recovered_hypotheses:
            row["raw_llm_response"] = json.dumps(recovered_hypotheses, ensure_ascii=False)
            changed = True
            stats["rows_filled_empty_raw"] += 1

    stats["changed"] = changed
    if not changed:
        return stats

    run_meta_raw = payload.get("run_meta", {})
    run_meta = dict(run_meta_raw) if isinstance(run_meta_raw, Mapping) else {}
    run_meta["repair_meta"] = {
        "timestamp": _utc_now_iso(),
        "target_languages": targets,
        "rows_total": stats["rows_total"],
        "rows_changed": stats["rows_changed"],
        "rows_repaired_from_raw": stats["rows_repaired_from_raw"],
        "rows_repaired_from_default_language": stats["rows_repaired_from_default_language"],
        "rows_filled_empty_raw": stats["rows_filled_empty_raw"],
    }
    payload["run_meta"] = run_meta

    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    tmp_path.replace(path)

    return stats


def _normalize_translator_response(candidate_response: Any, default_language: str = "en") -> dict[str, Any]:
    raw_text = ""
    hypotheses: dict[str, str] = {}

    if isinstance(candidate_response, Mapping):
        raw_text = str(candidate_response.get("raw_llm_response", "")).strip()

        hypotheses_raw = candidate_response.get("hypotheses")
        if not isinstance(hypotheses_raw, Mapping) and isinstance(candidate_response.get("translations"), Mapping):
            hypotheses_raw = candidate_response.get("translations")

        hypotheses = _normalize_hypotheses(hypotheses_raw)

        if not raw_text and isinstance(candidate_response.get("response"), str):
            raw_text = str(candidate_response.get("response", "")).strip()

        if not hypotheses and raw_text:
            parsed = robust_json_parser(raw_text)
            hypotheses = _normalize_hypotheses(parsed)

        if not hypotheses and raw_text:
            hypotheses = {default_language: raw_text}
    elif isinstance(candidate_response, str):
        raw_text = candidate_response.strip()
        parsed = robust_json_parser(raw_text)
        hypotheses = _normalize_hypotheses(parsed)
        if not hypotheses and raw_text:
            hypotheses = {default_language: raw_text}
    else:
        raise ValueError("Translator output must be a mapping or string.")

    if not hypotheses:
        raise ValueError("Translator output contains no non-empty hypotheses.")

    if not raw_text:
        raw_text = json.dumps(hypotheses, ensure_ascii=False)

    return {"hypotheses": hypotheses, "raw_llm_response": raw_text}


def mock_translator(
    model_id: str,
    source_text: str,
    test_block: str,
    dataset_item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Default translator used for local smoke tests."""
    del test_block, dataset_item
    translated_en = f"[{model_id}] {source_text}"
    return {
        "hypotheses": {"en": translated_en},
        "raw_llm_response": translated_en,
    }


class InferenceRunner:
    """Run model inference against normalized datasets with checkpoint support."""

    def __init__(self, translator: TranslatorFn | None = None) -> None:
        self.translator = translator or mock_translator

    def _call_single_translator(
        self,
        model_id: str,
        dataset_item: Mapping[str, Any],
        test_block: str,
    ) -> dict[str, Any]:
        source_text = str(dataset_item.get("source_text", ""))

        try:
            response = self.translator(model_id, source_text, test_block, dataset_item)
        except TypeError:
            response = self.translator(model_id, source_text, test_block)

        return _normalize_translator_response(response)

    def _translate_single_with_retries(
        self,
        model_id: str,
        dataset_item: Mapping[str, Any],
        test_block: str,
        max_retries: int,
    ) -> tuple[dict[str, Any], int]:
        last_exception: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return self._call_single_translator(model_id, dataset_item, test_block), attempt
            except Exception as exc:
                if _is_non_recoverable_runtime_error(exc):
                    raise RuntimeError(f"Non-recoverable translator error: {exc}") from exc
                last_exception = exc

        raise RuntimeError(f"Translator failed after {max_retries} attempts: {last_exception}")

    def _translate_batch_with_retries(
        self,
        model_id: str,
        dataset_items: list[Mapping[str, Any]],
        test_block: str,
        max_retries: int,
        inference_params: Mapping[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], int]:
        batch_callable = getattr(self.translator, "translate_batch", None)
        if not callable(batch_callable):
            raise ValueError("Batch translator is not available.")

        last_exception: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                try:
                    raw_responses = batch_callable(model_id, dataset_items, test_block, inference_params)
                except TypeError:
                    raw_responses = batch_callable(model_id, dataset_items, test_block)

                if not isinstance(raw_responses, list):
                    raise ValueError("translate_batch must return a list.")
                if len(raw_responses) != len(dataset_items):
                    raise ValueError(
                        f"translate_batch response length mismatch: {len(raw_responses)} != {len(dataset_items)}"
                    )

                normalized = [_normalize_translator_response(row) for row in raw_responses]
                return normalized, attempt
            except Exception as exc:
                if _is_non_recoverable_runtime_error(exc):
                    raise RuntimeError(f"Non-recoverable batch translator error: {exc}") from exc
                last_exception = exc

        raise RuntimeError(f"Batch translator failed after {max_retries} attempts: {last_exception}")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as fp:
            return json.load(fp)

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def run_block(
        self,
        model_id: str,
        test_block: str,
        dataset_path: Path,
        output_path: Path,
        checkpoint_path: Path | Any, # 兼容类型
        prompt_version: str = "v1",
        inference_params: Mapping[str, Any] | None = None,
        max_retries: int = 2,
        enable_dedup_cache: bool = True,
    ) -> dict[str, Any]:
        dataset = self._load_json(dataset_path)
        errors = validate_dataset_bundle(dataset)
        if errors:
            raise ValueError(f"Invalid dataset schema at {dataset_path}: {errors}")

        # 如果传入的是路径则初始化，否则直接使用传入的对象（支持外部强清空后的对象）
        checkpoint = checkpoint_path if not isinstance(checkpoint_path, (str, Path)) else CheckpointStore(Path(checkpoint_path))
        
        # 硬重置：如果推测出需要重跑（由调用者通过 clear() 处理），这里已经拿到了干净的 checkpoint 对象
        items = dataset["items"]
        source_cache: dict[str, dict[str, Any]] = {}
        source_attempts: dict[str, int] = {}
        source_errors: dict[str, str] = {}

        existing_results: dict[str, dict[str, Any]] = {}
        if output_path.exists():
            existing_report = self._load_json(output_path)
            for row in existing_report.get("results", []):
                existing_results[str(row.get("test_id"))] = row

        pending_items: list[Mapping[str, Any]] = []
        skipped_count = 0
        for item in items:
            test_id = str(item["test_id"])
            if checkpoint.is_done(test_id) and test_id in existing_results:
                skipped_count += 1
                continue
            pending_items.append(item)

        if skipped_count > 0:
            print(f"[{test_block}] Skipping {skipped_count} items from checkpoint ({checkpoint_path})")
        if pending_items:
            print(f"[{test_block}] Processing {len(pending_items)} new/pending items.")

        batch_callable = getattr(self.translator, "translate_batch", None)
        use_batch_translator = callable(batch_callable)
        inference_params_dict = dict(inference_params or {})
        batch_size = max(1, int(inference_params_dict.get("batch_size", 32)))

        if use_batch_translator and pending_items:
            chunk_size = max(1, batch_size)
            
            unique_items: list[Mapping[str, Any]] = []
            seen_source_for_batch: set[str] = set()

            for item in pending_items:
                source_text = str(item.get("source_text", ""))
                if enable_dedup_cache:
                    if source_text in source_cache or source_text in seen_source_for_batch:
                        continue
                    seen_source_for_batch.add(source_text)
                unique_items.append(item)

            if unique_items:
                print(f"[{test_block}] Total unique pending items to dispatch: {len(unique_items)}")
                
                for i in range(0, len(unique_items), chunk_size):
                    chunk = unique_items[i : i + chunk_size]
                    print(f"[{test_block}] Dispatching batch {i//chunk_size + 1} ({i}/{len(unique_items)} items) to vLLM engine...")
                    
                    try:
                        responses, attempts = self._translate_batch_with_retries(
                            model_id=model_id,
                            dataset_items=chunk,
                            test_block=test_block,
                            max_retries=max_retries,
                            inference_params=inference_params_dict,
                        )

                        for item, response in zip(chunk, responses):
                            source_text = str(item.get("source_text", ""))
                            source_cache[source_text] = copy.deepcopy(response)
                            source_attempts[source_text] = attempts
                            
                    except Exception as batch_exc:
                        if _is_non_recoverable_runtime_error(batch_exc):
                            raise RuntimeError(
                                f"[{test_block}] Non-recoverable batch failure, aborting block: {batch_exc}"
                            ) from batch_exc
                        print(f"[{test_block}] Batch chunk failed: {batch_exc}. Falling back to single-item mode for this chunk.")
                        # 核心修复：如果在 batch 模式下发生 OOM，必须彻底清理 translator 状态，否则后续 single 模式也会受阻
                        if "CUDA out of memory" in str(batch_exc):
                             print(f"[{test_block}] Detected OOM in batch. Hard-resetting translator device state...")
                             close_fn = getattr(self.translator, "close", None)
                             if callable(close_fn):
                                 try:
                                     close_fn()
                                 except Exception:
                                     pass

                        for item in chunk:
                            source_text = str(item.get("source_text", ""))
                            try:
                                response, attempts = self._translate_single_with_retries(
                                    model_id=model_id,
                                    dataset_item=item,
                                    test_block=test_block,
                                    max_retries=max_retries,
                                )
                                source_cache[source_text] = copy.deepcopy(response)
                                source_attempts[source_text] = attempts
                            except Exception as single_exc:
                                source_errors[source_text] = f"{batch_exc}; fallback={single_exc}"
                                if _is_non_recoverable_runtime_error(single_exc):
                                    raise RuntimeError(
                                        f"[{test_block}] Non-recoverable single fallback failure, aborting block: {single_exc}"
                                    ) from single_exc
                
                print(f"[{test_block}] All batches dispatched successfully.")

        run_meta = RunMeta(
            model_id=model_id,
            test_block=test_block,
            timestamp=_utc_now_iso(),
            prompt_version=prompt_version,
            inference_params=dict(inference_params or {}),
        )

        source_seen_count: dict[str, int] = {}
        for item in items:
            test_id = str(item["test_id"])
            source_text = str(item["source_text"])

            if checkpoint.is_done(test_id) and test_id in existing_results:
                continue

            attempts_used = 0
            cache_hit = False
            try:
                if enable_dedup_cache and source_text in source_cache:
                    model_response = copy.deepcopy(source_cache[source_text])
                    cache_hit = source_seen_count.get(source_text, 0) > 0
                    attempts_used = max(1, int(source_attempts.get(source_text, 1)))
                elif source_text in source_errors:
                    raise RuntimeError(source_errors[source_text])
                else:
                    model_response, attempts_used = self._translate_single_with_retries(
                        model_id=model_id,
                        dataset_item=item,
                        test_block=test_block,
                        max_retries=max_retries,
                    )

                    if enable_dedup_cache:
                        source_cache[source_text] = copy.deepcopy(model_response)
                        source_attempts[source_text] = attempts_used

                hypotheses = dict(model_response.get("hypotheses", {}))
                raw_llm_response = str(model_response.get("raw_llm_response", ""))
                status = "SUCCESS"
                error_message = ""
                checkpoint.mark_done(test_id)
            except Exception as exc:
                hypotheses = {}
                raw_llm_response = ""
                status = "ERROR"
                error_message = str(exc)
                if attempts_used <= 0:
                    attempts_used = max_retries
                checkpoint.mark_failed(test_id, str(exc))

            source_seen_count[source_text] = source_seen_count.get(source_text, 0) + 1

            # --- [Pass-through context for Local Handoff] ---
            term_rules = item.get("term_rules", {})
            term_context = ""
            if isinstance(term_rules, Mapping):
                term_context = str(term_rules.get("llm_instruction", "")).strip()

            existing_results[test_id] = RawInferenceItem(
                test_id=test_id,
                status=status,
                hypotheses=hypotheses,
                raw_llm_response=raw_llm_response,
                source_text=source_text,
                reference_translations=dict(item.get("reference_translations", {})),
                term_context=term_context,
                audit_tags=list(item.get("audit_tags", [])),
                error_message=error_message,
                attempts=attempts_used,
                cache_hit=cache_hit,
            ).__dict__

            ordered_results = [existing_results[str(row["test_id"])] for row in items if str(row["test_id"]) in existing_results]
            report = RawInferenceReport(run_meta=run_meta, results=[RawInferenceItem(**row) for row in ordered_results])
            self._save_json(output_path, report.to_dict())

        final_results = [existing_results[str(row["test_id"])] for row in items if str(row["test_id"]) in existing_results]
        final_report = RawInferenceReport(run_meta=run_meta, results=[RawInferenceItem(**row) for row in final_results])
        payload = final_report.to_dict()
        payload["run_meta"]["completed_at"] = _utc_now_iso()
        payload["run_meta"]["total_items"] = len(items)
        payload["run_meta"]["done_items"] = checkpoint.done_count()
        payload["run_meta"]["failed_items"] = checkpoint.failed_count()
        payload["run_meta"]["max_retries"] = max_retries
        payload["run_meta"]["dedup_cache_enabled"] = enable_dedup_cache
        payload["run_meta"]["dedup_cache_entries"] = len(source_cache)
        payload["run_meta"]["batch_translator_used"] = use_batch_translator
        payload["run_meta"]["batch_size"] = batch_size
        self._save_json(output_path, payload)
        return payload

# %%