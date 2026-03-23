# @title --- MODULE: runner_entry ---
import argparse
import gc
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

try:
    from .adapters import build_all_standard_datasets
    from .aggregator import GlobalMetricsAggregator
    from .audit_evaluator import AuditEvaluator
    from .checkpoint import CheckpointStore
    from .config import (
        B2_SCHEMA_VERSION,
        DEFAULT_TARGET_LANGUAGE,
        TARGET_LANGUAGES,
        TEST_BLOCKS,
        ArenaPaths,
        sanitize_model_id as config_sanitize_model_id,
    )
    from .filesystem import ExperimentDirectoryManager
    from .inference_runner import (
        InferenceRunner,
        _normalize_hypotheses,
        mock_translator,
        repair_raw_inference_file,
        robust_json_parser,
    )
    from .schemas import normalize_term_rules
except ImportError:
    from adapters import build_all_standard_datasets
    from aggregator import GlobalMetricsAggregator
    from audit_evaluator import AuditEvaluator
    from checkpoint import CheckpointStore
    from config import (
        B2_SCHEMA_VERSION,
        DEFAULT_TARGET_LANGUAGE,
        TARGET_LANGUAGES,
        TEST_BLOCKS,
        ArenaPaths,
        sanitize_model_id as config_sanitize_model_id,
    )
    from filesystem import ExperimentDirectoryManager
    from inference_runner import (
        InferenceRunner,
        _normalize_hypotheses,
        mock_translator,
        repair_raw_inference_file,
        robust_json_parser,
    )
    from schemas import normalize_term_rules


DEFAULT_15_MODEL_REPOS: Sequence[str] = (
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "tencent/HY-MT1.5-1.8B",
    "tencent/HY-MT1.5-7B",
    "google/translategemma-4b-it",
    "google/translategemma-12b-it",
    "CohereLabs/tiny-aya-global",
    "CohereLabs/tiny-aya-water",
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
)


DEFAULT_VLLM_MODEL_MAP: dict[str, str] = {
    # 标准 Qwen 系列
    "qwen2.5:7b-instruct": "/root/models/Qwen2.5-7B-Instruct",
    "qwen2.5:14b-instruct": "/root/models/Qwen2.5-14B-Instruct",
    
    # DeepSeek R1 系列 (Distill 版本通常存放在 Llama 或 Qwen 目录下)
    "deepseek-r1:7b": "/root/models/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:8b": "/root/models/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1:14b": "/root/models/DeepSeek-R1-Distill-Qwen-14B",

    # 定向翻译增强模型 (MT Specific)
    "translategemma:4b": "/root/models/TranslateGemma-4B",
    "cas/alma-r": "/root/models/ALMA-R",
    "thinkverse/towerinstruct": "/root/models/TowerInstruct-7B-v0.2",
    "huihui_ai/hy-mt1.5-abliterated": "/root/models/HY-MT1.5-Abliterated",

    # Llama / Gemma / DeepSeek 系列
    "llama-3.1-8b": "/root/models/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "/root/models/Meta-Llama-3.1-70B-Instruct",
    "gemma-2-9b": "/root/models/Gemma-2-9B-It",
    "deepseek-v3": "/root/models/DeepSeek-V3",

    # 多语言与通用强模型
    "aya:8b": "/root/models/Aya-23-8B",
    "aya-expanse:8b": "/root/models/Aya-Expanse-8B",
    "glm4:9b": "/root/models/GLM-4-9B-Chat",
    "yi1.5:9b": "/root/models/Yi-1.5-9B-Chat",
    "llama3.1:8b": "/root/models/Meta-Llama-3.1-8B-Instruct",
    "gemma2:9b": "/root/models/Gemma-2-9B-It",
    "gemma3:4b": "/root/models/Gemma-3-4B-It",

    # 审计后的 15 模型（repo_id 直连）
    "google/gemma-3-1b-it": "/root/models/google_gemma-3-1b-it",
    "google/gemma-3-4b-it": "/root/models/google_gemma-3-4b-it",
    "google/gemma-3-12b-it": "/root/models/google_gemma-3-12b-it",
    "qwen/qwen3.5-9b": "/root/models/Qwen_Qwen3.5-9B",
    "qwen/qwen3.5-4b": "/root/models/Qwen_Qwen3.5-4B",
    "qwen/qwen3-14b": "/root/models/Qwen_Qwen3-14B",
    "qwen/qwen2.5-7b-instruct": "/root/models/Qwen_Qwen2.5-7B-Instruct",
    "qwen/qwen2.5-14b-instruct": "/root/models/Qwen_Qwen2.5-14B-Instruct",
    "tencent/hy-mt1.5-1.8b": "/root/models/tencent_HY-MT1.5-1.8B",
    "tencent/hy-mt1.5-7b": "/root/models/tencent_HY-MT1.5-7B",
    "google/translategemma-4b-it": "/root/models/google_translategemma-4b-it",
    "google/translategemma-12b-it": "/root/models/google_translategemma-12b-it",
    "coherelabs/tiny-aya-global": "/root/models/CohereLabs_tiny-aya-global",
    "coherelabs/tiny-aya-water": "/root/models/CohereLabs_tiny-aya-water",
    "deepseek-ai/deepseek-r1-distill-llama-8b": "/root/models/deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
}


def sanitize_model_id(model_id: str) -> str:
    """Standardize model ID to filesystem-friendly format."""
    return model_id.replace(":", "-").replace("/", "--")


def _normalize_model_key(model_id: str) -> str:
    return str(model_id).strip().lower().replace(":", "-").replace("/", "-")


def _is_valid_local_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return True


def _count_missing_rope_type_fields(node: Any) -> int:
    """Count rope_parameters mappings that still miss a non-empty rope_type."""
    if isinstance(node, Mapping):
        missing = 0
        rope_parameters = node.get("rope_parameters")
        if isinstance(rope_parameters, Mapping):
            rope_type = str(rope_parameters.get("rope_type", "")).strip()
            if not rope_type:
                missing += 1

        for value in node.values():
            missing += _count_missing_rope_type_fields(value)
        return missing

    if isinstance(node, list):
        return sum(_count_missing_rope_type_fields(item) for item in node)

    return 0


def _patch_rope_fields_in_tree(node: dict[str, Any], inherited_rope_type: str = "") -> tuple[bool, str]:
    """Recursively inject rope_type for rope_parameters / rope_scaling variants."""
    changed = False
    current_rope_type = str(inherited_rope_type or "").strip()

    rope_parameters = node.get("rope_parameters")
    if isinstance(rope_parameters, Mapping):
        rope_parameters_dict = dict(rope_parameters)
        rope_type = str(rope_parameters_dict.get("rope_type", "")).strip()

        if not rope_type:
            from_type = rope_parameters_dict.get("type")
            if isinstance(from_type, str) and from_type.strip():
                rope_type = from_type.strip()

        if not rope_type:
            local_scaling = node.get("rope_scaling")
            if isinstance(local_scaling, Mapping):
                from_scaling = local_scaling.get("rope_type") or local_scaling.get("type")
                if isinstance(from_scaling, str) and from_scaling.strip():
                    rope_type = from_scaling.strip()

        if not rope_type:
            rope_type = current_rope_type or "default"

        if str(rope_parameters_dict.get("rope_type", "")).strip() != rope_type:
            rope_parameters_dict["rope_type"] = rope_type
            node["rope_parameters"] = rope_parameters_dict
            changed = True

        current_rope_type = rope_type

    rope_scaling = node.get("rope_scaling")
    if isinstance(rope_scaling, Mapping):
        rope_scaling_dict = dict(rope_scaling)
        scaling_rope_type = str(rope_scaling_dict.get("rope_type", "")).strip()
        if not scaling_rope_type:
            scaling_type = rope_scaling_dict.get("type")
            if isinstance(scaling_type, str) and scaling_type.strip():
                scaling_rope_type = scaling_type.strip()

        if not scaling_rope_type:
            scaling_rope_type = current_rope_type or "default"

        if str(rope_scaling_dict.get("rope_type", "")).strip() != scaling_rope_type:
            rope_scaling_dict["rope_type"] = scaling_rope_type
            node["rope_scaling"] = rope_scaling_dict
            changed = True

        current_rope_type = scaling_rope_type

    for key, value in list(node.items()):
        if isinstance(value, Mapping):
            child_node = dict(value)
            child_changed, child_rope_type = _patch_rope_fields_in_tree(child_node, current_rope_type)
            if child_changed or not isinstance(value, dict):
                node[key] = child_node
            if child_changed:
                changed = True
            if child_rope_type:
                current_rope_type = child_rope_type
            continue

        if isinstance(value, list):
            list_changed = False
            new_list: list[Any] = []
            list_rope_type = current_rope_type
            for item in value:
                if isinstance(item, Mapping):
                    child_node = dict(item)
                    child_changed, child_rope_type = _patch_rope_fields_in_tree(child_node, list_rope_type)
                    new_list.append(child_node)
                    if child_changed:
                        list_changed = True
                    if child_rope_type:
                        list_rope_type = child_rope_type
                else:
                    new_list.append(item)

            if list_changed:
                node[key] = new_list
                changed = True
            if list_rope_type:
                current_rope_type = list_rope_type

    return changed, current_rope_type


def _patch_local_rope_parameters_config(model_dir: Path, model_id: str) -> bool:
    """Patch legacy rope config fields for stricter vLLM/Pydantic validators.

    Some TranslateGemma checkpoints expose `rope_parameters` without
    `rope_type`, which can fail at vLLM model config validation time.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False

    try:
        with config_path.open("r", encoding="utf-8") as fp:
            payload_obj = json.load(fp)
    except Exception as exc:
        print(f"[vllm] Rope config patch skipped for {model_id}: cannot read config.json ({exc})")
        return False

    if not isinstance(payload_obj, (dict, Mapping)):
        return False

    payload = dict(payload_obj)
    missing_before = _count_missing_rope_type_fields(payload)
    changed, _ = _patch_rope_fields_in_tree(payload)
    missing_after = _count_missing_rope_type_fields(payload)

    if missing_after > 0:
        print(
            f"[vllm] Rope config patch verification failed for {model_id}: "
            f"missing rope_type before={missing_before}, after={missing_after}"
        )
        return False

    if not changed:
        if missing_before > 0:
            print(
                f"[vllm] Rope config verified for {model_id}: "
                "no file rewrite required after in-memory normalization."
            )
        else:
            print(f"[vllm] Rope config already compatible for {model_id}.")
        return True

    try:
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        print(
            f"[vllm] HARD-PATCH: Applied rope config fix to {model_id} at {config_path} "
            f"(missing_before={missing_before}, missing_after={missing_after})"
        )
        return True
    except Exception as exc:
        print(f"[vllm] Rope config hard-patch failed for {model_id}: {exc}")
        return False


def _canonicalize_model_ids(model_ids: Sequence[str]) -> list[str]:
    """Ensure we handle both 'default' keyword and raw lists correctly."""
    deduped: list[str] = []
    seen: set[str] = set()
    
    # 逻辑修复：处理用户可能传入的多种 'default' 情况
    input_list = list(model_ids)
    
    # 1. 处理 ['default']
    if len(input_list) == 1 and str(input_list[0]).lower() == "default":
        input_list = list(DEFAULT_15_MODEL_REPOS)
    # 2. 处理 'default' (作为单个字符串误传)
    elif isinstance(model_ids, str) and model_ids.lower() == "default":
        input_list = list(DEFAULT_15_MODEL_REPOS)
    # 3. 处理空列表，兜底到 default
    elif not input_list:
        input_list = list(DEFAULT_15_MODEL_REPOS)

    for raw in input_list:
        item = str(raw).strip()
        if not item:
            continue
        key = _normalize_model_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _auto_batch_size_for_model(model_id: str, requested: int, translator_name: str) -> int:
    if translator_name != "vllm":
        return max(1, int(requested))

    req = max(1, int(requested))
    # 如果用户显式配置了非默认值，优先尊重用户设置。
    if req != 128:
        return req

    free_gb: float | None = None
    try:
        torch = importlib.import_module("torch")
        cuda_module = getattr(torch, "cuda", None)
        mem_get_info = getattr(cuda_module, "mem_get_info", None) if cuda_module is not None else None
        if callable(mem_get_info):
            mem_info_raw = mem_get_info()
            if isinstance(mem_info_raw, (list, tuple)) and len(mem_info_raw) >= 2:
                free_gb = float(mem_info_raw[0]) / (1024 ** 3)
    except Exception:
        free_gb = None

    lowered = str(model_id).lower()
    if any(tag in lowered for tag in ("70b", "72b", "35b", "32b", "27b")):
        return 24
    if any(tag in lowered for tag in ("14b", "12b")):
        if free_gb is not None and free_gb >= 28.0:
            return 128
        if free_gb is not None and free_gb >= 20.0:
            return 96
        return 64
    if any(tag in lowered for tag in ("9b", "8b", "7b")):
        if free_gb is not None and free_gb >= 24.0:
            return 192
        return 128
    if any(tag in lowered for tag in ("4b", "3b", "2b", "1.8b", "1b", "0.8b")):
        if free_gb is not None and free_gb >= 24.0:
            return 256
        return 192
    return 128


def _is_non_recoverable_download_error(message: str) -> bool:
    lowered = str(message).lower()
    hard_markers = (
        "cannot access gated repo",
        "gated repo",
        "401",
        "403",
        "repository not found",
        "404",
        "is not a valid model identifier",
        "repo id must be in the form",
        "not supported",
        "architectures",
        "gemma3forcausallm",
    )
    return any(marker in lowered for marker in hard_markers)


def _default_metadata_root(workspace_root: Path) -> Path:
    local = workspace_root / "Metadatas"
    colab = Path("/content/drive/MyDrive/LinuxDo/Metadatas")
    if local.exists():
        return local
    return colab


def _default_benchmark_root(workspace_root: Path) -> Path:
    local = workspace_root / "Benchmarks"
    colab = Path("/content/drive/MyDrive/LinuxDo/Benchmarks")
    if local.exists() and any(local.iterdir()):
        return local
    return colab


def _runtime_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _runtime_load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"Expected JSON object at {path}")


def _collect_dataset_test_ids(dataset_path: Path) -> set[str]:
    dataset = _runtime_load_json(dataset_path)
    items = dataset.get("items", [])
    if not isinstance(items, list):
        return set()

    test_ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        test_id = str(item.get("test_id", "")).strip()
        if test_id:
            test_ids.add(test_id)
    return test_ids


def _inspect_raw_test_id_alignment(raw_inference_path: Path, dataset_test_ids: set[str]) -> dict[str, Any]:
    raw_payload = _runtime_load_json(raw_inference_path)
    rows = raw_payload.get("results", [])
    if not isinstance(rows, list):
        rows = []

    rows_total = 0
    rows_matched = 0
    unmatched_examples: list[str] = []

    for row in rows:
        if not isinstance(row, Mapping):
            continue

        test_id = str(row.get("test_id", "")).strip()
        if not test_id:
            continue

        rows_total += 1
        if test_id in dataset_test_ids:
            rows_matched += 1
        elif len(unmatched_examples) < 8:
            unmatched_examples.append(test_id)

    matched_rate = 1.0
    if rows_total > 0:
        matched_rate = rows_matched / rows_total

    return {
        "rows_total": rows_total,
        "rows_matched": rows_matched,
        "rows_unmatched": max(rows_total - rows_matched, 0),
        "matched_rate": matched_rate,
        "unmatched_examples": unmatched_examples,
    }


def _parse_csv_arg(raw: str) -> list[str]:
    return [segment.strip() for segment in str(raw).split(",") if segment.strip()]


def _resolve_existing_dataset_map(paths: ArenaPaths) -> dict[str, Path]:
    """Return dataset map from existing files, with fuzzy fallback for legacy names."""
    dataset_map: dict[str, Path] = {}
    if not paths.datasets_dir.exists() or not paths.datasets_dir.is_dir():
        return dataset_map

    json_files = [candidate for candidate in paths.datasets_dir.glob("*.json") if candidate.is_file()]
    indexed_files: list[tuple[Path, str, str]] = []
    for candidate in json_files:
        stem = candidate.stem
        indexed_files.append((candidate, _normalize_block_key(stem), stem.lower()))

    for block in TEST_BLOCKS:
        # 1) 精确优先：兼容带 _v1 和不带后缀
        exact_candidates = [
            paths.datasets_dir / f"{block}_v1.json",
            paths.datasets_dir / f"{block}.json",
        ]
        found_exact = False
        for candidate in exact_candidates:
            if candidate.exists() and candidate.is_file():
                dataset_map[block] = candidate
                found_exact = True
                break
        if found_exact:
            continue

        # 2) 模糊回退：例如 Baseline_Standard_20260315.json / Baseline-Standard.json
        expected_key = _normalize_block_key(block)
        scored_matches: list[tuple[tuple[int, int, int, int, int], Path]] = []
        for candidate, normalized_stem, stem_lower in indexed_files:
            if not normalized_stem:
                continue
            exact_match = 1 if normalized_stem == expected_key else 0
            starts_with = 1 if normalized_stem.startswith(expected_key) else 0
            contains = 1 if expected_key in normalized_stem else 0
            if exact_match == 0 and starts_with == 0 and contains == 0:
                continue

            has_v1_suffix = 1 if stem_lower.endswith("_v1") else 0
            # 评分优先级：精确 > 前缀 > 包含 > 带 v1 > 名称更短
            score = (exact_match, starts_with, contains, has_v1_suffix, -len(normalized_stem))
            scored_matches.append((score, candidate))

        if scored_matches:
            scored_matches.sort(key=lambda item: item[0], reverse=True)
            dataset_map[block] = scored_matches[0][1]

    return dataset_map


def _candidate_model_dir_names(model_id: str) -> list[str]:
    raw = str(model_id).strip()
    # 彻底解决导出包中 google--gemma-3-1b-it 和 google_gemma-3-1b-it 的不一致问题
    # 增加更多变体，确保 audit 阶段能找回所有 15 个模型
    names = [
        # 0. config 中的标准化结果（不同版本可能不同）
        sanitize_model_id(raw),
        # 1. 之前修正后的 sanitize 逻辑 (应生成 google--gemma...)
        sanitize_model_id(raw),
        # 2. 导出包中最稳妥的 vendor--model 格式
        raw.replace(":", "-").replace("/", "--"),
        # 3. 简单的下划线格式
        raw.replace(":", "-").replace("/", "_"),
        # 4. 混合双下划线
        raw.replace(":", "-").replace("/", "__"),
        # 5. 全小写变体
        raw.lower().replace(":", "-").replace("/", "_"),
        raw.lower().replace(":", "-").replace("/", "--"),
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        normalized = str(name).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_match_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _split_match_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    token_chars: list[str] = []
    for ch in str(value).lower():
        if ch.isalnum():
            token_chars.append(ch)
            continue
        if token_chars:
            token = "".join(token_chars)
            if len(token) >= 2:
                tokens.add(token)
            token_chars.clear()

    if token_chars:
        token = "".join(token_chars)
        if len(token) >= 2:
            tokens.add(token)

    return tokens

def _resolve_model_dir_for_audit(paths: ArenaPaths, model_id: str) -> Path:
    """Resolve model run directory via physical scan to tolerate Drive/FUSE naming drift."""
    runs_dir = paths.runs_dir
    if not runs_dir.exists() or not runs_dir.is_dir():
        return paths.model_run_dir(model_id)

    try:
        physical_dirs = [candidate for candidate in runs_dir.iterdir() if candidate.is_dir()]
    except Exception as exc:
        print(f"[audit] Warning: failed physical scan in {runs_dir} for {model_id}: {exc}")
        return paths.model_run_dir(model_id)

    raw_model_id = str(model_id).strip()
    model_tail = raw_model_id.split("/", 1)[-1] if "/" in raw_model_id else raw_model_id
    vendor = raw_model_id.split("/", 1)[0].lower().strip() if "/" in raw_model_id else ""

    signature_values = [
        raw_model_id,
        raw_model_id.lower(),
        model_tail,
        model_tail.lower(),
        sanitize_model_id(raw_model_id),
        # config_sanitize_model_id(raw_model_id),
    ]
    signature_values.extend(_candidate_model_dir_names(raw_model_id))
    signature_keys = {_normalize_match_key(value) for value in signature_values if str(value).strip()}
    model_tokens = _split_match_tokens(raw_model_id)

    best_path: Path | None = None
    best_score: tuple[int, int, int, int, int, int] | None = None

    for candidate in physical_dirs:
        candidate_name = candidate.name
        candidate_key = _normalize_match_key(candidate_name)
        if not candidate_key:
            continue

        exact_match = 1 if candidate_key in signature_keys else 0
        partial_match = 0
        if exact_match == 0:
            for key in signature_keys:
                if key and (key in candidate_key or candidate_key in key):
                    partial_match = 1
                    break

        token_overlap = len(model_tokens & _split_match_tokens(candidate_name))
        vendor_hit = 1 if vendor and vendor in candidate_name.lower() else 0

        # 必须满足至少一种强匹配信号，避免误命中其他模型目录。
        if exact_match == 0 and partial_match == 0 and token_overlap < 2:
            continue

        raw_dir = candidate / "raw_inference"
        audit_dir = candidate / "audited_reports"
        raw_count = len(list(raw_dir.glob("*_raw.json"))) if raw_dir.exists() else 0
        audit_count = len(list(audit_dir.glob("*_audit.json"))) if audit_dir.exists() else 0

        score = (exact_match, partial_match, vendor_hit, token_overlap, raw_count, audit_count)
        if best_score is None or score > best_score:
            best_score = score
            best_path = candidate

    if best_path is not None:
        return best_path

    # 最后回退到旧逻辑，保证路径行为可预测。
    for folder_name in _candidate_model_dir_names(raw_model_id):
        candidate = runs_dir / folder_name
        if candidate.exists() and candidate.is_dir():
            return candidate

    return paths.model_run_dir(model_id)


def _normalize_block_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _resolve_raw_path_for_block(model_dir: Path, block: str) -> Path | None:
    """Resolve block raw file robustly across naming variants.

    Expected format is <BlockName>_raw.json, but exported bundles may use
    lowercase/dash variants. This resolver keeps audit stage resilient.
    """
    raw_dir = model_dir / "raw_inference"
    if not raw_dir.exists() or not raw_dir.is_dir():
        return None

    direct_candidates = [
        raw_dir / f"{block}_raw.json",
        raw_dir / f"{block.lower()}_raw.json",
        raw_dir / f"{block.replace('_', '-')}_raw.json",
        raw_dir / f"{block.replace('-', '_')}_raw.json",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    expected_key = _normalize_block_key(block)
    fuzzy_matches: list[Path] = []
    for candidate in raw_dir.glob("*_raw.json"):
        stem = candidate.stem
        block_part = stem[:-4] if stem.endswith("_raw") else stem
        if _normalize_block_key(block_part) == expected_key:
            fuzzy_matches.append(candidate)

    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    if fuzzy_matches:
        return sorted(fuzzy_matches)[0]
    return None


def _normalize_text_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("title") or value.get("text") or value.get("content") or "").strip()
    return str(value).strip() if value is not None else ""


def _extract_reference_text(dataset_item: Mapping[str, Any], preferred_language: str) -> str:
    refs = dataset_item.get("reference_translations", {})
    if not isinstance(refs, Mapping):
        return ""

    preferred = _normalize_text_value(refs.get(preferred_language))
    if preferred:
        return preferred

    fallback_en = _normalize_text_value(refs.get("en"))
    if fallback_en:
        return fallback_en

    for value in refs.values():
        text = _normalize_text_value(value)
        if text:
            return text
    return ""


def _extract_hypothesis_text(raw_row: Mapping[str, Any], preferred_language: str) -> str:
    hypotheses = raw_row.get("hypotheses", {})
    if not isinstance(hypotheses, Mapping):
        return ""

    preferred = _normalize_text_value(hypotheses.get(preferred_language))
    if preferred:
        return preferred

    fallback_en = _normalize_text_value(hypotheses.get("en"))
    if fallback_en:
        return fallback_en

    for value in hypotheses.values():
        text = _normalize_text_value(value)
        if text:
            return text
    return ""


def _is_notebook_runtime() -> bool:
    try:
        from IPython.core.getipython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # 兼容更多 Notebook 类型
        name = shell.__class__.__name__
        return name in {"ZMQInteractiveShell", "TerminalInteractiveShell", "Shell"}
    except Exception:
        return False


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _torch_cuda_available(torch_module: Any) -> bool:
    cuda_module = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda_module, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


def _runtime_environment_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "python": sys.version.split()[0],
        "is_notebook": _is_notebook_runtime(),
        "cwd": str(Path.cwd()),
        "modules": {
            "requests": _module_available("requests"),
            "vllm": _module_available("vllm"),
            "sacrebleu": _module_available("sacrebleu"),
            "comet": _module_available("comet"),
        },
    }

    try:
        torch = importlib.import_module("torch")
        cuda_module = getattr(torch, "cuda", None)
        has_cuda = _torch_cuda_available(torch)

        device_count_fn = getattr(cuda_module, "device_count", None)
        get_name_fn = getattr(cuda_module, "get_device_name", None)

        count = 0
        if callable(device_count_fn) and has_cuda:
            try:
                count = int(str(device_count_fn()))
            except Exception:
                count = 0
        name = str(get_name_fn(0)) if callable(get_name_fn) and has_cuda and count > 0 else ""

        status["gpu"] = {
            "available": has_cuda,
            "count": count,
            "name": name,
        }
    except Exception as exc:
        status["gpu"] = {"available": False, "error": str(exc)}

    return status


def _ollama_healthcheck(base_url: str, timeout: int) -> None:
    health_url = base_url.rstrip("/") + "/api/tags"
    resp = requests.get(health_url, timeout=timeout)
    resp.raise_for_status()


def _warmup_model(base_url: str, model_id: str, timeout: int) -> None:
    endpoint = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model_id,
        "prompt": "warmup",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()


def _ensure_model_downloaded(model_id: str, target_root: Path, model_map: Mapping[str, str]) -> str:
    """Ensure vLLM model is available locally, or download from HF if missing."""
    normalized_id = _normalize_model_key(model_id)

    # 优先尝试 model_map 里的显式本地映射。
    map_keys = {
        str(model_id).strip().lower(),
        str(model_id).strip().lower().replace("-", ":"),
        str(model_id).strip().lower().replace(":", "-"),
    }
    for key in map_keys:
        mapped_path = model_map.get(key)
        if mapped_path:
            mapped = Path(mapped_path)
            if _is_valid_local_model_dir(mapped):
                print(f"[vllm] Local map hit: {model_id} -> {mapped}")
                return str(mapped)

    HF_REGISTRY = {
        # Qwen 2.5 / 3 / 3.5
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        "qwen3.5-9b": "Qwen/Qwen3.5-9B",
        "qwen3.5-4b": "Qwen/Qwen3.5-4B",
        "qwen3.5-14b": "Qwen/Qwen3-14B",
        "qwen3-14b": "Qwen/Qwen3-14B",

        # DeepSeek
        "deepseek-r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai-deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",

        # Gemma / TranslateGemma
        "gemma-3-1b-it": "google/gemma-3-1b-it",
        "gemma-3-4b-it": "google/gemma-3-4b-it",
        "gemma-3-12b-it": "google/gemma-3-12b-it",
        "google-gemma-3-1b-it": "google/gemma-3-1b-it",
        "google-gemma-3-4b-it": "google/gemma-3-4b-it",
        "google-gemma-3-12b-it": "google/gemma-3-12b-it",
        "translategemma-4b-it": "google/translategemma-4b-it",
        "translategemma-12b-it": "google/translategemma-12b-it",
        "google-translategemma-4b-it": "google/translategemma-4b-it",
        "google-translategemma-12b-it": "google/translategemma-12b-it",

        # Tencent
        "hy-mt1.5-1.8b": "tencent/HY-MT1.5-1.8B",
        "hy-mt1.5-7b": "tencent/HY-MT1.5-7B",
        "tencent-hy-mt1.5-1.8b": "tencent/HY-MT1.5-1.8B",
        "tencent-hy-mt1.5-7b": "tencent/HY-MT1.5-7B",

        # Cohere tiny aya
        "tiny-aya-global": "CohereLabs/tiny-aya-global",
        "tiny-aya-water": "CohereLabs/tiny-aya-water",
        "coherelabs-tiny-aya-global": "CohereLabs/tiny-aya-global",
        "coherelabs-tiny-aya-water": "CohereLabs/tiny-aya-water",
    }

    repo_id = HF_REGISTRY.get(normalized_id)
    if not repo_id:
        if model_id.startswith("/") or model_id.startswith("./"):
            local_path = Path(model_id)
            if not _is_valid_local_model_dir(local_path):
                raise RuntimeError(f"[vllm] LOCAL_PATH_INVALID: {model_id} (config.json missing)")
            return str(local_path)
        repo_id = model_id if "/" in model_id else model_id.replace(":", "/")

    sanitized_name = model_id.replace(":", "-").replace("/", "_")
    potential_path = target_root / sanitized_name

    if _is_valid_local_model_dir(potential_path):
        print(f"[vllm] Local cache verified: {potential_path}")
        return str(potential_path)

    free_gb = None
    try:
        free_gb = shutil.disk_usage(target_root).free / (1024 ** 3)
    except Exception:
        free_gb = None

    # 磁盘不足时才清理缓存，避免每次下载都触发大规模删库导致抖动。
    if free_gb is None or free_gb < 15.0: # 降低清理门槛，A100 环境通常有较大挂载盘
        free_note = "unknown" if free_gb is None else f"{free_gb:.2f}GB"
        print(f"[vllm] Disk free low ({free_note}). Safe-purging model cache...")
        for old_model in target_root.glob("*"):
            if old_model.is_dir() and old_model.name != sanitized_name:
                try:
                    shutil.rmtree(old_model)
                except Exception:
                    pass

    # 自动处理 repo_id：如果以 deepseek-ai/ 或 google/ 开头，保持原样
    if "/" in repo_id and not repo_id.startswith("/"):
        pass 
    elif normalized_id in HF_REGISTRY:
        repo_id = HF_REGISTRY[normalized_id]
    
    print(f"[vllm] Starting speed-optimized download: {repo_id} -> {potential_path}")
    try:
        from huggingface_hub import snapshot_download

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token and _is_notebook_runtime():
            try:
                colab_module = importlib.import_module("google.colab")
                userdata = getattr(colab_module, "userdata", None)
                hf_token = userdata.get("HF_TOKEN") if userdata is not None else None
                if hf_token:
                    print("[vllm] HF_TOKEN resolved via Colab UserData.")
            except Exception:
                pass

        path = snapshot_download(
            repo_id=repo_id,
            local_dir=potential_path,
            token=hf_token,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "onnx*", "flax*", "torch*"],
        )

        downloaded_path = Path(path)
        if not _is_valid_local_model_dir(downloaded_path):
            raise RuntimeError(f"[vllm] DOWNLOADED_MODEL_INVALID: {repo_id} -> {downloaded_path}")

        return str(downloaded_path)
    except Exception as e:
        error_message = str(e)
        if _is_non_recoverable_download_error(error_message):
            raise RuntimeError(f"[vllm] DOWNLOAD_NON_RECOVERABLE: {model_id} ({repo_id}) -> {error_message}") from e
        raise RuntimeError(f"[vllm] DOWNLOAD_RETRYABLE: {model_id} ({repo_id}) -> {error_message}") from e


def _build_glossary_prompt(term_rules: Mapping[str, Any], target_languages: list[str]) -> str:
    if not isinstance(term_rules, Mapping):
        return ""

    is_active = bool(term_rules.get("is_active", False))
    llm_instruction = str(term_rules.get("llm_instruction", "")).strip()
    veto = term_rules.get("veto_validation", {})

    if not is_active and not llm_instruction:
        return ""

    lines: list[str] = []
    if llm_instruction:
        lines.append(f"Instruction: {llm_instruction}")

    if isinstance(veto, Mapping):
        expected = veto.get("multilingual_expected", {})
        forbidden = veto.get("multilingual_forbidden", {})

        for lang in target_languages:
            expected_words = []
            forbidden_words = []
            if isinstance(expected, Mapping):
                expected_words = [str(x) for x in expected.get(lang, [])]
            if isinstance(forbidden, Mapping):
                forbidden_words = [str(x) for x in forbidden.get(lang, [])]

            if expected_words:
                lines.append(f"{lang} expected keywords: {expected_words}")
            if forbidden_words:
                lines.append(f"{lang} forbidden keywords: {forbidden_words}")

    if not lines:
        return ""

    return "<GLOSSARY>\\nMANDATORY LOCALIZATION RULES:\\n" + "\\n".join(lines) + "\\n</GLOSSARY>\\n"


def _build_multilingual_prompt(
    source_text: str,
    test_block: str,
    term_rules: Mapping[str, Any],
    target_languages: list[str],
) -> str:
    language_list = ", ".join(target_languages)
    glossary = _build_glossary_prompt(term_rules=term_rules, target_languages=target_languages)
    
    # 构造强约束的示例 (Few-shot)
    example_input = "登录"
    example_output = json.dumps({lang: f"Translation of '登录' in {lang}" for lang in target_languages}, ensure_ascii=False)
    
    return (
        "<|im_start|>system\n"
        "You are a professional software localization expert. You translate UI strings accurately into multiple languages.\n"
        "STRICT RULES:\n"
        "1. Output ONLY a valid JSON object.\n"
        "2. NO explanations, NO preambles, NO text completion/continuation.\n"
        "3. Maintain the exact meaning of UI elements.<|im_end|>\n"
        "<|im_start|>user\n"
        "### TERMINOLOGY RULES\n"
        f"{glossary if glossary else 'No specific terminology constraints.'}\n\n"
        f"### TASK\n"
        f"Translate the Input String into these languages: {language_list}.\n\n"
        "### EXAMPLE\n"
        f"Input: {example_input}\n"
        f"Output: {example_output}\n\n"
        "### TARGET INPUT\n"
        f"Input: {source_text}\n"
        "Output JSON:<|im_end|>\n"
        "<|im_start|>assistant\n"
        "{"
    )


def build_ollama_translator(
    base_url: str,
    temperature: float,
    timeout: int,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    max_retries: int = 2,
):
    endpoint = base_url.rstrip("/") + "/api/generate"

    def _translator(
        model_id: str,
        source_text: str,
        test_block: str,
        dataset_item: Mapping[str, Any] | None = None,
    ):
        term_rules = dataset_item.get("term_rules", {}) if isinstance(dataset_item, Mapping) else {}
        prompt = _build_multilingual_prompt(
            source_text=source_text,
            test_block=test_block,
            term_rules=term_rules,
            target_languages=[target_language],
        )
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                raw = str(resp.json().get("response", "")).strip()
                if not raw:
                    raise ValueError("Empty response from Ollama endpoint.")

                parsed = robust_json_parser(raw)
                hypotheses = _normalize_hypotheses(parsed)
                if not hypotheses:
                    hypotheses = {target_language: raw}

                return {"hypotheses": hypotheses, "raw_llm_response": raw}
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(min(2.0, 0.5 * attempt))

        raise RuntimeError(f"Ollama translation failed after {max_retries} attempts: {last_error}")

    return _translator


class VLLMTranslator:
    """vLLM-backed translator that supports batch inference for throughput."""

    def __init__(
        self,
        target_languages: list[str],
        model_root: str = "/root/models",
        model_map: Mapping[str, str] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
        dynamic_gpu_utilization: bool = True,
        vram_reserve_gb: float = 8.0,
        vram_safety_margin_gb: float = 2.0,
        min_gpu_utilization: float = 0.55,
        max_gpu_utilization: float = 0.92,
    ) -> None:
        self.target_languages = list(target_languages)
        self.model_root = Path(model_root)
        self.model_map = {str(k).lower(): str(v) for k, v in dict(model_map or {}).items()}
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = int(max_model_len)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.enforce_eager = bool(enforce_eager)
        self.dynamic_gpu_utilization = bool(dynamic_gpu_utilization)
        self.vram_reserve_gb = float(vram_reserve_gb)
        self.vram_safety_margin_gb = float(vram_safety_margin_gb)
        self.min_gpu_utilization = float(min_gpu_utilization)
        self.max_gpu_utilization = float(max_gpu_utilization)

        self._llm: Any = None
        self._sampling_params: Any = None
        self._loaded_model_id = ""

    def register_local_model_path(self, model_id: str, local_model_path: str) -> None:
        candidate = Path(local_model_path)
        if not _is_valid_local_model_dir(candidate):
            return

        keys = {
            str(model_id).lower(),
            str(model_id).lower().replace("-", ":"),
            str(model_id).lower().replace(":", "-"),
            _normalize_model_key(model_id),
        }
        for key in keys:
            self.model_map[key] = str(candidate)

    @staticmethod
    def _filter_supported_engine_args(llm_cls: Any, engine_args: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Filter engine args against the installed vLLM signature.

        Different vLLM versions expose different kwargs. Passing unknown kwargs
        causes immediate, non-recoverable init failures.
        """
        try:
            signature = inspect.signature(llm_cls.__init__)
            params = signature.parameters
            accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
            if accepts_var_kwargs:
                return dict(engine_args), []

            allowed = {name for name in params.keys() if name != "self"}
            filtered = {key: value for key, value in engine_args.items() if key in allowed}
            dropped = sorted([key for key in engine_args.keys() if key not in allowed])
            return filtered, dropped
        except Exception:
            # Fall back to original args when introspection is unavailable.
            return dict(engine_args), []

    def _select_gpu_utilization(self, model_id: str) -> float:
        default_util = max(self.min_gpu_utilization, min(self.max_gpu_utilization, self.gpu_memory_utilization))

        if not self.dynamic_gpu_utilization:
            return default_util

        try:
            import torch
            cuda_module = getattr(torch, "cuda", None)
            if cuda_module is None: return default_util
            
            mem_get_info = getattr(cuda_module, "mem_get_info", None)
            if not callable(mem_get_info): return default_util

            mem_info_raw = mem_get_info()
            if not isinstance(mem_info_raw, (list, tuple)) or len(mem_info_raw) < 2:
                return default_util

            free_gb = float(mem_info_raw[0]) / (1024 ** 3)
            total_gb = max(float(mem_info_raw[1]) / (1024 ** 3), 1.0)

            reserve = max(0.0, float(self.vram_reserve_gb))
            safety_margin = max(0.0, float(self.vram_safety_margin_gb))
            usable_fraction = (free_gb - reserve - safety_margin) / total_gb

            dynamic_util = max(self.min_gpu_utilization, min(self.max_gpu_utilization, usable_fraction))

            print(
                f"[vllm] Dynamic GPU util for {model_id}: {dynamic_util:.3f} "
                f"(free={free_gb:.2f}GB, total={total_gb:.2f}GB, reserve={reserve:.2f}GB, margin={safety_margin:.2f}GB)"
            )
            return dynamic_util
        except Exception as exc:
            print(f"[vllm] Dynamic util fallback for {model_id}: {exc}")
            return default_util

    def _cleanup_model(self) -> None:
        had_model = self._llm is not None
        if had_model:
            print(f"[vllm] PHASING OUT Model: {self._loaded_model_id or 'unknown'} - Purging device state...")
            
            # 1. 物理销毁 vLLM 内部状态
            try:
                parallel_state = importlib.import_module("vllm.distributed.parallel_state")
                destroy_model_parallel = getattr(parallel_state, "destroy_model_parallel", None)
                destroy_distributed_runtime = getattr(parallel_state, "destroy_distributed_runtime", None)
                if callable(destroy_model_parallel):
                    destroy_model_parallel()
                try:
                    if callable(destroy_distributed_runtime):
                        destroy_distributed_runtime()
                except Exception:
                    pass
            except ImportError:
                 pass
            
            # 2. 彻底切断引擎引用
            if hasattr(self._llm, "llm_engine") and self._llm.llm_engine is not None:
                try:
                    # 某些版本支持 shutdown
                    if hasattr(self._llm.llm_engine, "shutdown"):
                        self._llm.llm_engine.shutdown()
                except: pass
            
            del self._llm
            self._llm = None
            self._loaded_model_id = ""

        # 3. 强制垃圾回收
        gc.collect()
        
        # 4. 彻底物理刷新 GPU 状态 (针对 A100 多卡环境尤为重要)
        try:
            import torch

            cuda_module = getattr(torch, "cuda", None)
            is_available = getattr(cuda_module, "is_available", None)
            if cuda_module is not None and callable(is_available) and is_available():
                synchronize = getattr(cuda_module, "synchronize", None)
                empty_cache = getattr(cuda_module, "empty_cache", None)
                ipc_collect = getattr(cuda_module, "ipc_collect", None)
                memory_allocated = getattr(cuda_module, "memory_allocated", None)
                # 显现同步，确保之前的 Kernel 指令流水线已清空
                if callable(synchronize):
                    synchronize()
                if callable(empty_cache):
                    empty_cache()
                if callable(ipc_collect):
                    ipc_collect()
                # 再次同步确保清理指令生效
                if callable(synchronize):
                    synchronize()
                usage = 0.0
                if callable(memory_allocated):
                    try:
                        usage_raw = memory_allocated()
                        if isinstance(usage_raw, (int, float)):
                            usage = float(usage_raw) / 1024**3
                    except Exception:
                        usage = 0.0
                print(f"[vllm] Explicitly released VRAM. Current allocation: {usage:.2f}GB")
        except Exception as e:
            print(f"[vllm] Device cleanup warning: {e}")

        # 5. 冷启动保护：物理静置 (减少等待，仅保留基础同步时间)
        if had_model:
            print("[vllm] Entering state sync period (1s)...")
            time.sleep(1)

    def close(self) -> None:
        self._cleanup_model()

    def _resolve_model_path(self, model_id: str) -> str:
        # 1. 自动标准化 ID 进行解析 (处理 qwen2.5:7b-instruct vs qwen2.5-7b-instruct)
        normalized_id = model_id.replace("-", ":") if ":" not in model_id else model_id
        lowered = normalized_id.lower()

        # 2. 检查模型映射表 (忽略大小写)
        map_keys = {
            lowered,
            model_id.lower(),
            model_id.lower().replace(":", "-"),
            model_id.lower().replace("-", ":"),
            _normalize_model_key(model_id),
        }
        for key in map_keys:
            mapped = self.model_map.get(key)
            if mapped and _is_valid_local_model_dir(Path(mapped)):
                print(f"[vllm] Resolved {model_id} via map -> {mapped}")
                return mapped

        # 3. 检查是否已经是绝对路径且存在
        if Path(model_id).is_absolute() and _is_valid_local_model_dir(Path(model_id)):
            return model_id

        # 4. 尝试各种常见的 ID 转路径格式
        # 例如: qwen2.5:7b-instruct -> qwen2.5-7b-instruct
        sanitized = model_id.replace(":", "-").replace("/", "_")
        
        candidates = [
            self.model_root / model_id,
            self.model_root / sanitized,
            # 兼容 Colab 常见的 /content 路径或 /root/.cache
            Path("/content/models") / sanitized,
            Path("/content/models") / model_id,
            Path("/root/.cache/huggingface/hub") / f"models--{model_id.replace(':', '--')}"
        ]
        
        for candidate in candidates:
            if _is_valid_local_model_dir(candidate):
                print(f"[vllm] Resolved {model_id} via probe -> {candidate}")
                return str(candidate)

        # 5. 如果都找不到，返回原 ID (交给 vLLM 尝试从 HF 下载，虽然在内网或受限环境可能失败)
        return model_id

    def _ensure_model(self, model_id: str) -> None:
        if self._llm is not None and self._loaded_model_id == model_id:
            return

        # 检查是否因为之前的 Batch 失败导致 _llm 被 close 变成了 None
        # 如果是同模型重载，应当保留对象而不是重新加载（除非显式清理）
        if self._llm is not None:
            self._cleanup_model()

        vllm_module = importlib.import_module("vllm")
        llm_cls = getattr(vllm_module, "LLM", None)
        sampling_cls = getattr(vllm_module, "SamplingParams", None)
        if llm_cls is None or sampling_cls is None:
            raise ImportError("vLLM is installed but required classes LLM/SamplingParams are unavailable")

        resolved = self._resolve_model_path(model_id)
        
        # 严格校验本地权重是否存在
        target_path = Path(resolved)
        if not _is_valid_local_model_dir(target_path):
             raise FileNotFoundError(
                 f"[vllm] CRITICAL: Strict Local Mode. Model dir invalid at: {resolved}. "
                 "Expected local directory with config.json."
             )

        utilization = self._select_gpu_utilization(model_id)
        lowered_model_id = model_id.lower()
        is_gemma_3 = "gemma-3" in lowered_model_id
        is_translategemma = "translategemma" in lowered_model_id
        is_google_model = lowered_model_id.startswith("google/")

        # TranslateGemma local snapshots may contain rope fields that are valid for
        # HF runtime but rejected by stricter vLLM/Pydantic validation.
        if is_translategemma:
            patched_ok = _patch_local_rope_parameters_config(target_path, model_id)
            if not patched_ok:
                raise RuntimeError(
                    f"[vllm] NON-RECOVERABLE TRANSLATEGEMMA CONFIG: "
                    f"{target_path / 'config.json'} still incompatible with current vLLM "
                    "(missing required rope_type in rope_parameters)."
                )
        
        # vLLM v0.17.1 (2026-03) 针对 Gemma-3 的专业适配逻辑
        # Gemma-3 属于多模态架构，但在我们纯文本翻译场景下，必须禁用所有会导致卡死的视觉算子冷加载
        engine_args = {
            "model": resolved,
            "trust_remote_code": (is_translategemma or (not is_google_model)),
            "gpu_memory_utilization": utilization,
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": self.enforce_eager, # A100(80G) 下多模态模型捕获 CUDA Graphs 会导致数分钟卡顿
            "disable_custom_all_reduce": True,
            "swap_space": 16,
            "disable_log_stats": True,
        }

        if is_translategemma:
            engine_args["dtype"] = "bfloat16"
            print("[vllm] TranslateGemma compatibility mode engaged (rope patch + bf16 + trust_remote_code).")

        if is_gemma_3:
            # 兼容 v0.17.1 (2026-03)：专门针对 A100-80G 的“纯文本”极致加速路径
            # 【核心优化】强制模型仅作为语言模型运行，封印视觉/多模态组件
            engine_args["language_model_only"] = True
            engine_args["dtype"] = "bfloat16"
            # 性能模式设定：由于是 Benchmark 吞吐量测试，设为 throughput
            engine_args["performance_mode"] = "throughput"
            # 显式告知不含多模态输入，跳过多模态处理队列初始化
            engine_args["limit_mm_per_prompt"] = {} 
            # 解决 Gemma-3 巨大的逻辑超时风险
            os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "300")
            print(f"[vllm] v0.17.1 BATTLE-HARDENED: Gemma-3 Multi-modal logic deactivated. Pure-LLM mode engaged.")

        safe_engine_args, dropped_engine_args = self._filter_supported_engine_args(llm_cls, engine_args)
        if dropped_engine_args:
            print(f"[vllm] Dropping unsupported EngineArgs for this vLLM build: {', '.join(dropped_engine_args)}")
        if "model" not in safe_engine_args:
            raise RuntimeError("[vllm] CRITICAL: filtered EngineArgs lost required key 'model'.")

        print(f"[vllm] CRITICAL: Loading {model_id} (util={utilization:.3f})")

        self._llm = llm_cls(**safe_engine_args)
        self._sampling_params = sampling_cls(
            temperature=self.temperature,
            top_p=0.9,
            max_tokens=self.max_tokens,
            repetition_penalty=1.1,
            stop=["\n\n", "###", "Reasoning", "<thought>"],
        )
        self._loaded_model_id = model_id

    def __call__(
        self,
        model_id: str,
        source_text: str,
        test_block: str,
        dataset_item: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 核心修复：在 single 调用时，绝不能重新触发模型检查逻辑导致重载
        # 确保只要 model_id 没变，就复用当前 self._llm
        if self._llm is None or self._loaded_model_id != model_id:
             self._ensure_model(model_id)

        item = dataset_item or {
            "test_id": "single",
            "source_text": source_text,
            "term_rules": normalize_term_rules(None),
        }
        
        # 直接构造 prompt 并调用 generate，不再绕道 translate_batch 防止递归
        target_languages = list(self.target_languages) or [DEFAULT_TARGET_LANGUAGE]
        prompt = _build_multilingual_prompt(
            source_text=source_text,
            test_block=test_block,
            term_rules=item.get("term_rules", {}),
            target_languages=target_languages,
        )
        
        outputs = self._llm.generate([prompt], self._sampling_params, use_tqdm=False)
        output = outputs[0]
        raw_text = str(output.outputs[0].text).strip() if output.outputs else ""
        
        # 同样的增强容错逻辑
        parsed = robust_json_parser(raw_text)
        hypotheses = _normalize_hypotheses(parsed)
        if not hypotheses and raw_text:
            hypotheses = {target_languages[0]: raw_text}
        
        # 确保至少返回一个非空结果，防止触发 "no non-empty hypotheses" 报错
        if not hypotheses:
             hypotheses = {target_languages[0]: "[EMPTY_RESPONSE]"}

        return {"hypotheses": hypotheses, "raw_llm_response": raw_text}

    def translate_batch(
        self,
        model_id: str,
        dataset_items: list[Mapping[str, Any]],
        test_block: str,
        inference_params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not dataset_items:
            return []

        self._ensure_model(model_id)

        params = dict(inference_params or {})
        # 修复逻辑：优先从传参获取 target_languages，且过滤掉空字符串，防止退化
        raw_langs = params.get("target_languages")
        if isinstance(raw_langs, list):
            target_languages = [str(x).strip() for x in raw_langs if str(x).strip()]
        else:
            target_languages = list(self.target_languages)

        # 兜底：如果依然为空，使用全局默认
        if not target_languages:
            target_languages = [DEFAULT_TARGET_LANGUAGE]

        prompts: list[str] = []

        if not target_languages:
            target_languages = list(self.target_languages) or [DEFAULT_TARGET_LANGUAGE]

        prompts: list[str] = []
        for item in dataset_items:
            source_text = str(item.get("source_text", ""))
            term_rules = normalize_term_rules(item.get("term_rules"))
            prompt = _build_multilingual_prompt(
                source_text=source_text,
                test_block=test_block,
                term_rules=term_rules,
                target_languages=target_languages,
            )
            prompts.append(prompt)

        outputs = self._llm.generate(prompts, self._sampling_params)

        rows: list[dict[str, Any]] = []
        for output in outputs:
            raw_text = ""
            if getattr(output, "outputs", None):
                raw_text = str(output.outputs[0].text).strip()

            hypotheses = {}
            # 增强解析防御逻辑
            if raw_text:
                parsed = robust_json_parser(raw_text)
                hypotheses = _normalize_hypotheses(parsed)
                
                # 特殊兜底：如果模型输出了内容但不是有效 JSON，或者 JSON 字段为空
                if not hypotheses:
                    # 优先填入目标语言列表中的第一个语言，避免返回 empty
                    primary_lang = target_languages[0] if target_languages else DEFAULT_TARGET_LANGUAGE
                    hypotheses = {primary_lang: raw_text if raw_text.strip() else "[EMPTY_OR_FAILED_PARSE]"}

            rows.append({
                "hypotheses": hypotheses,
                "raw_llm_response": raw_text or "[EMPTY_RESPONSE]",
            })

        return rows


def _parse_model_map(raw_value: str) -> dict[str, str]:
    if not raw_value.strip():
        return dict(DEFAULT_VLLM_MODEL_MAP)

    candidate_path = Path(raw_value).expanduser()
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, Mapping):
            raise ValueError("vllm-model-map file must contain a JSON object.")
        merged = dict(DEFAULT_VLLM_MODEL_MAP)
        merged.update({str(k): str(v) for k, v in payload.items()})
        return merged

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --vllm-model-map JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("--vllm-model-map must decode to JSON object.")

    merged = dict(DEFAULT_VLLM_MODEL_MAP)
    merged.update({str(k): str(v) for k, v in payload.items()})
    return merged


def build_batch_objective_metric_provider(
    metric_mode: str,
    preferred_language: str,
    comet_model_name: str,
    comet_batch_size: int,
):
    if metric_mode == "proxy":
        return None

    try:
        sacrebleu = importlib.import_module("sacrebleu")
    except Exception as exc:
        if metric_mode == "real":
            raise RuntimeError(f"sacrebleu is required for real metrics: {exc}") from exc
        print(f"[metrics] sacrebleu unavailable, fallback to proxy: {exc}")
        return None

    comet_model = None
    torch_module = None
    comet_initialized = False

    def _ensure_comet_loaded() -> None:
        nonlocal comet_model, torch_module, comet_initialized
        if comet_initialized:
            return
        comet_initialized = True

        if metric_mode not in {"real", "auto"}:
            return

        try:
            comet_module = importlib.import_module("comet")
            torch_module = importlib.import_module("torch")
            comet_download_model = getattr(comet_module, "download_model")
            comet_load_from_checkpoint = getattr(comet_module, "load_from_checkpoint")

            checkpoint_path = comet_download_model(comet_model_name)
            comet_model = comet_load_from_checkpoint(checkpoint_path)
            comet_model = comet_model.to("cuda" if _torch_cuda_available(torch_module) else "cpu")
            print(f"[metrics] COMET loaded lazily: {comet_model_name}")
        except Exception as exc:
            if metric_mode == "real":
                raise RuntimeError(f"Failed to load COMET model: {exc}") from exc
            print(f"[metrics] COMET unavailable, fallback to chrF++ only: {exc}")
            comet_model = None
            torch_module = None

    def _provider(
        dataset_by_test_id: Mapping[str, Mapping[str, Any]],
        raw_rows: list[Mapping[str, Any]],
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        comet_inputs: list[dict[str, str]] = []
        comet_test_ids: list[str] = []

        for row in raw_rows:
            test_id = str(row.get("test_id", ""))
            dataset_item = dataset_by_test_id.get(test_id, {})
            status = str(row.get("status", ""))

            if status != "SUCCESS":
                result[test_id] = {"chrf_score": 0.0, "comet_score": 0.0}
                continue

            hyp = _extract_hypothesis_text(row, preferred_language)
            ref = _extract_reference_text(dataset_item, preferred_language)
            src = str(dataset_item.get("source_text", "")).strip()

            if not hyp:
                result[test_id] = {"chrf_score": 0.0, "comet_score": 0.0}
                continue

            if not ref:
                result[test_id] = {"chrf_score": 65.0, "comet_score": 65.0}
                continue

            chrf_score = float(sacrebleu.sentence_chrf(hyp, [ref]).score)
            result[test_id] = {"chrf_score": round(chrf_score, 4), "comet_score": 0.0}

            if metric_mode in {"real", "auto"}:
                comet_inputs.append({"src": src, "mt": hyp, "ref": ref})
                comet_test_ids.append(test_id)

        if comet_inputs:
            _ensure_comet_loaded()

        if comet_model is not None and comet_inputs and torch_module is not None:
            cuda_enabled = _torch_cuda_available(torch_module)
            preds = comet_model.predict(
                comet_inputs,
                batch_size=int(comet_batch_size),
                gpus=1 if cuda_enabled else 0,
            )

            for idx, score in enumerate(preds.scores):
                test_id = comet_test_ids[idx]
                if test_id in result:
                    result[test_id]["comet_score"] = round(float(score) * 100.0, 4)
        else:
            for score_row in result.values():
                if score_row.get("comet_score", 0.0) == 0.0 and score_row.get("chrf_score", 0.0) > 0.0:
                    score_row["comet_score"] = score_row["chrf_score"]

        return result

    return _provider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundled B2 runner for local/Colab")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_15_MODEL_REPOS))
    parser.add_argument("--workspace-root", type=str, default=str(Path.cwd()))
    parser.add_argument("--metadata-root", type=str, default="")
    parser.add_argument("--benchmark-root", type=str, default="")

    # --- 流程解耦参数 ---
    parser.add_argument("--stage", type=str, choices=["all", "inference", "audit"], default="all")
    
    parser.add_argument("--translator", type=str, choices=["mock", "ollama", "vllm"], default="mock")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    # 对于 vLLM 这种支持 Continuous Batching 的后端，这里的默认值会被内部逻辑覆盖为极大的数（如 100,000）
    # 这里的 128 主要为 Ollama 等不支持大规模全量并发的后端保留。
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--target-language", type=str, default=DEFAULT_TARGET_LANGUAGE)
    parser.add_argument("--target-languages", type=str, default=",".join(TARGET_LANGUAGES))

    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--warmup", action="store_true", help="Warm up model before block execution")
    parser.add_argument("--healthcheck", action="store_true", help="Check Ollama endpoint health before run")

    parser.add_argument("--vllm-model-root", type=str, default="/root/models")
    parser.add_argument("--vllm-model-map", type=str, default="")
    parser.add_argument("--vllm-max-tokens", type=int, default=1024)
    parser.add_argument("--vllm-gpu-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-dynamic-gpu-utilization", dest="vllm_dynamic_gpu_utilization", action="store_true")
    parser.add_argument("--no-vllm-dynamic-gpu-utilization", dest="vllm_dynamic_gpu_utilization", action="store_false")
    parser.add_argument("--vllm-vram-reserve-gb", type=float, default=0.0)
    parser.add_argument("--vllm-vram-safety-margin-gb", type=float, default=0.5)
    parser.add_argument("--vllm-min-gpu-utilization", type=float, default=0.55)
    parser.add_argument("--vllm-max-gpu-utilization", type=float, default=0.96)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-enforce-eager", dest="vllm_enforce_eager", action="store_true")
    parser.add_argument("--no-vllm-enforce-eager", dest="vllm_enforce_eager", action="store_false")
    parser.set_defaults(
        vllm_enforce_eager=True,
        vllm_dynamic_gpu_utilization=True,
        repair_raw_before_audit=True,
    )

    parser.add_argument("--metric-mode", type=str, choices=["auto", "proxy", "real"], default="auto")
    parser.add_argument("--objective-language", type=str, default="en")
    parser.add_argument("--comet-model", type=str, default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-batch-size", type=int, default=128)

    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately on model/block errors")
    parser.add_argument("--force", action="store_true", help="Force re-run: ignore existing checkpoints and raw outputs")
    parser.add_argument(
        "--audit-testid-check",
        type=str,
        choices=["off", "warn", "error"],
        default="error",
        help="Check test_id alignment between dataset and raw rows before audit scoring.",
    )
    parser.add_argument(
        "--audit-min-testid-match-rate",
        type=float,
        default=0.99,
        help="Minimum required matched test_id ratio for audit scoring (0-1).",
    )
    parser.add_argument(
        "--force-rebuild-datasets-in-audit",
        action="store_true",
        help="When stage=audit and --force is set, purge 01_Datasets to force metadata rebuild.",
    )
    parser.add_argument(
        "--repair-raw-before-audit",
        dest="repair_raw_before_audit",
        action="store_true",
        help="Repair raw inference JSON structure in-place before audit scoring",
    )
    parser.add_argument(
        "--no-repair-raw-before-audit",
        dest="repair_raw_before_audit",
        action="store_false",
        help="Disable in-place raw repair before audit",
    )

    # In Notebook environments, if no explicit argv is provided, 
    # we must default to an empty list to avoid parsing Colab's internal -f arguments.
    if argv is None:
        if _is_notebook_runtime():
            print("[args] Notebook detected, using empty argv to avoid -f conflict.")
            argv = []
        else:
            argv = sys.argv[1:]

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[args] ignored unknown arguments: {unknown}")
    
    # 强制将 models 覆盖（如果 argv 为空但提供了 models 参数）
    if not argv and not args.models:
        # 这个逻辑防止在 Notebook 里默认跑错模型
        pass
        
    return args


def run_bundle(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    workspace_root = Path(args.workspace_root).resolve()

    metadata_root = Path(args.metadata_root).resolve() if args.metadata_root else _default_metadata_root(workspace_root)
    benchmark_root = Path(args.benchmark_root).resolve() if args.benchmark_root else _default_benchmark_root(workspace_root)
    
    # 核心修复：支持 'default' 关键字自动映射到预定义的 15 模型列表
    raw_models = _parse_csv_arg(args.models)
    if len(raw_models) == 1 and raw_models[0].lower() == "default":
        model_ids = _canonicalize_model_ids(list(DEFAULT_15_MODEL_REPOS))
    else:
        model_ids = _canonicalize_model_ids(raw_models)

    if not model_ids:
        raise ValueError("No valid model IDs provided. Use --models modelA,modelB or --models default")
    print(f"[models] Canonicalized model list ({len(model_ids)}): {model_ids}")

    target_languages = _parse_csv_arg(args.target_languages)
    if not target_languages:
        target_languages = list(TARGET_LANGUAGES)

    # 路径安全化：仅在包含 inference 阶段且 translator=mock 时，才进入 mock 隔离目录。
    # audit-only 阶段需要读取真实 Benchmarks 导出目录，不应偏移到 .mock_results。
    is_mock_run = (args.stage in ("all", "inference") and args.translator == "mock")
    paths = ArenaPaths(benchmark_root, is_mock=is_mock_run)
    manager = ExperimentDirectoryManager(paths)
    
    # --force 在不同 stage 下应采用不同清理策略，避免 audit-only 误删 raw 产物。
    if args.force:
        if args.stage in ("all", "inference"):
            print("[force] Triple-purge: Clearing dataset, checkpoints, and experiment runs...")
            # 1. 清理 Datasets 目录，确保脚本逻辑更新后，数据集也会重新生成
            if paths.datasets_dir.exists():
                try:
                    shutil.rmtree(paths.datasets_dir)
                    print("[force] Physical purge of datasets directory complete.")
                except Exception as e:
                    print(f"[force] Dataset cleanup warning: {e}")

            # 2. 清理当前运行记录
            if paths.runs_dir.exists():
                try:
                    shutil.rmtree(paths.runs_dir)
                    print(f"[force] Physical purge of {paths.runs_dir} complete.")
                except Exception as e:
                    print(f"[force] Run cleanup warning: {e}")
        else:
            print(
                "[force] Audit-only mode detected: preserving 02_Experiment_Runs raw outputs; "
                "audit reports will be regenerated per block."
            )
            if args.force_rebuild_datasets_in_audit and paths.datasets_dir.exists():
                try:
                    shutil.rmtree(paths.datasets_dir)
                    print(
                        "[force] Audit-only option enabled: purged 01_Datasets to force metadata rebuild."
                    )
                except Exception as e:
                    print(f"[force] Audit-only dataset purge warning: {e}")

            summary_path = manager.global_summary_path()
            if summary_path.exists():
                try:
                    summary_path.unlink()
                    print(f"[force] Removed stale leaderboard file: {summary_path}")
                except Exception as e:
                    print(f"[force] Leaderboard cleanup warning: {e}")

        # 释放显存
        gc.collect()
        try:
            import torch
            cuda_module = getattr(torch, "cuda", None)
            is_available = getattr(cuda_module, "is_available", None)
            empty_cache = getattr(cuda_module, "empty_cache", None)
            if cuda_module is not None and callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            pass

    manager.ensure_base_tree()

    env_status = _runtime_environment_status()
    print("[env]", json.dumps(env_status, ensure_ascii=False))

    run_manifest = {
        "schema_version": B2_SCHEMA_VERSION,
        "started_at": _runtime_now_iso(),
        "workspace_root": str(workspace_root),
        "metadata_root": str(metadata_root),
        "benchmark_root": str(benchmark_root),
        "models": model_ids,
        "test_blocks": list(TEST_BLOCKS),
        "environment": env_status,
        "args": {
            "translator": args.translator,
            "temperature": args.temperature,
            "timeout": args.timeout,
            "target_language": args.target_language,
            "target_languages": target_languages,
            "max_retries": args.max_retries,
            "batch_size": args.batch_size,
            "warmup": bool(args.warmup),
            "healthcheck": bool(args.healthcheck),
            "stop_on_error": bool(args.stop_on_error),
            "limit": args.limit,
            "vllm_dynamic_gpu_utilization": bool(args.vllm_dynamic_gpu_utilization),
            "vllm_vram_reserve_gb": args.vllm_vram_reserve_gb,
            "vllm_vram_safety_margin_gb": args.vllm_vram_safety_margin_gb,
            "vllm_min_gpu_utilization": args.vllm_min_gpu_utilization,
            "vllm_max_gpu_utilization": args.vllm_max_gpu_utilization,
            "metric_mode": args.metric_mode,
            "objective_language": args.objective_language,
            "comet_model": args.comet_model,
            "comet_batch_size": args.comet_batch_size,
            "audit_testid_check": args.audit_testid_check,
            "audit_min_testid_match_rate": args.audit_min_testid_match_rate,
            "force_rebuild_datasets_in_audit": bool(args.force_rebuild_datasets_in_audit),
            "repair_raw_before_audit": bool(args.repair_raw_before_audit),
        },
    }
    _runtime_save_json(manager.run_manifest_path(), run_manifest)

    dataset_map: dict[str, Path] = {}
    if args.stage == "audit" and not args.force_rebuild_datasets_in_audit:
        existing_dataset_map = _resolve_existing_dataset_map(paths)
        missing_blocks = [block for block in TEST_BLOCKS if block not in existing_dataset_map]
        if not missing_blocks:
            dataset_map = existing_dataset_map
            print(
                f"[1/5] Reusing existing datasets for audit stage from: {paths.datasets_dir} "
                f"(skip metadata rebuild; limit ignored={args.limit})"
            )
        else:
            print(
                f"[dataset] Missing existing dataset files for blocks {missing_blocks}. "
                "Fallback to metadata rebuild."
            )
    elif args.stage == "audit" and args.force_rebuild_datasets_in_audit:
        print("[dataset] Audit-only dataset rebuild forced by --force-rebuild-datasets-in-audit.")

    if not dataset_map:
        # Check if we should ignore the limit because we're forcing a rebuild for audit
        effective_limit = 0 if (args.stage == "audit" and args.force_rebuild_datasets_in_audit) else args.limit
        
        if effective_limit:
            print(f"[1/5] Build normalized datasets from: {metadata_root} (limit={effective_limit})")
        else:
            print(f"[1/5] Build normalized datasets from: {metadata_root} (full mode)")
            
        dataset_map = build_all_standard_datasets(
            metadata_root=metadata_root,
            datasets_dir=paths.datasets_dir,
            limit=effective_limit,
        )

    dataset_test_ids_by_block: dict[str, set[str]] = {}
    # 增加数据集条数诊断日志
    for block, path in dataset_map.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                count = len(data.get("items", []))
                dataset_test_ids = {
                    str(item.get("test_id", "")).strip()
                    for item in data.get("items", [])
                    if isinstance(item, Mapping) and str(item.get("test_id", "")).strip()
                }
                dataset_test_ids_by_block[block] = dataset_test_ids
                print(
                    f"[dataset] Block '{block}' preparation complete: {count} items found "
                    f"(unique_test_ids={len(dataset_test_ids)})."
                )
        except Exception as e:
            print(f"[dataset] Warning: Could not verify count for {block}: {e}")
            dataset_test_ids_by_block[block] = set()
            
    run_manifest["datasets"] = {block: str(path) for block, path in dataset_map.items()}
    _runtime_save_json(manager.run_manifest_path(), run_manifest)

    resolved_model_map: dict[str, str] = {}
    translator: Any = None
    runner: InferenceRunner | None = None

    if args.stage in ("all", "inference"):
        if args.translator == "ollama":
            if args.healthcheck:
                _ollama_healthcheck(base_url=args.ollama_url, timeout=args.timeout)

            translator = build_ollama_translator(
                base_url=args.ollama_url,
                temperature=args.temperature,
                timeout=args.timeout,
                target_language=args.target_language,
                max_retries=args.max_retries,
            )
        elif args.translator == "vllm":
            if not _module_available("vllm"):
                 print("[vllm] CRITICAL: vllm module not found. Is it installed? Falling back to mock.")
                 translator = mock_translator
            else:
                resolved_model_map = _parse_model_map(args.vllm_model_map)
                print(f"[vllm] Init VLLMTranslator with {len(target_languages)} target languages.")
                translator = VLLMTranslator(
                    target_languages=target_languages,
                    model_root=args.vllm_model_root,
                    model_map=resolved_model_map,
                    temperature=args.temperature,
                    max_tokens=args.vllm_max_tokens,
                    gpu_memory_utilization=args.vllm_gpu_utilization,
                    max_model_len=args.vllm_max_model_len,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    enforce_eager=args.vllm_enforce_eager,
                    dynamic_gpu_utilization=args.vllm_dynamic_gpu_utilization,
                    vram_reserve_gb=args.vllm_vram_reserve_gb,
                    vram_safety_margin_gb=args.vllm_vram_safety_margin_gb,
                    min_gpu_utilization=args.vllm_min_gpu_utilization,
                    max_gpu_utilization=args.vllm_max_gpu_utilization,
                )
        else:
            print(f"[translator] Falling back to {args.translator} translator.")
            translator = mock_translator

        runner = InferenceRunner(translator=translator)
    else:
        print("[stage] Audit-only mode: skip translator initialization.")

    objective_metric_batch_provider = None

    evaluator = AuditEvaluator()
    print(f"[2/5] Inference for models: {model_ids}")

    all_errors: list[dict[str, Any]] = []
    audit_stats: dict[str, int] = {
        "models_total": len(model_ids),
        "models_with_dir": 0,
        "models_audited": 0,
        "blocks_audited": 0,
        "blocks_alignment_failed": 0,
        "blocks_missing_raw": 0,
        "blocks_missing_dataset": 0,
        "rows_alignment_checked": 0,
        "rows_alignment_matched": 0,
    }

    try:
        # 第一阶段：运行推理 (仅在 stage 为 all 或 inference 时运行)
        if args.stage in ("all", "inference"):
            for model_id in model_ids:
                manager.ensure_model_tree(model_id)
                if args.translator == "vllm":
                    try:
                        local_model_path = _ensure_model_downloaded(
                            model_id,
                            Path(args.vllm_model_root),
                            resolved_model_map,
                        )
                        if hasattr(translator, "register_local_model_path"):
                            translator.register_local_model_path(model_id, local_model_path)
                    except Exception as exc:
                        message = f"[error] Skip model {model_id}: preparation failed -> {exc}"
                        print(message)
                        all_errors.append({"stage": "model_prepare", "model_id": model_id, "error": str(exc)})
                        if args.stop_on_error:
                            raise
                        continue

                for block in TEST_BLOCKS:
                    dataset_path = dataset_map.get(block)
                    if not dataset_path:
                        continue
                    raw_path = manager.raw_output_path(model_id, block)
                    checkpoint_path = manager.checkpoint_path(model_id, block)

                    # Initialize Checkpoint store and check for translator type mismatch (Mock vs vLLM)
                    store = CheckpointStore(checkpoint_path, expected_translator=args.translator)

                    if args.force:
                        if checkpoint_path.exists(): checkpoint_path.unlink()
                        store.clear()
                        if raw_path.exists(): raw_path.unlink()
                    
                    try:
                        effective_batch_size = _auto_batch_size_for_model(
                            model_id=model_id,
                            requested=args.batch_size,
                            translator_name=args.translator,
                        )
                        
                        if runner is None:
                            raise RuntimeError("Inference runner is unavailable in current stage")

                        runner.run_block(
                            model_id=model_id,
                            test_block=block,
                            dataset_path=dataset_path,
                            output_path=raw_path,
                            checkpoint_path=store,
                            inference_params={
                                "batch_size": effective_batch_size, 
                                "translator": args.translator, 
                                "target_languages": target_languages
                            },
                        )
                    except Exception as exc:
                        print(f"[error] Model {model_id} block {block} failed: {exc}")
                        all_errors.append(
                            {
                                "stage": "inference",
                                "model_id": model_id,
                                "test_block": block,
                                "error": str(exc),
                            }
                        )
                        if args.stop_on_error:
                            raise
                
                # --- 模型轮替物理切断 ---
                if args.translator == "vllm" and hasattr(translator, "close") and callable(getattr(translator, "close")):
                    print(f"[vllm] Cycle finished for: {model_id}. Initiating deep reset...")
                    translator.close()
                    gc.collect()
                time.sleep(2)
        else:
            print("[stage] Skipping inference stage (manual override).")

        # 第二阶段：推理结束后，主动关闭 Translator 彻底释放 GPU 显存，然后启动评分
        if translator is not None and hasattr(translator, "close"):
            translator.close()

        # 第三阶段：运行审计评分 (仅在 stage 为 all 或 audit 时运行)
        if args.stage in ("all", "audit"):
            objective_metric_batch_provider = build_batch_objective_metric_provider(
                metric_mode=args.metric_mode,
                preferred_language=args.objective_language,
                comet_model_name=args.comet_model,
                comet_batch_size=args.comet_batch_size,
            )
            
            print(f"[3.5/5] Executing stage: '{args.stage}'. Starting score audit...")
            
            for model_id in model_ids:
                model_dir_for_audit = _resolve_model_dir_for_audit(paths, model_id)
                
                # 诊断日志：显示最终选定的目录
                if model_dir_for_audit.exists() and model_dir_for_audit.is_dir():
                    audit_stats["models_with_dir"] += 1
                    print(f"[audit] Scanning model: {model_id} via directory: {model_dir_for_audit.name}")
                else:
                    # 获取该 model_id 的所有候选目录名用于调试
                    candidates = _candidate_model_dir_names(model_id)
                    print(f"[audit] Skipping {model_id}: no valid directory found in {paths.runs_dir}")
                    print(f"      (Checked candidates: {', '.join(candidates)})")
                    continue

                model_audited_blocks = 0

                for block in TEST_BLOCKS:
                    dataset_path = dataset_map.get(block)
                    if not dataset_path:
                        audit_stats["blocks_missing_dataset"] += 1
                        print(f"[audit] Skipping {model_id} {block}: dataset not prepared.")
                        continue

                    raw_path = _resolve_raw_path_for_block(model_dir_for_audit, block)
                    audit_path = model_dir_for_audit / "audited_reports" / f"{block}_audit.json"
                    audit_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if args.force and audit_path.exists():
                        audit_path.unlink()
                    
                    if raw_path is None:
                        audit_stats["blocks_missing_raw"] += 1
                        expected_raw = model_dir_for_audit / "raw_inference" / f"{block}_raw.json"
                        print(f"[audit] Skipping {model_id} {block}: raw file missing ({expected_raw.name}).")
                        continue

                    if raw_path.name != f"{block}_raw.json":
                        print(
                            f"[audit] Resolved non-standard raw filename for {model_id} {block}: {raw_path.name}"
                        )

                    dataset_test_ids = dataset_test_ids_by_block.get(block)
                    if dataset_test_ids is None:
                        dataset_test_ids = _collect_dataset_test_ids(dataset_path)
                        dataset_test_ids_by_block[block] = dataset_test_ids

                    alignment = _inspect_raw_test_id_alignment(raw_path, dataset_test_ids)
                    audit_stats["rows_alignment_checked"] += int(alignment["rows_total"])
                    audit_stats["rows_alignment_matched"] += int(alignment["rows_matched"])

                    if alignment["rows_total"] > 0:
                        print(
                            f"[audit] test_id coverage {model_id} {block}: "
                            f"matched={alignment['rows_matched']}/{alignment['rows_total']} "
                            f"({alignment['matched_rate'] * 100:.2f}%)"
                        )

                    min_match_rate = min(max(float(args.audit_min_testid_match_rate), 0.0), 1.0)
                    if (
                        args.audit_testid_check != "off"
                        and alignment["rows_total"] > 0
                        and alignment["matched_rate"] < min_match_rate
                    ):
                        audit_stats["blocks_alignment_failed"] += 1
                        mismatch_message = (
                            f"test_id alignment below threshold for {model_id} {block}: "
                            f"matched={alignment['rows_matched']}/{alignment['rows_total']} "
                            f"({alignment['matched_rate'] * 100:.2f}%), "
                            f"required>={min_match_rate * 100:.2f}%, "
                            f"sample_unmatched={alignment['unmatched_examples']}"
                        )
                        if args.audit_testid_check == "error":
                            print(f"[audit] HARD-FAIL: {mismatch_message}")
                            all_errors.append(
                                {
                                    "stage": "audit_alignment",
                                    "model_id": model_id,
                                    "test_block": block,
                                    "error": mismatch_message,
                                }
                            )
                            if args.stop_on_error:
                                raise RuntimeError(mismatch_message)
                            continue

                        print(f"[audit] WARNING: {mismatch_message}")

                    if args.repair_raw_before_audit:
                        try:
                            repair_stats = repair_raw_inference_file(
                                raw_inference_path=raw_path,
                                target_languages=target_languages,
                            )
                            changed_rows = int(repair_stats.get("rows_changed", 0))
                            if changed_rows > 0:
                                repaired_raw = int(repair_stats.get("rows_repaired_from_raw", 0))
                                repaired_fragment = int(
                                    repair_stats.get("rows_repaired_from_default_language", 0)
                                )
                                print(
                                    f"[repair] {model_id} {block}: repaired {changed_rows}/"
                                    f"{repair_stats.get('rows_total', 0)} rows "
                                    f"(raw={repaired_raw}, default-fragment={repaired_fragment})"
                                )
                        except Exception as exc:
                            print(f"[repair] Raw repair for {model_id} {block} failed: {exc}")
                            all_errors.append(
                                {
                                    "stage": "repair_raw",
                                    "model_id": model_id,
                                    "test_block": block,
                                    "error": str(exc),
                                }
                            )
                            if args.stop_on_error:
                                raise

                    try:
                        print(f"[audit] Auditing {model_id} {block} from {raw_path.name}")
                        evaluator.audit_block(
                            raw_inference_path=raw_path,
                            dataset_path=dataset_path,
                            output_path=audit_path,
                            objective_metric_batch_provider=objective_metric_batch_provider,
                            score_spec_overrides={
                                "metric_mode": args.metric_mode,
                                "comet_model": args.comet_model,
                                "objective_language": args.objective_language,
                            },
                        )
                        audit_stats["blocks_audited"] += 1
                        model_audited_blocks += 1
                    except Exception as exc:
                        print(f"[error] Audit for {model_id} {block} failed: {exc}")
                        all_errors.append(
                            {
                                "stage": "audit",
                                "model_id": model_id,
                                "test_block": block,
                                "error": str(exc),
                            }
                        )
                        if args.stop_on_error:
                            raise

                if model_audited_blocks > 0:
                    audit_stats["models_audited"] += 1
        else:
            print("[stage] Skipping audit stage (manual override).")

        if args.stage in ("all", "audit"):
            print(
                "[audit] Summary: "
                f"models_with_dir={audit_stats['models_with_dir']}/{audit_stats['models_total']}, "
                f"models_audited={audit_stats['models_audited']}, "
                f"blocks_audited={audit_stats['blocks_audited']}, "
                f"alignment_failed={audit_stats['blocks_alignment_failed']}, "
                f"missing_raw={audit_stats['blocks_missing_raw']}, "
                f"missing_dataset={audit_stats['blocks_missing_dataset']}"
            )
            checked_rows = audit_stats["rows_alignment_checked"]
            matched_rows = audit_stats["rows_alignment_matched"]
            matched_pct = 100.0 if checked_rows == 0 else (matched_rows / checked_rows) * 100.0
            print(
                f"[audit] Global test_id coverage: matched={matched_rows}/{checked_rows} "
                f"({matched_pct:.2f}%)"
            )

            if args.audit_testid_check == "error" and audit_stats["blocks_alignment_failed"] > 0:
                raise RuntimeError(
                    "Audit blocked by test_id alignment guard. "
                    "Regenerate full datasets (avoid stale/truncated 01_Datasets) and rerun audit."
                )

            if audit_stats["blocks_audited"] == 0:
                print(
                    "[audit] WARNING: No block was audited. "
                    "Please verify /02_Experiment_Runs/<model>/raw_inference/*_raw.json structure."
                )

    finally:
        if translator is not None and hasattr(translator, "close") and callable(getattr(translator, "close")):
            translator.close()

    print("[3/5] Build global leaderboard")
    aggregator = GlobalMetricsAggregator(manager)
    leaderboard = aggregator.aggregate()

    leaderboard_rows_any = leaderboard.get("models")
    if not isinstance(leaderboard_rows_any, list):
        fallback_rows = leaderboard.get("leaderboard")
        if isinstance(fallback_rows, list):
            leaderboard_rows_any = fallback_rows
    if not isinstance(leaderboard_rows_any, list):
        leaderboard_rows_any = []

    leaderboard_rows: list[dict[str, Any]] = []
    for row in leaderboard_rows_any:
        if isinstance(row, Mapping):
            leaderboard_rows.append(dict(row))

    leaderboard["models"] = leaderboard_rows
    leaderboard["leaderboard"] = leaderboard_rows

    print(f"[3.5/5] Re-mapping models for summary: {model_ids}")
    final_models = []
    # 建立 model_id 到 model_row 的映射以便排序
    models_dict = {m.get("model_id"): m for m in leaderboard_rows if m.get("model_id")}
    
    # 按命令行传入的 models 顺序重新构建列表（仅保留存在的）
    for mid in model_ids:
        if mid in models_dict:
            final_models.append(models_dict[mid])
        else:
            # 兼容性查找：尝试 sanitize 后的 ID
            sanitized = sanitize_model_id(mid)
            found = False
            for row in leaderboard_rows:
                if row.get("model_folder") == sanitized:
                    final_models.append(row)
                    found = True
                    break
            if not found:
                 # 依然没找到，说明没跑过，跳过
                 pass
    
    # 更新聚合结果中的模型列表（保持输入顺序）
    if final_models:
        leaderboard["models"] = final_models
        leaderboard["leaderboard"] = final_models
        leaderboard["model_count"] = len(final_models)
    else:
        leaderboard["model_count"] = len(leaderboard_rows)

    run_manifest["completed_at"] = _runtime_now_iso()
    run_manifest["leaderboard_path"] = str(manager.global_summary_path())
    run_manifest["leaderboard_model_count"] = int(leaderboard.get("model_count", len(leaderboard.get("models", []))))
    run_manifest["error_event_count"] = len(all_errors)
    run_manifest["audit_stats"] = audit_stats
    _runtime_save_json(manager.run_manifest_path(), run_manifest)

    print("[4/5] Outputs")
    print(f"Datasets: {paths.datasets_dir}")
    print(f"Runs: {paths.runs_dir}")
    print(f"Leaderboard: {manager.global_summary_path()}")
    print(f"Model count: {leaderboard.get('model_count', 0)}")
    print(f"Error events: {len(all_errors)}")

    print("[5/5] Done")


if __name__ == "__main__":
    run_bundle()

"""=============================="""