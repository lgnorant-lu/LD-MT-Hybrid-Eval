from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import JudgeSlotConfig, PollConfig
from .types import JudgeDecision, JudgeError
from .logging_utils import get_logger
from .tracer import DiagnosticTracer

logger = logging.getLogger("local_poll_mqm")

SYSTEM_PROMPT = (
    "You are a strict MQM evaluator. "
    "Return JSON only. No markdown, no explanations. "
    "Analyze the text enclosed in <hypothesis> against <source> and <reference>."
)


@dataclass
class JudgeTask:
    test_id: str
    source_text: str
    hypothesis_text: str
    reference_text: str
    target_language: str


def _extract_json_text(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw[start : end + 1]


def _build_user_prompt(task: JudgeTask) -> str:
    return (
        "Evaluate translation quality with MQM style error extraction.\n"
        'Return JSON: {"errors": [{"span": str, "severity": "minor|major|critical", '
        '"category": "accuracy|fluency|terminology|style|other", "reason": str}]}\n'
        'If no issue, return {"errors": []}.\n\n'
        f"Test ID: {task.test_id}\n"
        f"Target Language: {task.target_language}\n"
        f"<source>\n{task.source_text}\n</source>\n\n"
        f"<hypothesis>\n{task.hypothesis_text}\n</hypothesis>\n\n"
        f"<reference>\n{task.reference_text}\n</reference>\n"
    )


def _build_batch_user_prompt(tasks: list[JudgeTask]) -> str:
    head = (
        "Evaluate translation quality with MQM style error extraction for each item independently.\n"
        "Return JSON only with this schema exactly:\n"
        '{"results": [{"test_id": str, "errors": [{"span": str, "severity": "minor|major|critical", '
        '"category": "accuracy|fluency|terminology|style|other", "reason": str}]}]}\n'
        "If an item has no issue, set errors to [].\n\n"
    )
    chunks: list[str] = [head]
    for i, task in enumerate(tasks, start=1):
        chunks.append(
            (
                f"--- Item {i} ---\n"
                f"Test ID: {task.test_id}\n"
                f"Target Language: {task.target_language}\n"
                f"<source>\n{task.source_text}\n</source>\n\n"
                f"<hypothesis>\n{task.hypothesis_text}\n</hypothesis>\n\n"
                f"<reference>\n{task.reference_text}\n</reference>\n\n"
            )
        )
    return "".join(chunks)





def _parse_errors_payload(errors_payload: Any, judge_id: str) -> list[JudgeError]:
    errors: list[JudgeError] = []
    if not isinstance(errors_payload, list):
        return errors

    for item in errors_payload:
        if not isinstance(item, dict):
            continue
        span = str(item.get("span", "")).strip()
        if not span:
            continue
        severity = str(item.get("severity", "minor")).strip().lower()
        category = str(item.get("category", "other")).strip().lower()
        reason = str(item.get("reason", "")).strip()
        errors.append(
            JudgeError(
                span=span,
                severity=severity,
                category=category,
                reason=reason,
                judge_id=judge_id,
            )
        )
    return errors


class AsyncJudgeClient:
    """Async judge client using httpx for HTTP/2 support."""
    active_batches: int = 0
    _active_lock = asyncio.Lock()

    def __init__(
        self,
        slot: JudgeSlotConfig,
        poll_config: PollConfig,
        session: httpx.AsyncClient,
        semaphore: asyncio.Semaphore | None = None,
        tracer: DiagnosticTracer | None = None,
    ) -> None:
        self.slot = slot
        self.poll_config = poll_config
        self.session = session
        self.semaphore = semaphore
        self.tracer = tracer

    def _api_key(self) -> str:
        return os.environ.get(self.slot.api_key_env, "").strip()

    def _url(self) -> str:
        base = self.slot.endpoint.rstrip("/")
        if self.slot.provider == "google":
            return f"{base}/v1beta/models/{self.slot.model}:generateContent"
        return base + "/chat/completions"

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
        }
        if self.slot.provider == "google":
            headers["x-goog-api-key"] = api_key
        return headers

    def _validate_raw_text(self, text: str) -> tuple[bool, str]:
        """
        [Standardized Component: In-flight Guard]
        Validates the raw response text before JSON parsing.
        Returns (is_valid, error_reason).
        """
        if not text or not str(text).strip():
            return False, "Empty response body"
            
        lower_text = text.lower()
        # 1. Detect WAF/Proxy HTML
        if "<!doctype html" in lower_text:
            if "请求次数过多" in lower_text or "too many requests" in lower_text:
                return False, "Rate limit reached (HTML 429)"
            if "error 1015" in lower_text:
                return False, "Cloudflare Ray ID block (1015)"
            return False, "Proxy/WAF error page intercepted"
            
        # 2. Config-driven dynamic keyword detection (KISS & Extensible)
        if self.slot.retry_if_matches:
            for kw in self.slot.retry_if_matches:
                if kw.lower() in lower_text:
                    return False, f"Provider-specific error matched: {kw}"
            
        # 2. Detect common proxy-level JSON errors that return 200
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                error = payload.get("error")
                if error and isinstance(error, dict):
                    return False, f"API Error: {error.get('message', 'Unknown')}"
                
                # Check for empty choices/candidates
                if self.slot.provider == "google":
                    if not payload.get("candidates"):
                        return False, "Gemini returned empty candidates"
                else:
                    if not payload.get("choices"):
                        return False, "OpenAI-Proxy returned empty choices"
        except json.JSONDecodeError:
            if not ("{" in text and "}" in text):
                return False, "Non-JSON response detected"
                
        return True, ""

    def _extract_content(self, text: str) -> str:
        # Pre-validation check
        is_ok, reason = self._validate_raw_text(text)
        if not is_ok:
            raise ValueError(reason)

        try:
            payload = json.loads(text)
            if self.slot.provider == "google":
                candidates = payload.get("candidates", [])
                if candidates and isinstance(candidates[0], dict):
                    content_obj = candidates[0].get("content", {})
                    parts = content_obj.get("parts", [])
                    if parts and isinstance(parts[0], dict):
                        return str(parts[0].get("text", ""))
            else:
                choices = payload.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    return str(choices[0].get("message", {}).get("content", ""))
        except Exception:
            pass
        return str(text)

    def _retries(self) -> int:
        retries = int(
            self.slot.max_retries
            if self.slot.max_retries is not None
            else self.poll_config.max_retries
        )
        return max(1, retries)

    def _backoff(self) -> float:
        backoff = float(
            self.slot.backoff_seconds
            if self.slot.backoff_seconds is not None
            else self.poll_config.backoff_seconds
        )
        return max(0.0, backoff)

    async def judge_once(self, task: JudgeTask, q_wait: float = 0.0) -> JudgeDecision:
        """Single atomic call."""
        return await self._call_once(task, q_wait)

    async def judge_batch_once(self, tasks: list[JudgeTask], q_wait: float = 0.0) -> dict[str, JudgeDecision]:
        """Batch atomic call."""
        return await self._call_batch_once(tasks, q_wait)

    async def _call_once(self, task: JudgeTask, q_wait: float = 0.0) -> JudgeDecision:
        api_key = self._api_key()
        if not api_key:
            return JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                ok=False,
                error_message=f"Missing API key env: {self.slot.api_key_env}",
            )

        if self.slot.provider == "google":
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": SYSTEM_PROMPT + "\n\n" + _build_user_prompt(task)}]
                    }
                ],
                "generationConfig": {
                    "temperature": float(self.slot.temperature)
                }
            }
        else:
            payload = {
                "model": self.slot.model,
                "temperature": float(self.slot.temperature),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(task)},
                ],
            }

        async def _do_call(q_wait_in: float):
            nonlocal queue_wait_time, request_start, request_end
            queue_wait_time = q_wait_in
            request_start = time.perf_counter()
            
            # Real-time feedback
            body_size = len(json.dumps(payload))
            async with self._active_lock:
                AsyncJudgeClient.active_batches += 1
                curr_active = AsyncJudgeClient.active_batches

            if self.tracer:
                self.tracer.record(task.test_id, self.slot.slot_id, self.slot.model, "api_call")
            
            try:
                resp = await self.session.post(
                    self._url(),
                    json=payload,
                    headers=self._headers(api_key),
                    timeout=float(self.slot.timeout_seconds),
                )
                text = resp.text
                if self.tracer:
                    t_id_ref = task.test_id if hasattr(task, 'test_id') else tasks[0].test_id
                    self.tracer.record(t_id_ref, self.slot.slot_id, self.slot.model, "api_done")
                request_end = time.perf_counter()
                return resp.status_code, text
            except Exception as e:
                request_end = time.perf_counter()
                raise e
            finally:
                async with self._active_lock:
                    AsyncJudgeClient.active_batches -= 1

        queue_wait_time = 0.0
        request_start = 0.0
        request_end = 0.0

        try:
            status, text = await _do_call(q_wait)
            
            if status != 200:
                return JudgeDecision(
                    judge_id=self.slot.slot_id,
                    model=self.slot.model,
                    ok=False,
                    error_message=f"HTTPError {status}: {text[:500]}",
                    metadata={
                        "queue_wait_time": queue_wait_time,
                        "request_start": request_start,
                        "request_end": request_end,
                        "api_latency": request_end - request_start,
                    }
                )
        except asyncio.TimeoutError:
            return JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                ok=False,
                error_message=f"Timeout after {self.slot.timeout_seconds}s",
                metadata={"queue_wait_time": queue_wait_time}
            )
        except Exception as exc:
            return JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                ok=False,
                error_message=f"Request failed: {exc}",
                metadata={"queue_wait_time": queue_wait_time}
            )

        content = self._extract_content(text)
        metadata = {
            "queue_wait_time": queue_wait_time,
            "request_start": request_start,
            "request_end": request_end,
            "api_latency": request_end - request_start,
        }

        json_text = _extract_json_text(content)
        if json_text is None:
            return JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                raw_text=content,
                ok=False,
                error_message="LLM output contains no valid JSON object (braces not found)",
                metadata=metadata,
            )

        try:
            parsed = json.loads(json_text)
        except Exception as exc:
            return JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                raw_text=content,
                ok=False,
                error_message=f"JSON parse failed: {exc}",
                metadata=metadata,
            )

        if isinstance(parsed, dict) and "error" in parsed:
            return JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                raw_text=content,
                ok=False,
                error_message=f"API Error payload: {parsed['error']}",
                metadata=metadata,
            )
        
        # Robustness: Handle degenerate OpenAI success with empty choices
        if isinstance(parsed, dict) and "choices" in parsed and not parsed.get("choices"):
            return JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                raw_text=content,
                ok=False,
                error_message="API response contains empty choices",
                metadata=metadata,
            )

        errors_payload = parsed.get("errors", []) if isinstance(parsed, dict) else []
        errors = _parse_errors_payload(errors_payload, self.slot.slot_id)

        # Detect degenerate success (no errors key and no error key but not valid MQM either)
        # However, for now, we follow the current logic but ensure error payloads fail.
        
        return JudgeDecision(
            judge_id=self.slot.slot_id,
            model=self.slot.model,
            errors=errors,
            raw_text=content,
            ok=True,
            metadata=metadata,
        )

    async def _call_batch_once(self, tasks: list[JudgeTask], q_wait: float = 0.0) -> dict[str, JudgeDecision]:
        expected_ids = [t.test_id for t in tasks]
        api_key = self._api_key()
        if not api_key:
            return {
                test_id: JudgeDecision(
                    judge_id=self.slot.slot_id,
                    model=self.slot.model,
                    ok=False,
                    error_message=f"Missing API key env: {self.slot.api_key_env}",
                )
                for test_id in expected_ids
            }

        if self.slot.provider == "google":
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": SYSTEM_PROMPT + "\n\n" + _build_batch_user_prompt(tasks)}]
                    }
                ],
                "generationConfig": {
                    "temperature": float(self.slot.temperature)
                }
            }
        else:
            payload = {
                "model": self.slot.model,
                "temperature": float(self.slot.temperature),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_batch_user_prompt(tasks)},
                ],
            }

        async def _do_batch_call(q_wait: float):
            nonlocal queue_wait_time, request_start, request_end
            queue_wait_time = q_wait
            request_start = time.perf_counter()
            
            # Real-time feedback
            body_size = len(json.dumps(payload))
            async with self._active_lock:
                AsyncJudgeClient.active_batches += 1
                curr_active = AsyncJudgeClient.active_batches

            if self.tracer:
                for t in tasks:
                    self.tracer.record(t.test_id, self.slot.slot_id, self.slot.model, "api_call")
            
            try:
                resp = await self.session.post(
                    self._url(),
                    json=payload,
                    headers=self._headers(api_key),
                    timeout=float(self.slot.timeout_seconds),
                )
                text = resp.text
                if self.tracer:
                    for t in tasks:
                        self.tracer.record(t.test_id, self.slot.slot_id, self.slot.model, "api_done")
                request_end = time.perf_counter()
                return resp.status_code, text
            except Exception as e:
                request_end = time.perf_counter()
                raise e
            finally:
                async with self._active_lock:
                    AsyncJudgeClient.active_batches -= 1

        queue_wait_time = 0.0
        request_start = 0.0
        request_end = 0.0

        status, text = await _do_batch_call(q_wait)

        metadata = {
            "queue_wait_time": queue_wait_time,
            "request_start": request_start,
            "request_end": request_end,
            "api_latency": request_end - request_start if request_end > 0 else 0,
        }

        # CRITICAL: Detect HTML and raise error to trigger scheduler backoff
        if "<!doctype html" in text.lower() or "请求次数过多" in text:
            raise ValueError(f"Proxy/WAF error page intercepted (HTTP {status})")

        if status != 200:
            return {
                test_id: JudgeDecision(
                    judge_id=self.slot.slot_id,
                    model=self.slot.model,
                    ok=False,
                    error_message=f"HTTPError {status}: {text[:500]}",
                    metadata=metadata
                )
                for test_id in expected_ids
            }

        content = self._extract_content(text)
        json_text = _extract_json_text(content)
        if json_text is None:
            return {
                test_id: JudgeDecision(
                    judge_id=self.slot.slot_id,
                    model=self.slot.model,
                    raw_text=content,
                    ok=False,
                    error_message="Batch LLM output contains no valid JSON object",
                )
                for test_id in expected_ids
            }

        try:
            parsed = json.loads(json_text)
        except Exception as exc:
            return {
                test_id: JudgeDecision(
                    judge_id=self.slot.slot_id,
                    model=self.slot.model,
                    raw_text=content,
                    ok=False,
                    error_message=f"JSON parse failed: {exc}",
                )
                for test_id in expected_ids
            }

        if isinstance(parsed, dict) and "error" in parsed:
            return {
                test_id: JudgeDecision(
                    judge_id=self.slot.slot_id,
                    model=self.slot.model,
                    raw_text=content,
                    ok=False,
                    error_message=f"API Error payload (batch): {parsed['error']}",
                    metadata=metadata,
                )
                for test_id in expected_ids
            }

        if isinstance(parsed, dict) and "choices" in parsed and not parsed.get("choices"):
            return {
                test_id: JudgeDecision(
                    judge_id=self.slot.slot_id,
                    model=self.slot.model,
                    raw_text=content,
                    ok=False,
                    error_message="API response contains empty choices (batch)",
                    metadata=metadata,
                )
                for test_id in expected_ids
            }

        result_rows: list[dict[str, Any]] = []
        if isinstance(parsed, dict):
            rows = parsed.get("results", [])
            if isinstance(rows, list):
                result_rows = [r for r in rows if isinstance(r, dict)]

        by_id: dict[str, JudgeDecision] = {}
        for row in result_rows:
            test_id = str(row.get("test_id", "")).strip()
            if not test_id or test_id not in expected_ids:
                continue
            errors = _parse_errors_payload(row.get("errors", []), self.slot.slot_id)
            by_id[test_id] = JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                errors=errors,
                raw_text=content,
                ok=True,
            )

        for test_id in expected_ids:
            if test_id in by_id:
                by_id[test_id].metadata = metadata
                continue
            by_id[test_id] = JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                raw_text=content,
                ok=False,
                error_message=f"Missing result for test_id: {test_id}",
                metadata=metadata
            )

        return by_id

    async def judge_with_retry(self, task: JudgeTask) -> JudgeDecision:
        """Judge a single task with exponential backoff retries, respecting semaphore."""
        retries = self._retries()
        backoff = self._backoff()
        last = JudgeDecision(
            judge_id=self.slot.slot_id, model=self.slot.model, ok=False, error_message="Unknown"
        )

        for i in range(retries):
            q_start = time.perf_counter()
            if self.semaphore is not None:
                # tqdm.write(f"  [queue]: 评审员={self.slot.slot_id} 等待信号量...")
                async with self.semaphore:
                    q_wait = time.perf_counter() - q_start
                    result = await self._call_once(task, q_wait)
            else:
                q_wait = time.perf_counter() - q_start
                result = await self._call_once(task, q_wait)

            if result.ok:
                # lat = result.metadata.get("api_latency", 0) if result.metadata else 0
                # tqdm.write(f"  [done]: 评审员={self.slot.slot_id} 成功 (耗时={lat:.1f}s)")
                return result
            last = result

            if i < retries - 1 and backoff > 0:
                wait = backoff * (2**i)
                tqdm.write(
                    f"  [retry]: 评审员={self.slot.slot_id} 尝试={i+1}/{retries} "
                    f"原因='{result.error_message}' 等待={wait:.1f}s"
                )
                await asyncio.sleep(wait)

        return last

    async def judge_batch_with_retry(self, tasks: list[JudgeTask]) -> dict[str, JudgeDecision]:
        """Judge a batch of tasks with exponential backoff, only retrying failed items in each attempt."""
        if not tasks:
            return {}

        task_map = {t.test_id: t for t in tasks}
        retries = self._retries()
        backoff = self._backoff()

        # Initialize results with all failed entries
        final_results = {
            t.test_id: JudgeDecision(
                judge_id=self.slot.slot_id,
                model=self.slot.model,
                ok=False,
                error_message="Not started",
            )
            for t in tasks
        }

        for i in range(retries):
            # Identify which tasks still need to be judged
            remaining_ids = [tid for tid, res in final_results.items() if not res.ok]
            if not remaining_ids:
                break
            
            remaining_tasks = [task_map[tid] for tid in remaining_ids]

            q_start = time.perf_counter()
            if self.semaphore is not None:
                # tqdm.write(f"  [queue-batch]: 评审员={self.slot.slot_id} 批次={len(remaining_tasks)}项 等待信号量...")
                async with self.semaphore:
                    q_wait = time.perf_counter() - q_start
                    start_t = time.perf_counter()
                    batch_res = await self._call_batch_once(remaining_tasks, q_wait)
                    batch_elapsed = time.perf_counter() - start_t
            else:
                q_wait = time.perf_counter() - q_start
                start_t = time.perf_counter()
                batch_res = await self._call_batch_once(remaining_tasks, q_wait)
                batch_elapsed = time.perf_counter() - start_t

            # Update final_results with successful (and unsuccessful) attempts from this batch
            for tid, decision in batch_res.items():
                final_results[tid] = decision

            # If all are now OK, we can stop retrying
            if all(final_results[tid].ok for tid in remaining_ids):
                # tqdm.write(f"  [done-batch]: 评审员={self.slot.slot_id} 批次={len(remaining_ids)}项 成功 (耗时={batch_elapsed:.1f}s)")
                break

            if i < retries - 1 and backoff > 0:
                wait = backoff * (2**i)
                first_err = next(
                    (d.error_message for d in batch_res.values() if not d.ok), "Unknown"
                )
                tqdm.write(
                    f"  [retry-batch]: 评审员={self.slot.slot_id} 尝试={i+1}/{retries} "
                    f"剩余待评={len([tid for tid, res in final_results.items() if not res.ok])}/{len(tasks)} "
                    f"首个错误='{first_err}' 等待={wait:.1f}s"
                )
                await asyncio.sleep(wait)

        # Final Fallback: Recover remaining partial-batch misses by retrying failed items individually.
        failed_ids = [tid for tid, decision in final_results.items() if not decision.ok]
        if failed_ids:
            # We only fallback if at least some items in the block succeeded (meaning the slot isn't totally dead)
            has_some_ok = any(d.ok for d in final_results.values())
            if has_some_ok:
                recovered = 0
                for tid in failed_ids:
                    task = task_map.get(tid)
                    if task:
                        single = await self.judge_with_retry(task)
                        if single.ok:
                            recovered += 1
                        final_results[tid] = single
                if recovered:
                    tqdm.write(
                        f"  [batch-fallback]: 评审员={self.slot.slot_id} "
                        f"成功恢复={recovered}/{len(failed_ids)} (通过单项重试)"
                    )

        return final_results


# ---------------------------------------------------------------------------
#  Backward-compatible sync wrapper (kept for standalone / testing usage)
# ---------------------------------------------------------------------------
class OpenAIJudgeClient:
    """Thin sync wrapper around AsyncJudgeClient for backward compatibility."""

    def __init__(self, slot: JudgeSlotConfig, poll_config: PollConfig) -> None:
        self.slot = slot
        self.poll_config = poll_config

    def judge_with_retry(self, task: JudgeTask) -> JudgeDecision:
        async def _inner() -> JudgeDecision:
            async with aiohttp.ClientSession() as session:
                client = AsyncJudgeClient(self.slot, self.poll_config, session)
                return await client.judge_with_retry(task)
        return asyncio.run(_inner())

    def judge_batch_with_retry(self, tasks: list[JudgeTask]) -> dict[str, JudgeDecision]:
        async def _inner() -> dict[str, JudgeDecision]:
            async with aiohttp.ClientSession() as session:
                client = AsyncJudgeClient(self.slot, self.poll_config, session)
                return await client.judge_batch_with_retry(tasks)
        return asyncio.run(_inner())
