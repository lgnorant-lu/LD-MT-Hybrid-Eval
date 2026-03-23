from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
import random
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx
from tqdm import tqdm

from .config import JudgeSlotConfig, PipelineConfig
from .io_utils import (
    extract_hypothesis,
    extract_reference,
    load_dataset_index,
    read_json,
    resolve_dataset_map,
    resolve_model_dirs,
    write_json,
)
from .judge_client import AsyncJudgeClient, JudgeTask
from .metrics import ObjectiveMetricsEngine
from .mqm import (
    arbitrate_errors,
    compute_objective_penalty,
    compute_s_final,
    compute_s_mqm,
    resolve_vote_threshold,
)
from .scheduler import PriorityJudgeScheduler, RequestPriority
from .term_gate import evaluate_term_gate
from .types import JudgeDecision, JudgeError
from .logging_utils import get_logger
from .tracer import DiagnosticTracer

def _reconstruct_decision(d_dict: dict) -> JudgeDecision:
    """Helper to reconstruct JudgeDecision from dictionary, ensuring nested errors are JudgeError objects."""
    if not d_dict:
        return JudgeDecision(judge_id="unknown", model="unknown", ok=False)
    
    # Deep copy/convert errors
    errors_raw = d_dict.get("errors", [])
    processed_errors = []
    for e in errors_raw:
        if isinstance(e, dict):
            # Map dict keys to JudgeError fields
            processed_errors.append(JudgeError(
                span=e.get("span", ""),
                severity=e.get("severity", "minor"),
                category=e.get("category", "other"),
                reason=e.get("reason", ""),
                judge_id=e.get("judge_id", "")
            ))
        else:
            processed_errors.append(e)
            
    # Prepare clean dict for JudgeDecision constructor
    clean_dict = dict(d_dict)
    clean_dict["errors"] = processed_errors
    return JudgeDecision(**clean_dict)

logger = get_logger("pipeline")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalPollMqmPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        smoke_enabled: bool | None = None,
        smoke_judges: int | None = None,
        smoke_repeat: int | None = None,
        smoke_max_items: int | None = None,
        single_test_id: str | None = None,
        single_block: str | None = None,
        single_model: str | None = None,
        only_judge_slot_id: str | None = None,
        request_batch_size: int | None = None,
        judge_parallelism: int | None = None,
        model_parallelism: int | None = None,
        min_valid_judges: int | None = None,
        vote_threshold: int | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        force_overwrite: bool = False,
        metrics_parallelism: int | None = None,
        dry_run: bool = False,
        stages: list[str] | None = None,
    ) -> None:
        self.config = config
        self.dry_run = bool(dry_run)
        self.force_overwrite = bool(force_overwrite)
        self.stages = stages
        self.only_judge_slot_id = str(only_judge_slot_id or "").strip()
        self.model_parallelism = max(1, int(model_parallelism)) if model_parallelism is not None else 1

        if smoke_enabled is not None:
            self.config.smoke.enabled = bool(smoke_enabled)
        if smoke_judges is not None:
            self.config.smoke.judge_count = max(1, int(smoke_judges))
        if smoke_repeat is not None:
            self.config.smoke.repeat_per_judge = max(1, int(smoke_repeat))
        if smoke_max_items is not None:
            self.config.smoke.max_items_per_block = max(1, int(smoke_max_items))
        if request_batch_size is not None:
            self.config.poll.request_batch_size = max(1, int(request_batch_size))
        if judge_parallelism is not None:
            self.config.poll.judge_parallelism = max(1, int(judge_parallelism))
        if min_valid_judges is not None:
            self.config.poll.min_valid_judges = max(1, int(min_valid_judges))
        if vote_threshold is not None:
            self.config.poll.vote_threshold = max(1, int(vote_threshold))
        if max_retries is not None:
            self.config.poll.max_retries = max(0, int(max_retries))
        if backoff_seconds is not None:
            self.config.poll.backoff_seconds = float(backoff_seconds)
        if metrics_parallelism is not None:
            self.config.scoring.metrics_parallelism = max(1, int(metrics_parallelism))

        if single_test_id:
            self.config.runtime.test_id_allowlist = [str(single_test_id).strip()]
        if single_block:
            self.config.runtime.blocks = [str(single_block).strip()]
        if single_model:
            self.config.runtime.models = [str(single_model).strip()]

        self.project_root = Path(__file__).resolve().parents[2]
        bench_root_raw = Path(self.config.paths.benchmarks_root)
        self.benchmarks_root = (
            bench_root_raw if bench_root_raw.is_absolute() else (self.project_root / bench_root_raw)
        ).resolve()

        self.metadata_root: Path | None = None
        metadata_root_raw = Path(self.config.paths.metadata_root) if self.config.paths.metadata_root else None
        if metadata_root_raw is not None:
            self.metadata_root = (
                metadata_root_raw if metadata_root_raw.is_absolute() else (self.project_root / metadata_root_raw)
            ).resolve()

        bench_datasets_dir = self.benchmarks_root / self.config.paths.datasets_dir_name
        metadata_datasets_dir = (
            self.metadata_root / self.config.paths.datasets_dir_name if self.metadata_root is not None else None
        )
        if metadata_datasets_dir is not None and metadata_datasets_dir.exists() and metadata_datasets_dir.is_dir():
            self.datasets_dir = metadata_datasets_dir
        else:
            self.datasets_dir = bench_datasets_dir

        self.runs_dir = self.benchmarks_root / self.config.paths.runs_dir_name
        output_root_raw = Path(self.config.paths.output_root)
        self.output_root = (
            output_root_raw if output_root_raw.is_absolute() else (self.project_root / output_root_raw)
        ).resolve()

        self.metrics = ObjectiveMetricsEngine(
            metric_mode=self.config.scoring.metric_mode,
            preferred_language=self.config.scoring.objective_language,
            comet_model_name=self.config.scoring.comet_model,
            no_reference_chrf=self.config.scoring.no_reference_chrf,
            no_reference_comet=self.config.scoring.no_reference_comet,
            comet_batch_size=self.config.scoring.comet_batch_size,
        )
        self.metrics_semaphore = asyncio.Semaphore(
            max(1, int(self.config.scoring.metrics_parallelism))
        )
        
        # Diagnostics
        self.tracer = DiagnosticTracer()
        self.scheduler = PriorityJudgeScheduler(
            parallelism=max(1, int(self.config.poll.judge_parallelism)),
            tracer=self.tracer
        )

    # ------------------------------------------------------------------
    #  Internal Helpers
    # ------------------------------------------------------------------
    def _select_virtual_slots(self) -> list[JudgeSlotConfig]:
        slots = list(self.config.poll.judge_slots)
        if self.only_judge_slot_id:
            slots = [s for s in slots if s.slot_id == self.only_judge_slot_id]
        if not slots:
            # Fallback to all if no match
            slots = list(self.config.poll.judge_slots)
        return slots

    def _resolve_raw_path(self, model_dir: Path, block: str) -> Path | None:
        """Resolve inference results JSON under model_dir/raw_inference/."""
        raw_dir = model_dir / "raw_inference"
        if not raw_dir.exists():
            return None
        # Support both new standard (_raw.json) and legacy (.json)
        candidate1 = raw_dir / f"{block}_raw.json"
        if candidate1.exists():
            return candidate1
        candidate2 = raw_dir / f"{block}.json"
        return candidate2 if candidate2.exists() else None

    def _get_cache_path(self, model_name: str, block: str) -> Path:
        """Get path to the fine-grained judge cache for a specific model and block."""
        cache_dir = self.output_root / "checkpoints" / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{block}_judge_cache.json"

    def _load_block_cache(self, model_name: str, block: str) -> dict[str, dict[str, dict]]:
        """Load fine-grained cache: test_id -> { slot_id: JudgeDecision_dict }."""
        path = self._get_cache_path(model_name, block)
        if path.exists():
            try:
                return read_json(path)
            except Exception:
                return {}
        return {}

    def _save_block_cache(self, model_name: str, block: str, cache_data: dict):
        """Save fine-grained cache to disk immediately."""
        path = self._get_cache_path(model_name, block)
        write_json(path, cache_data)

    async def _score_report_only(self, report: dict[str, Any], parent_pbar: tqdm | None = None) -> dict[str, Any]:
        """Pass 2 logic: Update metrics for a single report."""
        rows = report.get("results", [])
        if not rows:
            return report

        alpha = float(self.config.scoring.alpha)
        delta = float(self.config.scoring.delta)
        threshold = float(self.config.scoring.threshold)
        omega_1 = float(self.config.scoring.omega_1)
        omega_2 = float(self.config.scoring.omega_2)

        # Batch compute metrics
        async with self.metrics_semaphore:
            loop = asyncio.get_running_loop()
            batch_inputs = [
                {
                    "source_text": r.get("source", ""),
                    "hypothesis_text": r.get("hypothesis", ""),
                    "reference_text": r.get("reference", "")
                }
                for r in rows
            ]
            scores = await loop.run_in_executor(None, self.metrics.score_batch, batch_inputs)
            
            updated_rows = list(rows)
            for i, score_map in enumerate(scores):
                updated_rows[i]["chrf_score"] = score_map["chrf_score"]
                updated_rows[i]["comet_score"] = score_map["comet_score"]

        # Final MQM consolidation
        s_mqm_values = []
        chrf_values = []
        comet_values = []
        s_final_values = []

        for row in updated_rows:
            s_mqm = float(row.get("s_mqm", 0.0))
            chrf = float(row.get("chrf_score", 0.0))
            comet = float(row.get("comet_score", 0.0))
            e_term = float(row.get("term_gate", {}).get("e_term", 0.0))
            
            if s_mqm < 0:
                row["p_obj"] = 0.0
                row["s_final"] = -1.0
                continue

            p_obj = compute_objective_penalty(chrf, comet, omega_1, omega_2, threshold, alpha)
            row["p_obj"] = round(p_obj, 4)
            
            s_final = compute_s_final(s_mqm, p_obj, e_term, delta)
            row["s_final"] = round(s_final, 4)
            
            s_mqm_values.append(s_mqm)
            chrf_values.append(chrf)
            comet_values.append(comet)
            s_final_values.append(s_final)

        avg_s_final = sum(s_final_values) / len(s_final_values) if s_final_values else 0.0
        avg_s_mqm = sum(s_mqm_values) / len(s_mqm_values) if s_mqm_values else 0.0
        avg_chrf = sum(chrf_values) / len(chrf_values) if chrf_values else 0.0
        avg_comet = sum(comet_values) / len(comet_values) if comet_values else 0.0
        
        report["results"] = updated_rows
        if "block_summary" not in report:
            report["block_summary"] = {}
        
        report["block_summary"].update({
            "avg_s_final": round(avg_s_final, 4),
            "avg_s_mqm": round(avg_s_mqm, 4),
            "avg_chrf_score": round(avg_chrf, 4),
            "avg_comet_score": round(avg_comet, 4),
        })
        report["audit_meta"]["status"] = "completed"
        report["audit_meta"]["timestamp_completed"] = _utc_now_iso()
        return report

    @staticmethod
    def _batched(items: list[JudgeTask], batch_size: int) -> list[list[JudgeTask]]:
        size = max(1, int(batch_size))
        return [items[i : i + size] for i in range(0, len(items), size)]

    # ------------------------------------------------------------------
    #  ASYNC judge orchestration
    # ------------------------------------------------------------------
    async def _judge_batch_with_priority_retry(
        self,
        tasks: list[JudgeTask],
        client: AsyncJudgeClient,
        priority: int,
        max_retries: int | None = None,
        backoff: float | None = None,
        parent_pbar: tqdm | None = None,
    ) -> dict[str, JudgeDecision]:
        """Judge a batch with external priority-based retries and tracer integration."""
        if not tasks:
            return {}

        retries = max_retries if max_retries is not None else int(self.config.poll.max_retries)
        backoff = backoff if backoff is not None else float(self.config.poll.backoff_seconds)
        
        task_map = {t.test_id: t for t in tasks}
        final_results = {tid: JudgeDecision(judge_id=client.slot.slot_id, model=client.slot.model, ok=False, error_message="Pending") for tid in task_map}
        expected_ids = list(task_map.keys())

        # Main loop across retries
        for i in range(retries + 1):
            remaining_ids = [tid for tid, res in final_results.items() if not res.ok]
            if not remaining_ids:
                break
            
            remaining_tasks = [task_map[tid] for tid in remaining_ids]
            curr_priority = int(RequestPriority.SUPPLEMENT) if i > 0 else priority
            
            task_id = f"{client.slot.slot_id}:batch:{len(remaining_tasks)}:try:{i}"
            
            # Submit to scheduler and wait
            try:
                fut = await self.scheduler.submit(
                    coro_func=lambda rt=remaining_tasks: client.judge_batch_once(rt),
                    priority=curr_priority,
                    task_id=task_id,
                    test_id=remaining_ids, # Pass list for batch tracing
                    slot_id=client.slot.slot_id,
                    model=client.slot.model
                )
                batch_res = await fut
                
                for tid, decision in batch_res.items():
                    final_results[tid] = decision
                    
                if all(final_results[tid].ok for tid in remaining_ids):
                    # Batch succeeded! Update progress
                    if parent_pbar:
                        parent_pbar.update(1)
                    break
            except Exception as e:
                logger.warning(f"批次请求异常 (try {i}): {e}")
                # We don't break here, let it sleep and retry
                
            if i < retries and backoff > 0:
                wait_time = backoff * (2**i)
                logger.debug(f"等待重试退避: {wait_time}s")
                await asyncio.sleep(wait_time)

        # Fallback to single if some still failed
        # DISABLED: This can cause massive RPM spikes and IP bans when batches fail.
        # failed_ids = [tid for tid, decision in final_results.items() if not decision.ok]
        # if failed_ids and any(d.ok for d in final_results.values()):
        #     async def _retry_single(tid: str):
        #         task = task_map[tid]
        #         fut = await self.scheduler.submit(
        #             coro_func=lambda: client.judge_once(task),
        #             priority=int(RequestPriority.SUPPLEMENT),
        #             task_id=f"{client.slot.slot_id}:{tid}:fallback",
        #             test_id=tid,
        #             slot_id=client.slot.slot_id,
        #             model=client.slot.model
        #         )
        #         res = await fut
        #         if res.ok:
        #             final_results[tid] = res
        #
        #     await asyncio.gather(*[_retry_single(tid) for tid in failed_ids])
            
        return final_results

    async def _judge_slot_async(
        self,
        tasks: list[JudgeTask],
        slot: JudgeSlotConfig,
        batch_size: int,
        session: httpx.AsyncClient,
        priority: int,
        parent_pbar: tqdm | None = None,
    ) -> dict[str, JudgeDecision]:
        """Run all tasks through one judge slot."""
        client = AsyncJudgeClient(
            slot=slot,
            poll_config=self.config.poll,
            session=session,
            semaphore=None,
            tracer=self.tracer
        )
        
        if batch_size <= 1:
            async def _single_retry(task: JudgeTask) -> tuple[str, JudgeDecision]:
                retries = int(self.config.poll.max_retries)
                backoff = float(self.config.poll.backoff_seconds)
                last_res = None
                for i in range(retries + 1):
                    curr_p = int(RequestPriority.SUPPLEMENT) if i > 0 else priority
                    fut = await self.scheduler.submit(
                        coro_func=lambda: client.judge_once(task),
                        priority=curr_p,
                        task_id=f"{slot.slot_id}:{task.test_id}:try:{i}",
                        test_id=task.test_id,
                        slot_id=slot.slot_id,
                        model=slot.model
                    )
                    last_res = await fut
                    if last_res.ok:
                        return task.test_id, last_res
                    if i < retries:
                        await asyncio.sleep(backoff * (2**i))
                return task.test_id, last_res

            results = await asyncio.gather(*[_single_retry(t) for t in tasks], return_exceptions=True)
            slot_results = {}
            for item in results:
                if isinstance(item, tuple):
                    tid, dec = item
                    slot_results[tid] = dec
            return slot_results

        chunks = self._batched(tasks, batch_size)
        all_results = await asyncio.gather(*[
            self._judge_batch_with_priority_retry(c, client, priority, parent_pbar=parent_pbar)
            for c in chunks
        ], return_exceptions=True)
        
        slot_results = {}
        for res_map in all_results:
            if isinstance(res_map, dict):
                slot_results.update(res_map)
        return slot_results

    async def _judge_tasks_async(
        self,
        tasks: list[JudgeTask],
        virtual_slots: list[JudgeSlotConfig],
        session: httpx.AsyncClient,
        priority: int,
        model_name: str = "",
        block: str = "",
        parent_pbar: tqdm | None = None,
    ) -> dict[str, list[JudgeDecision]]:
        """Orchestrate judging across all slots with fine-grained caching."""
        if not tasks:
            return {t.test_id: [] for t in tasks}

        # 1. Handle cache based on force_overwrite
        if self.force_overwrite:
            logger.info(f"[FORCE] 正在忽略并清理缓存: {model_name} / {block}")
            cache = {}
            # Robust Cache Update: Clear the file on disk if it exists
            cache_path = self._get_cache_path(model_name, block)
            if cache_path.exists():
                cache_path.unlink()
        else:
            cache = self._load_block_cache(model_name, block)
        
        # 2. Reconstruct decisions from cache and identify gaps
        final_decisions: dict[str, list[JudgeDecision]] = {t.test_id: [] for t in tasks}
        tasks_by_slot: dict[str, list[JudgeTask]] = {s.slot_id: [] for s in virtual_slots}
        
        hit_count = 0
        miss_count = 0

        for task in tasks:
            test_id = task.test_id
            cached_item = cache.get(test_id, {})
            
            for slot in virtual_slots:
                slot_id = slot.slot_id
                # Requirement 1: Force Overwrite Penetration
                if not self.force_overwrite and slot_id in cached_item:
                    d_dict = cached_item[slot_id]
                    if d_dict.get("ok"):
                        final_decisions[test_id].append(_reconstruct_decision(d_dict))
                        hit_count += 1
                        continue
                
                # If not cached or not OK (or force_overwrite is True), add to pending list for this slot
                tasks_by_slot[slot_id].append(task)
                miss_count += 1

        # Requirement 4: Detailed Logging
        if hit_count > 0:
            logger.info(f"缓存命中: {model_name}/{block} 命中 {hit_count} 个槽位决策")
        if miss_count > 0:
            logger.info(f"缓存未命中: {model_name}/{block} 需要进行 {miss_count} 个槽位 API 调用")

        # 3. Define runner that updates cache per slot
        batch_size = max(1, int(self.config.poll.request_batch_size))
        
        async def _run_slot_and_update_cache(slot: JudgeSlotConfig, pending_tasks: list[JudgeTask]):
            if not pending_tasks:
                return {}
            
            logger.debug(f"正在对槽位 {slot.slot_id} 发起 {len(pending_tasks)} 个任务的 API 请求")
            slot_results = await self._judge_slot_async(
                pending_tasks, slot, batch_size, session, priority, parent_pbar=parent_pbar
            )
            
            # Persist results immediately to disk
            if slot_results:
                current_cache = self._load_block_cache(model_name, block)
                for tid, dec in slot_results.items():
                    if tid not in current_cache:
                        current_cache[tid] = {}
                    current_cache[tid][slot.slot_id] = asdict(dec)
                self._save_block_cache(model_name, block, current_cache)
            
            return slot_results

        # 4. Fire only the missing pieces
        coros = [
            _run_slot_and_update_cache(slot, tasks_by_slot[slot.slot_id])
            for slot in virtual_slots
        ]
        
        new_results_list = await asyncio.gather(*coros, return_exceptions=True)

        # 5. Merge new results into final_decisions
        for slot, slot_result in zip(virtual_slots, new_results_list):
            if isinstance(slot_result, Exception):
                logger.warning(f"槽位任务失败: {slot.slot_id} {slot_result}")
                continue
            
            for tid, dec in slot_result.items():
                if tid in final_decisions:
                    final_decisions[tid].append(dec)
                    
        return final_decisions

    # ------------------------------------------------------------------
    #  Block evaluation
    # ------------------------------------------------------------------
    async def _evaluate_block_inference_only(
        self,
        model_dir: Path,
        block: str,
        dataset_index: dict[str, dict[str, Any]],
        raw_path: Path,
        virtual_slots: list[JudgeSlotConfig],
        session: httpx.AsyncClient,
        priority: int,
        parent_pbar: tqdm | None = None,
    ) -> tuple[dict[str, list[JudgeDecision]], list[dict[str, Any]], dict[str, Any]]:
        """Pass 1a: LLM Inference Only."""
        raw_report = read_json(raw_path)
        rows = raw_report.get("results", [])
        
        target_language = self.config.runtime.target_language
        objective_language = self.config.scoring.objective_language
        require_ref = block in self.config.runtime.require_reference_blocks

        prepared_rows = []
        for row in rows:
            test_id = str(row.get("test_id", ""))
            dataset_item = dataset_index.get(test_id, {})
            if not dataset_item: continue
            
            prepared_rows.append({
                "test_id": test_id,
                "row_status": row.get("status", "SUCCESS"),
                "dataset_item": dataset_item,
                "source_text": dataset_item.get("source_text", ""),
                "hypothesis_text": extract_hypothesis(row, target_language),
                "reference_text": extract_reference(dataset_item, objective_language),
            })

        # Apply limits
        block_limit = int(self.config.runtime.block_sample_limits.get(block, 0))
        if block_limit > 0 and len(prepared_rows) > block_limit:
            rng = random.Random(int(self.config.runtime.random_seed) + sum(ord(c) for c in block))
            prepared_rows = rng.sample(prepared_rows, block_limit)

        tasks = [
            JudgeTask(
                test_id=str(item["test_id"]),
                source_text=str(item["source_text"]),
                hypothesis_text=str(item["hypothesis_text"]),
                reference_text=str(item["reference_text"]),
                target_language=target_language,
            )
            for item in prepared_rows
        ]

        if parent_pbar:
            parent_pbar.set_postfix({"block": block, "items": len(tasks), "phase": "LLM"})
        
        decisions_map = await self._judge_tasks_async(
            tasks, virtual_slots, session, priority, 
            model_name=model_dir.name, block=block, parent_pbar=parent_pbar
        )
        
        stats = {"input_rows": len(rows), "evaluated": len(tasks)}
        return decisions_map, prepared_rows, stats

    def _audit_block_from_results(
        self,
        model_dir: Path,
        block: str,
        decisions_map: dict[str, list[JudgeDecision]],
        prepared_rows: list[dict[str, Any]],
        raw_report_meta: dict[str, Any],
        inference_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Pass 1b: Local Arbitration."""
        row_outputs = []
        valid_judge_counts = []
        
        vote_thresh_config = self.config.poll.vote_threshold
        min_v = self.config.poll.min_valid_judges
        
        for item in prepared_rows:
            test_id = item["test_id"]
            decisions = decisions_map.get(test_id, [])
            valid_decisions = [d for d in decisions if d.ok]
            valid_count = len(valid_decisions)
            valid_judge_counts.append(valid_count)
            
            audit_status = "success"
            audit_message = ""
            
            if valid_count < min_v:
                audit_status = "insufficient_judges"
                audit_message = f"Found {valid_count} valid judges, but need at least {min_v}."
                s_mqm = -1.0
                accepted, rejected, severities = [], [], {}
                vote_threshold = min_v
            else:
                vote_threshold = resolve_vote_threshold(vote_thresh_config, valid_count)
                accepted, rejected, severities = arbitrate_errors(
                    valid_decisions, vote_threshold, self.config.poll.span_overlap_threshold
                )
                s_mqm = compute_s_mqm(severities, item["source_text"], item["hypothesis_text"])
            
            term_rules = item["dataset_item"].get("term_rules", {})
            term_res = evaluate_term_gate(term_rules, item["hypothesis_text"], self.config.runtime.target_language)
            
            row_outputs.append({
                "test_id": test_id,
                "source": item["source_text"],
                "hypothesis": item["hypothesis_text"],
                "reference": item["reference_text"],
                "valid_judges": valid_count,
                "vote_threshold": vote_threshold,
                "accepted_errors": accepted,
                "rejected_errors": rejected,
                "s_mqm": s_mqm,
                "term_gate": term_res,
                "audit_status": audit_status,
                "audit_message": audit_message,
            })

        # Calculate average MQM, excluding failed rows (-1.0)
        valid_mqm_rows = [r["s_mqm"] for r in row_outputs if r["s_mqm"] >= 0]
        avg_s_mqm = sum(valid_mqm_rows) / len(valid_mqm_rows) if valid_mqm_rows else 0
        return {
            "audit_meta": {
                "model_id": raw_report_meta.get("model_id", model_dir.name),
                "block": block,
                "status": "llm_evaluated",
                "timestamp": _utc_now_iso(),
                "inference_stats": inference_stats
            },
            "block_summary": {"avg_s_mqm": round(avg_s_mqm, 4)},
            "results": row_outputs
        }

    # ------------------------------------------------------------------
    #  Model processing
    # ------------------------------------------------------------------
    async def _process_model_inference(
        self,
        model_dir: Path,
        dataset_indexes: dict[str, dict[str, Any]],
        virtual_slots: list[JudgeSlotConfig],
        session: httpx.AsyncClient,
        model_priority_offset: int = 0,
        parent_pbar: tqdm | None = None,
    ) -> dict[str, Any]:
        """Pass 1a: Parallel blocks for one model."""
        blocks = self.config.runtime.blocks
        inference_data = {}
        
        audits_root = self.output_root / "audited_reports"
        model_report_dir = audits_root / model_dir.name

        for i, block in enumerate(blocks):
            raw_path = self._resolve_raw_path(model_dir, block)
            if not raw_path: continue
            
            # True Decoupling: Skip LLM API calls if the audit report already exists
            report_path = model_report_dir / f"{block}_poll_mqm_audit.json"
            if not self.force_overwrite and report_path.exists():
                logger.info(f"跳过已评审的块: {model_dir.name} / {block}")
                inference_data[block] = (None, None, None, raw_path)
                # Advance the progress bar for the skipped batches
                if parent_pbar:
                    raw = read_json(raw_path)
                    count = len(raw.get("results", []))
                    limit = int(self.config.runtime.block_sample_limits.get(block, 0))
                    if limit > 0: count = min(count, limit)
                    batch_size = max(1, int(self.config.poll.request_batch_size))
                    skipped_batches = math.ceil(count / batch_size) * len(virtual_slots)
                    parent_pbar.update(skipped_batches)
                continue
            
            priority = int(RequestPriority.SAME_MODEL_SAME_BLOCK) + model_priority_offset + i
            try:
                res = await self._evaluate_block_inference_only(
                    model_dir, block, dataset_indexes[block], raw_path, virtual_slots, session, priority, parent_pbar
                )
                inference_data[block] = res + (raw_path,)
            except Exception as e:
                logger.error(f"推理失败: {model_dir.name} {block} - {e}")
                inference_data[block] = (None, None, None, raw_path)
        return inference_data

    async def _process_model_audit(
        self,
        model_dir: Path,
        dataset_indexes: dict[str, dict[str, Any]],
        parent_pbar: tqdm | None = None,
    ) -> list[Path]:
        """Pass 1b: Local processing. Truly decoupled from Phase 1a by reading disk cache."""
        audits_root = self.output_root / "audited_reports"
        model_report_dir = audits_root / model_dir.name
        model_report_dir.mkdir(parents=True, exist_ok=True)
        
        target_language = self.config.runtime.target_language
        objective_language = self.config.scoring.objective_language
        blocks = self.config.runtime.blocks
        
        paths = []
        for block in blocks:
            raw_path = self._resolve_raw_path(model_dir, block)
            if not raw_path:
                continue
            
            # 1. Load data from disk
            cache = self._load_block_cache(model_dir.name, block)
            raw_report = read_json(raw_path)
            rows = raw_report.get("results", [])
            dataset_index = dataset_indexes.get(block, {})
            
            # 2. Reconstruct prepared_rows (logic must match Phase 1a)
            prepared_rows = []
            for row in rows:
                test_id = str(row.get("test_id", ""))
                dataset_item = dataset_index.get(test_id, {})
                if not dataset_item: continue
                prepared_rows.append({
                    "test_id": test_id,
                    "row_status": row.get("status", "SUCCESS"),
                    "dataset_item": dataset_item,
                    "source_text": dataset_item.get("source_text", ""),
                    "hypothesis_text": extract_hypothesis(row, target_language),
                    "reference_text": extract_reference(dataset_item, objective_language),
                })
            
            # Apply same limits as Phase 1a
            block_limit = int(self.config.runtime.block_sample_limits.get(block, 0))
            if block_limit > 0 and len(prepared_rows) > block_limit:
                rng = random.Random(int(self.config.runtime.random_seed) + sum(ord(c) for c in block))
                prepared_rows = rng.sample(prepared_rows, block_limit)

            # 3. Reconstruct decisions_map from cache
            decisions_map: dict[str, list[JudgeDecision]] = {}
            for item in prepared_rows:
                tid = item["test_id"]
                cached_slots = cache.get(tid, {})
                decisions_map[tid] = [
                    _reconstruct_decision(d_dict) 
                    for d_dict in cached_slots.values() 
                    if d_dict.get("ok")
                ]

            # 4. Audit
            report = self._audit_block_from_results(
                model_dir, block, decisions_map, prepared_rows, raw_report.get("run_meta", {}),
                inference_stats={"reconstructed": True}
            )
            report_path = model_report_dir / f"{block}_poll_mqm_audit.json"
            write_json(report_path, report)
            paths.append(report_path)
            if parent_pbar:
                parent_pbar.update(1)
        return paths

    async def _process_model_scoring(self, model_dir: Path, parent_pbar: tqdm | None = None) -> dict[str, Any]:
        """Pass 2: Final scoring."""
        audits_root = self.output_root / "audited_reports"
        model_report_dir = audits_root / model_dir.name
        blocks = self.config.runtime.blocks
        scores = {}
        model_id = model_dir.name
        
        for block in blocks:
            path = model_report_dir / f"{block}_poll_mqm_audit.json"
            if not path.exists(): continue
            
            report = read_json(path)
            if not self.force_overwrite and report.get("audit_meta", {}).get("status") == "completed":
                scores[block] = report["block_summary"]["avg_s_final"]
                continue
                
            updated = await self._score_report_only(report, parent_pbar)
            write_json(path, updated)
            scores[block] = updated["block_summary"]["avg_s_final"]
            model_id = updated["audit_meta"]["model_id"]
            if parent_pbar: parent_pbar.update(1)

        return {
            "model_row": {
                "model_id": model_id,
                "model_folder": model_dir.name,
                "blocks": scores,
                "overall_avg_s_final": round(sum(scores.values())/len(scores), 4) if scores else 0,
                "last_updated": _utc_now_iso(),
            }
        }

    # ------------------------------------------------------------------
    #  Main Entry
    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        leaderboard_root = self.output_root / "leaderboard"
        leaderboard_root.mkdir(parents=True, exist_ok=True)

        # Diagnostics Logging
        log_path = self.output_root / "scheduler_diagnostic.log"
        diag_logger = logging.getLogger("scheduler")
        diag_logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        diag_logger.handlers = [fh]
        self.scheduler.logger = diag_logger

        dataset_map = resolve_dataset_map(self.datasets_dir, self.config.runtime.blocks)
        model_dirs = resolve_model_dirs(self.runs_dir, self.config.runtime.models)
        
        num_slots = len(self._select_virtual_slots())
        batch_size = max(1, int(self.config.poll.request_batch_size))
        
        # Estimate work for tqdm
        total_batches = 0
        total_blocks_to_process = 0
        for m_dir in model_dirs:
            for block in self.config.runtime.blocks:
                raw_path = self._resolve_raw_path(m_dir, block)
                if raw_path:
                    raw = read_json(raw_path)
                    count = len(raw.get("results", []))
                    limit = int(self.config.runtime.block_sample_limits.get(block, 0))
                    if limit > 0: count = min(count, limit)
                    total_batches += math.ceil(count / batch_size)
                    total_blocks_to_process += 1
        
        phase1a_steps = total_batches * num_slots
        phase1b_steps = total_blocks_to_process
        phase2_steps = total_blocks_to_process

        active_stages = self.stages if self.stages else ["inference", "audit", "scoring"]
        total_steps = 0
        if "inference" in active_stages: total_steps += phase1a_steps
        if "audit" in active_stages: total_steps += phase1b_steps
        if "scoring" in active_stages: total_steps += phase2_steps

        pbar = tqdm(total=total_steps, desc="[Overall Progress]", unit="step")

        await self.scheduler.start()
        
        dataset_indexes = {b: load_dataset_index(dataset_map[b]) for b in dataset_map}

        # Pass 1a: Global Inference (fills judge_cache.json)
        if "inference" in active_stages:
            pbar.set_description("[Pass 1a/3]: Global Inference")
            async with httpx.AsyncClient(http2=False, timeout=300.0) as session:
                tasks = [
                    self._process_model_inference(m, dataset_indexes, 
                                                self._select_virtual_slots(), session, i*100, pbar)
                    for i, m in enumerate(model_dirs)
                ]
                await asyncio.gather(*tasks)
        else:
            logger.info("跳过阶段: inference")

        # Pass 1b: Audit & Arbitration (reads from judge_cache.json)
        if "audit" in active_stages:
            pbar.set_description("[Pass 1b/3]: Audit & Arbitration")
            audit_tasks = [self._process_model_audit(m, dataset_indexes, pbar) for m in model_dirs]
            await asyncio.gather(*audit_tasks)
        else:
            logger.info("跳过阶段: audit")

        # Pass 2: Scoring
        model_rows = []
        if "scoring" in active_stages:
            pbar.set_description("[Pass 2/3]: Scoring")
            for m in model_dirs:
                res = await self._process_model_scoring(m, pbar)
                if res["model_row"]: model_rows.append(res["model_row"])
        else:
            logger.info("跳过阶段: scoring")

        pbar.close()
        await self.scheduler.stop()

        # Profiling Report
        self.tracer.save_report(self.output_root / "profiling_report.json")
        
        # Save Leaderboard
        model_rows.sort(key=lambda x: x["overall_avg_s_final"], reverse=True)
        lb = {"generated_at": _utc_now_iso(), "models": model_rows}
        write_json(leaderboard_root / "Global_PoLL_MQM_Summary.json", lb)
        
        logger.info(f"审计完成! 诊断报告 -> {self.output_root / 'profiling_report.json'}")
        return lb
