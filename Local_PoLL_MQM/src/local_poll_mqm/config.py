from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_BLOCKS = ["Baseline_Standard", "Jargon_Tech", "Slang_Ambiguous"]


@dataclass
class JudgeSlotConfig:
    slot_id: str
    provider: str
    model: str
    endpoint: str
    api_key_env: str
    timeout_seconds: int = 90
    temperature: float = 0.0
    max_retries: int | None = None
    backoff_seconds: float | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    retry_if_matches: list[str] = field(default_factory=list)


@dataclass
class PollConfig:
    vote_threshold: int = 4
    min_valid_judges: int = 4
    repeat_per_judge: int = 1
    judge_parallelism: int = 3
    request_batch_size: int = 15
    max_retries: int = 3
    backoff_seconds: float = 3.0
    span_overlap_threshold: float = 0.5
    judge_slots: list[JudgeSlotConfig] = field(default_factory=list)


@dataclass
class ScoringConfig:
    omega_1: float = 0.4
    omega_2: float = 0.6
    threshold: float = 60.0
    alpha: float = 0.2
    delta: float = 0.5
    metric_mode: str = "auto"  # auto | proxy | real
    objective_language: str = "en"
    comet_model: str = "Unbabel/wmt22-comet-da"
    comet_batch_size: int = 64
    no_reference_chrf: float = 65.0
    no_reference_comet: float = 65.0
    metrics_parallelism: int = 1


@dataclass
class RuntimeConfig:
    target_language: str = "en"
    blocks: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCKS))
    models: list[str] = field(default_factory=list)
    block_sample_limits: dict[str, int] = field(default_factory=dict)
    require_reference_blocks: list[str] = field(default_factory=list)
    test_id_allowlist: list[str] = field(default_factory=list)
    random_seed: int = 20260316


@dataclass
class PathsConfig:
    benchmarks_root: str = "../Benchmarks"
    metadata_root: str = ""
    datasets_dir_name: str = "01_Datasets"
    runs_dir_name: str = "02_Experiment_Runs"
    output_root: str = "output"


@dataclass
class SmokeConfig:
    enabled: bool = False
    judge_count: int = 3
    repeat_per_judge: int = 2
    max_items_per_block: int = 30


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    poll: PollConfig = field(default_factory=PollConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            raise ValueError(f"Config root must be JSON object: {path}")
        return payload

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path).resolve()
        data = cls._read_json(config_path)

        paths_data = dict(data.get("paths", {}))
        runtime_data = dict(data.get("runtime", {}))
        poll_data = dict(data.get("poll", {}))
        scoring_data = dict(data.get("scoring", {}))
        smoke_data = dict(data.get("smoke", {}))

        slots_raw = list(poll_data.get("judge_slots", []))
        slots: list[JudgeSlotConfig] = []
        for item in slots_raw:
            if not isinstance(item, dict):
                continue
            raw_slot_retries = item.get("max_retries")
            raw_slot_backoff = item.get("backoff_seconds")
            slots.append(
                JudgeSlotConfig(
                    slot_id=str(item.get("slot_id", "")).strip(),
                    provider=str(item.get("provider", "gemini")).strip().lower(),
                    model=str(item.get("model", "")).strip(),
                    endpoint=str(item.get("endpoint", "")).strip(),
                    api_key_env=str(item.get("api_key_env", "")).strip(),
                    timeout_seconds=int(item.get("timeout_seconds", 90)),
                    temperature=float(item.get("temperature", 0.0)),
                    max_retries=(
                        int(str(raw_slot_retries).strip())
                        if raw_slot_retries is not None
                        else None
                    ),
                    backoff_seconds=(
                        float(str(raw_slot_backoff).strip())
                        if raw_slot_backoff is not None
                        else None
                    ),
                    extra_headers=dict(item.get("headers", {})),
                    retry_if_matches=list(item.get("retry_if_matches", [])),
                )
            )

        config = cls(
            paths=PathsConfig(
                benchmarks_root=str(paths_data.get("benchmarks_root", "../Benchmarks")),
                metadata_root=str(paths_data.get("metadata_root", "")).strip(),
                datasets_dir_name=str(paths_data.get("datasets_dir_name", "01_Datasets")),
                runs_dir_name=str(paths_data.get("runs_dir_name", "02_Experiment_Runs")),
                output_root=str(paths_data.get("output_root", "output")),
            ),
            runtime=RuntimeConfig(
                target_language=str(runtime_data.get("target_language", "en")),
                blocks=list(runtime_data.get("blocks", DEFAULT_BLOCKS)),
                models=list(runtime_data.get("models", [])),
                block_sample_limits={
                    str(k): int(v)
                    for k, v in dict(runtime_data.get("block_sample_limits", {})).items()
                    if str(k).strip()
                },
                require_reference_blocks=[str(x) for x in list(runtime_data.get("require_reference_blocks", []))],
                test_id_allowlist=[str(x) for x in list(runtime_data.get("test_id_allowlist", []))],
                random_seed=int(runtime_data.get("random_seed", 20260316)),
            ),
            poll=PollConfig(
                vote_threshold=int(poll_data.get("vote_threshold", 4)),
                min_valid_judges=int(poll_data.get("min_valid_judges", 5)),
                repeat_per_judge=int(poll_data.get("repeat_per_judge", 1)),
                judge_parallelism=int(poll_data.get("judge_parallelism", 1)),
                request_batch_size=int(poll_data.get("request_batch_size", 1)),
                max_retries=int(poll_data.get("max_retries", 2)),
                backoff_seconds=float(poll_data.get("backoff_seconds", 1.5)),
                span_overlap_threshold=float(poll_data.get("span_overlap_threshold", 0.5)),
                judge_slots=slots,
            ),
            scoring=ScoringConfig(
                omega_1=float(scoring_data.get("omega_1", 0.4)),
                omega_2=float(scoring_data.get("omega_2", 0.6)),
                threshold=float(scoring_data.get("threshold", 60.0)),
                alpha=float(scoring_data.get("alpha", 0.2)),
                delta=float(scoring_data.get("delta", 0.5)),
                metric_mode=str(scoring_data.get("metric_mode", "auto")),
                objective_language=str(scoring_data.get("objective_language", "en")),
                comet_model=str(scoring_data.get("comet_model", "Unbabel/wmt22-comet-da")),
                comet_batch_size=int(scoring_data.get("comet_batch_size", 64)),
                no_reference_chrf=float(scoring_data.get("no_reference_chrf", 65.0)),
                no_reference_comet=float(scoring_data.get("no_reference_comet", 65.0)),
                metrics_parallelism=int(scoring_data.get("metrics_parallelism", 1)),
            ),
            smoke=SmokeConfig(
                enabled=bool(smoke_data.get("enabled", False)),
                judge_count=int(
                    smoke_data.get(
                        "judge_count",
                        1 if bool(smoke_data.get("use_single_judge", False)) else 3,
                    )
                ),
                repeat_per_judge=int(smoke_data.get("repeat_per_judge", 2)),
                max_items_per_block=int(smoke_data.get("max_items_per_block", smoke_data.get("sample_per_block", 30))),
            ),
        )

        config.validate()
        return config

    def validate(self) -> None:
        if not self.poll.judge_slots:
            raise ValueError("At least one judge slot is required in poll.judge_slots")

        for slot in self.poll.judge_slots:
            if not slot.model:
                raise ValueError(f"Judge slot missing model: {slot.slot_id}")
            if not slot.endpoint:
                raise ValueError(f"Judge slot missing endpoint: {slot.slot_id}")
            if not slot.api_key_env:
                raise ValueError(f"Judge slot missing api_key_env: {slot.slot_id}")

        if not self.runtime.blocks:
            raise ValueError("runtime.blocks cannot be empty")

        if self.poll.request_batch_size <= 0:
            raise ValueError("poll.request_batch_size must be >= 1")

        if self.poll.judge_parallelism <= 0:
            raise ValueError("poll.judge_parallelism must be >= 1")

        for block, limit in self.runtime.block_sample_limits.items():
            if int(limit) <= 0:
                raise ValueError(f"runtime.block_sample_limits must be positive: {block}={limit}")
