from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import PipelineConfig
from .pipeline import LocalPollMqmPipeline


def _load_dotenv() -> None:
    """Auto-load .env from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            print(f"[env] Loaded {env_path}")
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local PoLL + MQM evaluator (async)")
    parser.add_argument("--config", type=str, default="configs/local_poll_mqm.json", help="Path to config JSON")
    parser.add_argument("--smoke", action="store_true", help="Enable smoke mode override")
    parser.add_argument("--smoke-judges", type=int, default=None, help="Judge count in smoke mode")
    parser.add_argument("--smoke-repeat", type=int, default=None, help="Repeat count per judge in smoke mode")
    parser.add_argument("--smoke-max-items", type=int, default=None, help="Max items per block in smoke mode")
    parser.add_argument("--single-model", type=str, default=None, help="Run only one model folder/model id")
    parser.add_argument("--single-block", type=str, default=None, help="Run only one test block")
    parser.add_argument("--single-test-id", type=str, default=None, help="Run only one test_id")
    parser.add_argument("--only-judge-slot", type=str, default=None, help="Use only the specified judge slot_id")
    parser.add_argument("--request-batch-size", type=int, default=None, help="Override poll.request_batch_size")
    parser.add_argument("--judge-parallelism", type=int, default=None, help="Override poll.judge_parallelism")
    parser.add_argument("--model-parallelism", type=int, default=None, help="Run multiple model folders in parallel")
    parser.add_argument("--min-valid-judges", type=int, default=None, help="Override poll.min_valid_judges")
    parser.add_argument("--vote-threshold", type=int, default=None, help="Override poll.vote_threshold")
    parser.add_argument("--max-retries", type=int, default=None, help="Override poll.max_retries")
    parser.add_argument("--backoff", type=float, default=None, help="Override poll.backoff_seconds")
    parser.add_argument("--metrics-parallelism", type=int, default=None, help="Parallelism for GPU metrics (e.g. COMET)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing audit reports")
    parser.add_argument("--dry-run", action="store_true", help="No API call; generate empty judge outputs")
    parser.add_argument("--stages", type=str, default=None, help="Comma-separated list of stages to run (inference,audit,scoring)")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        _load_dotenv()
        args = build_parser().parse_args(argv)
        cfg = PipelineConfig.from_file(Path(args.config))

        stages = None
        if args.stages:
            stages = [s.strip().lower() for s in args.stages.split(",") if s.strip()]

        pipeline = LocalPollMqmPipeline(
            config=cfg,
            smoke_enabled=True if args.smoke else None,
            smoke_judges=args.smoke_judges,
            smoke_repeat=args.smoke_repeat,
            smoke_max_items=args.smoke_max_items,
            single_test_id=args.single_test_id,
            single_block=args.single_block,
            single_model=args.single_model,
            only_judge_slot_id=args.only_judge_slot,
            request_batch_size=args.request_batch_size,
            judge_parallelism=args.judge_parallelism,
            model_parallelism=args.model_parallelism,
            min_valid_judges=args.min_valid_judges,
            vote_threshold=args.vote_threshold,
            max_retries=args.max_retries,
            backoff_seconds=args.backoff,
            force_overwrite=bool(args.force),
            metrics_parallelism=args.metrics_parallelism,
            dry_run=bool(args.dry_run),
            stages=stages,
        )

        summary = pipeline.run()
        print("[summary] models:", summary.get("model_count", 0))
        return 0
    except Exception as e:
        import sys
        print(f"[error] {e}", file=sys.stderr)
        return 1
