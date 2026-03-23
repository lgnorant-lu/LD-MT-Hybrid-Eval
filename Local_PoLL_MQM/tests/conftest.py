import pytest
import asyncio
import aiohttp
from pathlib import Path
import json
from local_poll_mqm.config import PipelineConfig, PathsConfig, RuntimeConfig, PollConfig, ScoringConfig, SmokeConfig, JudgeSlotConfig

# event_loop fixture removed to use pytest-asyncio default behavior

@pytest.fixture
def mock_config():
    """Common mock configuration for testing."""
    return {
        "poll": {
            "vote_threshold": 4,
            "min_valid_judges": 5,
            "span_overlap_threshold": 0.5,
            "request_batch_size": 20
        },
        "scoring": {
            "omega_1": 0.4,
            "omega_2": 0.4,
            "threshold": 0.5,
            "alpha": 0.1,
            "delta": 0.1
        }
    }

@pytest.fixture
def sample_report_data():
    """Minimal report structure for testing pipeline passes."""
    return {
        "audit_meta": {
            "model_id": "test_model",
            "status": "llm_evaluated",
            "timestamp": "2026-03-18T00:00:00Z"
        },
        "block_summary": {
            "avg_s_mqm": 90.0
        },
        "row_outputs": [
            {
                "test_id": "item_1",
                "source_text": "Hello",
                "hypothesis": "Halo",
                "reference": "Hello world",
                "metrics": {
                    "s_mqm": 90.0,
                    "chrf_score": 0.0,
                    "comet_score": 0.0,
                    "s_final": 0.0
                }
            }
        ]
    }

@pytest.fixture
def mock_pipeline_config(tmp_path):
    benchmarks = tmp_path / "benchmarks"
    metadata = tmp_path / "metadata"
    output = tmp_path / "output"
    
    # Create required directory structure
    datasets_dir = benchmarks / "01_Datasets"
    datasets_dir.mkdir(parents=True)
    (benchmarks / "02_Experiment_Runs").mkdir(parents=True)
    metadata.mkdir()
    output.mkdir()
    
    # Create a dummy dataset file
    (datasets_dir / "test_block.json").write_text('{"items": []}', encoding="utf-8")
    
    return PipelineConfig(
        paths=PathsConfig(
            benchmarks_root=str(benchmarks),
            metadata_root=str(metadata),
            datasets_dir_name="01_Datasets",
            runs_dir_name="02_Experiment_Runs",
            output_root=str(output)
        ),
        runtime=RuntimeConfig(
            target_language="en",
            blocks=["test_block"],
            models=["test_model"],
            block_sample_limits={"test_block": 10},
            require_reference_blocks=["test_block"]
        ),
        poll=PollConfig(
            judge_slots=[
                JudgeSlotConfig(
                    slot_id="s1",
                    provider="openai",
                    model="m1",
                    endpoint="http://api.test",
                    api_key_env="K1",
                    timeout_seconds=90,
                    temperature=0.0
                )
            ],
            min_valid_judges=1,
            vote_threshold=1
        ),
        scoring=ScoringConfig(),
        smoke=SmokeConfig()
    )
