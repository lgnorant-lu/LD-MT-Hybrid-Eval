# @title --- MODULE: filesystem ---

from pathlib import Path

try:
    from .config import ArenaPaths, TEST_BLOCKS
except ImportError:
    from config import ArenaPaths, TEST_BLOCKS



class ExperimentDirectoryManager:
    """Create and resolve benchmark file paths."""

    def __init__(self, paths: ArenaPaths) -> None:
        self.paths = paths

    def ensure_base_tree(self) -> None:
        self.paths.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.leaderboard_dir.mkdir(parents=True, exist_ok=True)

    def ensure_model_tree(self, model_id: str) -> Path:
        model_dir = self.model_dir(model_id)
        self.paths.raw_inference_dir(model_id).mkdir(parents=True, exist_ok=True)
        self.paths.audited_reports_dir(model_id).mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints_dir(model_id).mkdir(parents=True, exist_ok=True)
        return model_dir

    def model_dir(self, model_id: str) -> Path:
        return self.paths.model_run_dir(model_id)

    def dataset_path(self, test_block: str, version: str = "v1") -> Path:
        return self.paths.datasets_dir / f"{test_block}_{version}.json"

    def raw_output_path(self, model_id: str, test_block: str) -> Path:
        return self.paths.raw_inference_dir(model_id) / f"{test_block}_raw.json"

    def audit_output_path(self, model_id: str, test_block: str) -> Path:
        return self.paths.audited_reports_dir(model_id) / f"{test_block}_audit.json"

    def checkpoint_path(self, model_id: str, test_block: str) -> Path:
        return self.paths.checkpoints_dir(model_id) / f"{test_block}_checkpoint.json"

    def model_summary_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "metrics_summary.json"

    def global_summary_path(self) -> Path:
        return self.paths.leaderboard_dir / "Global_Metrics_Summary.json"

    def run_manifest_path(self) -> Path:
        return self.paths.root / "run_manifest.json"

    def model_error_report_path(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "run_errors.json"

    def list_expected_dataset_paths(self, version: str = "v1") -> list[Path]:
        return [self.dataset_path(test_block, version=version) for test_block in TEST_BLOCKS]

# %%