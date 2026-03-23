# @title --- MODULE: config ---
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Sequence

TEST_BLOCKS: Sequence[str] = (
    "Baseline_Standard",
    "Jargon_Tech",
    "Slang_Ambiguous",
)

ACTIVE_CANDIDATE_MODELS: list[str] = [
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
]

ALL_CANDIDATE_MODELS: list[str] = [
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
]

TARGET_LANGUAGES: Sequence[str] = (
    "en",
    "zh-TW",
    "ja",
    "ko",
    "de",
    "fr",
    "it",
)

COLAB_WORKSPACE_HINTS: Sequence[Path] = (
    Path("/content/drive/MyDrive/LinuxDo"),
    Path("/content/drive/MyDrive"),
)

B2_SCHEMA_VERSION = "1.1"
SCORE_SPEC_VERSION = "2026-03-13"
DEFAULT_PROMPT_VERSION = "bundled-v2"
DEFAULT_TARGET_LANGUAGE = "en"


def sanitize_model_id(model_id: str) -> str:
    """Convert model IDs to safe directory names.
    
    Supports:
    1. standard: [org]/[model] -> [org]--[model]
    2. legacy export: [org]--[model] -> [org]--[model]
    3. underscores: [org]_[model] -> [org]--[model]
    """
    # 统一将 / 或 _ 转换为 --，以固定匹配导出包中的文件夹
    raw = model_id.strip()
    # 替换规则：
    # / -> -- (huggingface style)
    # : -> -  (ollama style)
    # _ -> -- (some export tools use underscore)
    sanitized = raw.replace("/", "--").replace(":", "-")
    # 如果原本就是 -- 这种形式，replace "/" 并不冲突。
    # 额外补充：如果用户解压出来的是 [org]_[model]，我们也兼容。
    return sanitized


@dataclass(frozen=True)
class ArenaPaths:
    """Canonical benchmark directory structure."""

    root: Path
    is_mock: bool = False
    datasets_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    leaderboard_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        # 如果是 mock 模式，将根目录偏移到 .mock_results
        base_root = Path(self.root)
        if self.is_mock:
            base_root = base_root / ".mock_results"
            
        object.__setattr__(self, "root", base_root)
        object.__setattr__(self, "datasets_dir", base_root / "01_Datasets")
        object.__setattr__(self, "runs_dir", base_root / "02_Experiment_Runs")
        object.__setattr__(self, "leaderboard_dir", base_root / "03_Leaderboard")

    def model_run_dir(self, model_id: str) -> Path:
        return self.runs_dir / sanitize_model_id(model_id)

    def raw_inference_dir(self, model_id: str) -> Path:
        return self.model_run_dir(model_id) / "raw_inference"

    def audited_reports_dir(self, model_id: str) -> Path:
        return self.model_run_dir(model_id) / "audited_reports"

    def checkpoints_dir(self, model_id: str) -> Path:
        return self.model_run_dir(model_id) / "checkpoints"


def discover_workspace_root(start: Path | None = None) -> Path:
    """Find workspace root by checking Colab mounts first, then local parents."""
    for candidate in COLAB_WORKSPACE_HINTS:
        if (candidate / "IPYNB-PY").exists() and (candidate / "Metadatas").exists():
            return candidate
        nested = candidate / "LinuxDo"
        if (nested / "IPYNB-PY").exists() and (nested / "Metadatas").exists():
            return nested

    cursor = (start or Path.cwd()).resolve()
    candidates = [cursor, *cursor.parents]
    for candidate in candidates:
        if (candidate / "IPYNB-PY").exists() and (candidate / "Metadatas").exists():
            return candidate
    return cursor


def build_default_paths(start: Path | None = None) -> ArenaPaths:
    workspace_root = discover_workspace_root(start=start)
    return ArenaPaths(workspace_root / "Benchmarks")

# %%