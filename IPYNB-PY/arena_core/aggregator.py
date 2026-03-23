# @title --- MODULE: aggregator ---

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from .config import TEST_BLOCKS
    from .filesystem import ExperimentDirectoryManager
except ImportError:
    from config import TEST_BLOCKS
    from filesystem import ExperimentDirectoryManager


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


class GlobalMetricsAggregator:
    """Aggregate audited block-level reports into a global leaderboard."""

    def __init__(self, dir_manager: ExperimentDirectoryManager) -> None:
        self.dir_manager = dir_manager

    def aggregate(self, output_path: Path | None = None) -> dict[str, Any]:
        runs_dir = self.dir_manager.paths.runs_dir
        models: list[dict[str, Any]] = []

        if not runs_dir.exists():
            payload = {
                "generated_at": _utc_now_iso(),
                "model_count": 0,
                "models": [],
                "leaderboard": [],
            }
            target = output_path or self.dir_manager.global_summary_path()
            _save_json(target, payload)
            return payload

        # 遍历所有模型运行目录
        for model_dir in sorted(runs_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            audited_dir = model_dir / "audited_reports"
            if not audited_dir.exists():
                continue

            block_scores: dict[str, float] = {}
            found_model_id = None

            for block in TEST_BLOCKS:
                report_path = audited_dir / f"{block}_audit.json"
                if not report_path.exists():
                    continue
                
                try:
                    report = _load_json(report_path)
                    # 尝试从元数据提取真实 Model ID
                    if not found_model_id:
                        found_model_id = report.get("audit_meta", {}).get("model_id")
                    
                    # 获取该 Block 的平均 S_final
                    summary = report.get("block_summary", {})
                    avg_s = summary.get("avg_s_final")
                    if avg_s is not None:
                        block_scores[block] = float(avg_s)
                except Exception:
                    continue

            if not block_scores:
                continue

            overall_avg = sum(block_scores.values()) / len(block_scores)
            model_id = found_model_id or model_dir.name

            model_row = {
                "model_id": model_id,
                "model_folder": model_dir.name,
                "blocks": {k: round(v, 4) for k, v in block_scores.items()},
                "overall_avg_s_final": round(overall_avg, 4),
                "last_updated": _utc_now_iso()
            }
            models.append(model_row)

            # 为每个模型生成自己的汇总文件
            _save_json(
                model_dir / "metrics_summary.json",
                {
                    "model_id": model_id,
                    "generated_at": _utc_now_iso(),
                    "blocks": model_row["blocks"],
                    "overall_avg_s_final": model_row["overall_avg_s_final"],
                },
            )

        # 按分数降序排列
        models.sort(key=lambda x: x.get("overall_avg_s_final", 0.0), reverse=True)
        
        payload = {
            "generated_at": _utc_now_iso(),
            "model_count": len(models),
            "models": models,
            "leaderboard": models,
        }
        
        target = output_path or self.dir_manager.global_summary_path()
        _save_json(target, payload)
        return payload

# %%