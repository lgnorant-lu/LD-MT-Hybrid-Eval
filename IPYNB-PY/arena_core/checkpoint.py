# @title --- MODULE: checkpoint ---

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CheckpointState:
    version: str = "1.1" # Bumped version for extra metadata
    updated_at: str = field(default_factory=utc_now_iso)
    done_ids: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    translator_type: str = "unknown" # Added for Mock vs Real detection

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "done_ids": sorted(set(self.done_ids)),
            "failed": self.failed,
            "translator_type": self.translator_type,
        }


class CheckpointStore:
    """Persist resumable progress for long-running inference tasks.
    
    Ensures safe concurrent-ish access and atomic writes to prevent 
    corruption on Colab/Drive environments.
    """

    def __init__(self, path: Path, expected_translator: str | None = None) -> None:
        self.path = Path(path)
        self.state = CheckpointState()
        self._done_set: set[str] = set()
        self._load()
        
        # 核心保护逻辑：如果检测到 Translator 类型从 Mock 切换到 vLLM，强制清除进度
        if expected_translator and self.state.translator_type != "unknown":
            if self.state.translator_type != expected_translator:
                msg = f"检测到运行模式变更 ({self.state.translator_type} -> {expected_translator})，正在重置进度以确保数据真实性..."
                print(f"\n[CHECKPOINT-GUARD] {msg}")
                self.clear()
        
        # 更新当前的类型
        if expected_translator:
            self.state.translator_type = expected_translator

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            with self.path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            
            # 兼容旧版本或空文件
            if not isinstance(payload, dict):
                payload = {}
        except (json.JSONDecodeError, IOError):
            # 如果文件损坏，尝试寻找备份或直接重置
            payload = {}

        self.state = CheckpointState(
            version=str(payload.get("version", "1.1")),
            updated_at=str(payload.get("updated_at", utc_now_iso())),
            done_ids=list(payload.get("done_ids", [])),
            failed=dict(payload.get("failed", {})),
            translator_type=str(payload.get("translator_type", "unknown")),
        )
        self._done_set = set(self.state.done_ids)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state.updated_at = utc_now_iso()
        self.state.done_ids = list(self._done_set) # 确保同步

        # 原子性写入：先写临时文件再重命名，防止进程中断导致 JSON 损坏
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fp:
                json.dump(self.state.to_dict(), fp, ensure_ascii=False, indent=2)
            # 在 Windows 上 replace 可能会失败如果目标文件被占用，但在 Linux/Colab 上是原子的
            tmp_path.replace(self.path)
        except Exception:
            # 如果原子替换失败，尝试直接写（兜底）
            with self.path.open("w", encoding="utf-8") as fp:
                json.dump(self.state.to_dict(), fp, ensure_ascii=False, indent=2)

    def is_done(self, test_id: str) -> bool:
        """Check if a test_id is already completed to skip it."""
        return str(test_id) in self._done_set

    def clear(self) -> None:
        """Wipe all progress for this store (force reset)."""
        self._done_set.clear()
        self.state.done_ids.clear()
        self.state.failed.clear()
        if self.path.exists():
            try:
                self.path.unlink()
            except Exception:
                pass

    def mark_done(self, test_id: str) -> None:
        """Mark a single ID as done and persist immediately."""
        tid = str(test_id)
        if tid not in self._done_set:
            self._done_set.add(tid)
            if tid in self.state.failed:
                self.state.failed.pop(tid)
            self._save()

    def mark_failed(self, test_id: str, reason: str) -> None:
        self.state.failed[test_id] = reason
        self._save()

    def done_count(self) -> int:
        return len(self._done_set)

    def failed_count(self) -> int:
        return len(self.state.failed)

# %%