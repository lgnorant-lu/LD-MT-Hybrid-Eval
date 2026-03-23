from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception as e:
        raise ValueError(f"Failed to read JSON at {path}: {e}")
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected JSON object at {path}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
            json.dump(dict(payload), fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def resolve_dataset_map(datasets_dir: Path, blocks: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for block in blocks:
        candidates = [
            datasets_dir / f"{block}_v1.json",
            datasets_dir / f"{block}.json",
        ]
        found = next((p for p in candidates if p.exists() and p.is_file()), None)
        if found is not None:
            result[block] = found
    return result


def load_dataset_index(dataset_path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(dataset_path)
    items = payload.get("items", [])
    index: dict[str, dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, Mapping):
                continue
            test_id = str(item.get("test_id", "")).strip()
            if not test_id:
                continue
            index[test_id] = dict(item)
    return index


def resolve_model_dirs(runs_dir: Path, requested_models: list[str]) -> list[Path]:
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    existing = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not requested_models:
        return sorted(existing)

    model_map = {p.name: p for p in existing}
    resolved: list[Path] = []
    for model in requested_models:
        raw = str(model).strip()
        if not raw:
            continue

        candidates = [
            raw,
            raw.replace("/", "--"),
            raw.replace("/", "_"),
            raw.replace(":", "-"),
            raw.replace(":", "-").replace("/", "--"),
        ]

        hit = None
        for c in candidates:
            if c in model_map:
                hit = model_map[c]
                break
        if hit is not None:
            resolved.append(hit)

    # Remove duplicates while preserving order
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def extract_hypothesis(row: Mapping[str, Any], preferred_language: str) -> str:
    hypotheses = row.get("hypotheses", {})
    if not isinstance(hypotheses, Mapping):
        return ""

    preferred = str(hypotheses.get(preferred_language, "")).strip()
    if preferred:
        return preferred

    fallback_en = str(hypotheses.get("en", "")).strip()
    if fallback_en:
        return fallback_en

    for value in hypotheses.values():
        text = str(value).strip()
        if text:
            return text
    return ""


def extract_reference(dataset_item: Mapping[str, Any], preferred_language: str) -> str:
    refs = dataset_item.get("reference_translations", {})
    if not isinstance(refs, Mapping):
        return ""

    preferred = str(refs.get(preferred_language, "")).strip()
    if preferred:
        return preferred

    fallback_en = str(refs.get("en", "")).strip()
    if fallback_en:
        return fallback_en

    for value in refs.values():
        text = str(value).strip()
        if text:
            return text
    return ""
