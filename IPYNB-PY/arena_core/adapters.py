# @title --- MODULE: adapters ---

import json
from pathlib import Path
from typing import Any

try:
    from .schemas import DatasetBundle, DatasetItem, DatasetMeta, normalize_term_rules
except ImportError:
    from schemas import DatasetBundle, DatasetItem, DatasetMeta, normalize_term_rules



def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _format_title_content(node: Any) -> str:
    if isinstance(node, dict):
        title = str(node.get("title", "")).strip()
        content = str(node.get("content", "")).strip()
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if content:
            parts.append(f"Content: {content}")
        return "\n".join(parts).strip()
    return str(node).strip()


def _build_glossary_lookup(glossary_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = _load_json(glossary_path)
    terms = payload.get("terms", [])

    by_id: dict[str, dict[str, Any]] = {}
    by_pattern: dict[str, dict[str, Any]] = {}
    for term in terms:
        term_id = str(term.get("term_id", "")).strip()
        pattern = str(term.get("term_pattern", "")).strip()
        normalized = {
            "is_active": True,
            "category": str(term.get("category", "")),
            "llm_instruction": str(term.get("llm_instruction", "")),
            "veto_validation": {
                "multilingual_expected": dict(term.get("veto_validation", {}).get("multilingual_expected", {})),
                "multilingual_forbidden": dict(term.get("veto_validation", {}).get("multilingual_forbidden", {})),
                "expected_keywords": list(term.get("veto_validation", {}).get("multilingual_expected", {}).get("en", [])),
                "forbidden_keywords": list(term.get("veto_validation", {}).get("multilingual_forbidden", {}).get("en", [])),
            },
        }
        if term_id:
            by_id[term_id] = normalized
        if pattern:
            by_pattern[pattern] = normalized

    return by_id, by_pattern


def build_baseline_dataset(reference_path: Path, output_path: Path, limit: int | None = None) -> Path:
    payload = _load_json(reference_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload for baseline dataset: {reference_path}")

    items: list[DatasetItem] = []

    rows = list(payload.items())
    # 修复逻辑：只有当 limit > 0 时才切片，0 或 None 表示全量
    if limit is not None and limit > 0:
        rows = rows[:limit]

    for idx, (legacy_key, data) in enumerate(rows, start=1):
        source_text = str(data.get("source", legacy_key))
        refs = dict(data.get("translations", {}))

        items.append(
            DatasetItem(
                test_id=f"baseline_{idx:04d}",
                source_text=source_text,
                reference_translations=refs,
                term_rules=normalize_term_rules(None),
                audit_tags=["baseline", "ui"],
                source_meta={"legacy_key": legacy_key},
            )
        )

    bundle = DatasetBundle(
        dataset_meta=DatasetMeta(
            version="1.0",
            test_block="Baseline_Standard",
            total_items=len(items),
            source=str(reference_path),
        ),
        items=items,
    )
    _save_json(output_path, bundle.to_dict())
    return output_path


def build_jargon_dataset(
    slang_context_path: Path,
    glossary_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> Path:
    payload = _load_json(slang_context_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload for jargon dataset: {slang_context_path}")

    by_id, by_pattern = _build_glossary_lookup(glossary_path)

    items: list[DatasetItem] = []

    source_rows = list(payload.get("items", []))
    # 修复逻辑：全量任务 (limit=0 或 None) 不切片
    is_full_run = limit is None or limit <= 0
    if not is_full_run:
        source_rows = source_rows[: limit]

    for row in source_rows:
        if not isinstance(row, dict):
            continue

        context = str(row.get("test_context", "")).strip()
        focus = str(row.get("slang_focus", "")).strip()
        if not context:
            continue

        term_rules = by_id.get(str(row.get("term_id_ref", "")).strip()) or by_pattern.get(focus) or normalize_term_rules(None)

        items.append(
            DatasetItem(
                test_id=str(row.get("test_id", f"jargon_{len(items)+1:04d}")),
                source_text=context,
                reference_translations={},
                term_rules=normalize_term_rules(term_rules),
                audit_tags=["jargon", "tech"],
                source_meta={"slang_focus": focus, "term_id_ref": row.get("term_id_ref")},
            )
        )

        if not is_full_run and limit is not None and len(items) >= limit:
            break

    bundle = DatasetBundle(
        dataset_meta=DatasetMeta(
            version="1.0",
            test_block="Jargon_Tech",
            total_items=len(items),
            source=str(slang_context_path),
        ),
        items=items,
    )
    _save_json(output_path, bundle.to_dict())
    return output_path


def build_slang_dataset(
    slang_benchmark_path: Path,
    glossary_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> Path:
    payload = _load_json(slang_benchmark_path)
    if isinstance(payload, dict):
        rows = list(payload.get("items", []))
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Expected list/dict payload for slang dataset: {slang_benchmark_path}")

    by_id, by_pattern = _build_glossary_lookup(glossary_path)

    # 修复逻辑：只有当 limit > 0 时才切片
    if limit is not None and limit > 0:
        rows = rows[:limit]

    items: list[DatasetItem] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        topic_id = row.get("topic_id", idx)
        source_text = _format_title_content(row.get("source", {}))

        refs_raw = row.get("reference_translations", {})
        references: dict[str, str] = {}
        for lang, value in refs_raw.items():
            references[str(lang)] = _format_title_content(value)

        term_id_ref = str(row.get("term_id_ref", "")).strip()
        slang_focus = ""
        annotations = row.get("slang_annotations", [])
        if annotations:
            first = str(annotations[0])
            slang_focus = first.split("(")[0].strip()

        if slang_focus and slang_focus in by_pattern:
            term_rules = by_pattern[slang_focus]
        elif term_id_ref and term_id_ref in by_id:
            term_rules = by_id[term_id_ref]
        else:
            term_rules = normalize_term_rules(None)
            if slang_focus:
                term_rules["is_active"] = True
                term_rules["category"] = "slang"
                term_rules["llm_instruction"] = f"Translate '{slang_focus}' with LinuxDo community context."
                term_rules["veto_validation"]["expected_keywords"] = [slang_focus]

        items.append(
            DatasetItem(
                test_id=f"slang_{topic_id}",
                source_text=source_text,
                reference_translations=references,
                term_rules=normalize_term_rules(term_rules),
                audit_tags=["slang", "ambiguous"],
                source_meta={"topic_id": topic_id, "slang_annotations": annotations, "term_id_ref": term_id_ref},
            )
        )

    bundle = DatasetBundle(
        dataset_meta=DatasetMeta(
            version="1.0",
            test_block="Slang_Ambiguous",
            total_items=len(items),
            source=str(slang_benchmark_path),
        ),
        items=items,
    )
    _save_json(output_path, bundle.to_dict())
    return output_path


def build_all_standard_datasets(metadata_root: Path, datasets_dir: Path, limit: int | None = None) -> dict[str, Path]:
    datasets_dir.mkdir(parents=True, exist_ok=True)

    glossary_path = metadata_root / "L_Station_Glossary.json"
    out_baseline = datasets_dir / "Baseline_Standard_v1.json"
    out_jargon = datasets_dir / "Jargon_Tech_v1.json"
    out_slang = datasets_dir / "Slang_Ambiguous_v1.json"

    build_baseline_dataset(
        reference_path=metadata_root / "Benchmark_Reference_Translations.json",
        output_path=out_baseline,
        limit=limit,
    )
    build_jargon_dataset(
        slang_context_path=metadata_root / "Benchmark_Slang_Context.json",
        glossary_path=glossary_path,
        output_path=out_jargon,
        limit=limit,
    )
    build_slang_dataset(
        slang_benchmark_path=metadata_root / "Benchmark_Slang_Golden_V1_Final.json",
        glossary_path=glossary_path,
        output_path=out_slang,
        limit=limit,
    )

    return {
        "Baseline_Standard": out_baseline,
        "Jargon_Tech": out_jargon,
        "Slang_Ambiguous": out_slang,
    }

# %%