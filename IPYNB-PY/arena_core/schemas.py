# @title --- MODULE: schemas ---

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


DEFAULT_TERM_RULES = {
    "is_active": False,
    "category": "",
    "llm_instruction": "",
    "veto_validation": {
        "expected_keywords": [],
        "forbidden_keywords": [],
        "multilingual_expected": {},
        "multilingual_forbidden": {},
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_term_rules(raw_rules: Mapping[str, Any] | None) -> dict[str, Any]:
    rules = {
        "is_active": False,
        "category": "",
        "llm_instruction": "",
        "veto_validation": {
            "expected_keywords": [],
            "forbidden_keywords": [],
            "multilingual_expected": {},
            "multilingual_forbidden": {},
        },
    }
    if not isinstance(raw_rules, Mapping):
        return rules

    rules["is_active"] = bool(raw_rules.get("is_active", False))
    rules["category"] = str(raw_rules.get("category", ""))
    rules["llm_instruction"] = str(raw_rules.get("llm_instruction", ""))

    veto = raw_rules.get("veto_validation", {})
    if isinstance(veto, Mapping):
        rules["veto_validation"]["expected_keywords"] = list(veto.get("expected_keywords", []))
        rules["veto_validation"]["forbidden_keywords"] = list(veto.get("forbidden_keywords", []))
        rules["veto_validation"]["multilingual_expected"] = dict(veto.get("multilingual_expected", {}))
        rules["veto_validation"]["multilingual_forbidden"] = dict(veto.get("multilingual_forbidden", {}))

    return rules


@dataclass
class DatasetMeta:
    version: str
    test_block: str
    total_items: int
    source: str = ""


@dataclass
class DatasetItem:
    test_id: str
    source_text: str
    reference_translations: dict[str, str] = field(default_factory=dict)
    term_rules: dict[str, Any] = field(default_factory=lambda: DEFAULT_TERM_RULES.copy())
    audit_tags: list[str] = field(default_factory=list)
    source_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    dataset_meta: DatasetMeta
    items: list[DatasetItem]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunMeta:
    model_id: str
    test_block: str
    timestamp: str
    prompt_version: str
    inference_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawInferenceItem:
    test_id: str
    status: str
    hypotheses: dict[str, str]
    raw_llm_response: str
    
    # --- [New Fields for Handoff: The Reviewer's Eyes] ---
    source_text: str = ""
    reference_translations: dict[str, str] = field(default_factory=dict)
    term_context: str = ""
    audit_tags: list[str] = field(default_factory=list)
    # ----------------------------------------------------
    
    error_message: str = ""
    attempts: int = 0
    cache_hit: bool = False


@dataclass
class RawInferenceReport:
    run_meta: RunMeta
    results: list[RawInferenceItem]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditMetrics:
    chrf_score: float
    comet_score: float
    objective_penalty_pobj: float
    term_veto_passed: bool
    term_hit_rate: float
    mqm_score: float
    s_final: float


@dataclass
class AuditScoreItem:
    test_id: str
    metrics: AuditMetrics
    veto_details: dict[str, Any] = field(default_factory=dict)
    
    # --- [Context Fields for Local Handoff (Audited version)] ---
    hypotheses: dict[str, str] = field(default_factory=dict)
    source_text: str = ""
    reference_translations: dict[str, str] = field(default_factory=dict)
    term_context: str = ""
    audit_tags: list[str] = field(default_factory=list)
    # ------------------------------------------------------------


@dataclass
class AuditReport:
    audit_meta: dict[str, Any]
    scores: list[AuditScoreItem]
    block_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_dataset_bundle(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    if not isinstance(payload, Mapping):
        return ["Dataset must be a JSON object."]

    meta = payload.get("dataset_meta")
    if not isinstance(meta, Mapping):
        errors.append("Missing or invalid dataset_meta.")
    else:
        for key in ("version", "test_block", "total_items"):
            if key not in meta:
                errors.append(f"dataset_meta.{key} is required.")

    items = payload.get("items")
    if not isinstance(items, list):
        errors.append("Missing or invalid items array.")
        return errors

    required_item_fields = {"test_id", "source_text", "reference_translations", "term_rules"}
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"items[{idx}] must be an object.")
            continue
        missing = required_item_fields.difference(item.keys())
        if missing:
            errors.append(f"items[{idx}] missing fields: {sorted(missing)}")

    return errors

# %%