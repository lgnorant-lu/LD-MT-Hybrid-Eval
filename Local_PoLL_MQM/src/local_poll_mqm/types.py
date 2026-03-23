from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEVERITY_WEIGHTS: dict[str, int] = {
    "minor": 1,
    "major": 5,
    "critical": 25,
}

WEIGHT_TO_SEVERITY: dict[int, str] = {
    1: "minor",
    5: "major",
    25: "critical",
}


@dataclass
class JudgeError:
    span: str
    severity: str
    category: str = "other"
    reason: str = ""
    judge_id: str = ""


@dataclass
class JudgeDecision:
    judge_id: str
    model: str
    errors: list[JudgeError] = field(default_factory=list)
    raw_text: str = ""
    ok: bool = True
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)  # Performance metrics


@dataclass
class ErrorCluster:
    cluster_id: str
    category: str
    canonical_span: str
    judge_ids: list[str] = field(default_factory=list)
    severities: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def votes(self) -> int:
        return len(self.judge_ids)


@dataclass
class RowScore:
    test_id: str
    source_text: str
    hypothesis: str
    reference_text: str
    valid_judges: int
    vote_threshold: int
    accepted_errors: list[dict[str, Any]]
    rejected_errors: list[dict[str, Any]]
    severity_counts: dict[str, int]
    s_mqm: float | None = None
    chrf_score: float | None = None
    comet_score: float | None = None
    p_obj: float | None = None
    e_term: float | None = None
    term_hit_rate: float | None = None
    term_passed: bool = True
    s_final: float | None = None
    audit_status: str = "success"
    audit_message: str = ""
