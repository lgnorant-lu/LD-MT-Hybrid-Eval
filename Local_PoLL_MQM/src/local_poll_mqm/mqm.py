from __future__ import annotations

import difflib
import re
from collections import Counter
from statistics import median
from typing import Iterable

from .types import (
    SEVERITY_WEIGHTS,
    WEIGHT_TO_SEVERITY,
    ErrorCluster,
    JudgeDecision,
    JudgeError,
)

# ------------------------------------------------------------------
# Improved token counting: CJK ideographs (Chinese), Hiragana,
# Katakana, Hangul, and fullwidth punctuation are each counted as
# one unit.  Latin/Cyrillic words (runs of alphanumerics) are
# counted as one unit per word.  This eliminates the previous bias
# that penalised Chinese/Japanese/Korean translations unfairly.
# ------------------------------------------------------------------
_CJK_CHAR_RANGES = (
    r"\u4e00-\u9fff"  # CJK Unified Ideographs (Chinese)
    r"\u3400-\u4dbf"  # CJK Extension A
    r"\u3040-\u309f"  # Hiragana
    r"\u30a0-\u30ff"  # Katakana
    r"\uac00-\ud7af"  # Hangul Syllables
    r"\uff01-\uff60"  # Fullwidth Latin / CJK punctuation
)
TOKEN_PATTERN = re.compile(rf"[A-Za-z0-9_]+|[{_CJK_CHAR_RANGES}]")


def count_units(text: str) -> int:
    """Return the number of semantic units in *text*.

    Latin words count as 1 unit each; each CJK / Kana / Hangul
    character counts as 1 unit.  The minimum return value is 1.
    """
    return max(1, len(TOKEN_PATTERN.findall(str(text))))


def normalize_severity(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in SEVERITY_WEIGHTS:
        return key
    # fallback to nearest reasonable bucket
    if key in {"low", "minor_error", "small"}:
        return "minor"
    if key in {"high", "severe"}:
        return "major"
    if key in {"fatal", "blocker"}:
        return "critical"
    return "minor"


def _similar_span(a: str, b: str, threshold: float) -> bool:
    sa = str(a).strip().lower()
    sb = str(b).strip().lower()
    if not sa or not sb:
        return False
    if sa in sb or sb in sa:
        return True
    ratio = difflib.SequenceMatcher(a=sa, b=sb).ratio()
    return ratio >= threshold


def cluster_errors(
    decisions: Iterable[JudgeDecision], overlap_threshold: float
) -> list[ErrorCluster]:
    clusters: list[ErrorCluster] = []
    next_id = 1

    for decision in decisions:
        if not decision.ok:
            continue

        for err in decision.errors:
            placed = False
            category = str(err.category or "other").strip().lower()
            span = str(err.span or "").strip()
            severity = normalize_severity(err.severity)
            reason = str(err.reason or "").strip()

            if not span:
                continue

            for cluster in clusters:
                if cluster.category != category:
                    continue
                if not _similar_span(cluster.canonical_span, span, overlap_threshold):
                    continue

                if decision.judge_id not in cluster.judge_ids:
                    cluster.judge_ids.append(decision.judge_id)
                    cluster.severities.append(severity)
                    if reason:
                        cluster.reasons.append(reason)
                placed = True
                break

            if placed:
                continue

            clusters.append(
                ErrorCluster(
                    cluster_id=f"cluster_{next_id:04d}",
                    category=category,
                    canonical_span=span,
                    judge_ids=[decision.judge_id],
                    severities=[severity],
                    reasons=[reason] if reason else [],
                )
            )
            next_id += 1

    return clusters


def resolve_vote_threshold(requested_threshold: int, jury_size: int) -> int:
    if jury_size <= 0:
        return 1
    majority = (jury_size // 2) + 1
    if requested_threshold <= 0:
        return majority
    if requested_threshold > jury_size:
        return majority
    return requested_threshold


def arbitrate_errors(
    decisions: list[JudgeDecision],
    vote_threshold: int,
    overlap_threshold: float,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    clusters = cluster_errors(decisions, overlap_threshold)
    accepted: list[dict] = []
    rejected: list[dict] = []

    severity_counter: Counter[str] = Counter({"minor": 0, "major": 0, "critical": 0})

    for cluster in clusters:
        unique_judge_ids = []
        seen = set()
        for j in cluster.judge_ids:
            if j in seen:
                continue
            seen.add(j)
            unique_judge_ids.append(j)

        votes = len(unique_judge_ids)
        if votes < vote_threshold:
            rejected.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "category": cluster.category,
                    "span": cluster.canonical_span,
                    "votes": votes,
                    "judge_ids": unique_judge_ids,
                    "reason": "VOTES_BELOW_THRESHOLD",
                }
            )
            continue

        weights = sorted(
            SEVERITY_WEIGHTS[normalize_severity(s)] for s in cluster.severities
        )
        final_weight = int(median(weights)) if weights else 1
        final_severity = WEIGHT_TO_SEVERITY.get(final_weight, "minor")

        row = {
            "cluster_id": cluster.cluster_id,
            "category": cluster.category,
            "span": cluster.canonical_span,
            "votes": votes,
            "judge_ids": unique_judge_ids,
            "raw_severities": cluster.severities,
            "final_severity": final_severity,
            "final_weight": final_weight,
            "reasons": cluster.reasons,
        }
        accepted.append(row)
        severity_counter[final_severity] += 1

    return accepted, rejected, dict(severity_counter)


def compute_s_mqm(
    severity_counts: dict[str, int],
    source_text: str,
    hypothesis_text: str,
) -> float:
    c_minor = int(severity_counts.get("minor", 0))
    c_major = int(severity_counts.get("major", 0))
    c_critical = int(severity_counts.get("critical", 0))

    penalty = (c_minor * 1) + (c_major * 5) + (c_critical * 25)
    length_unit = max(1, max(count_units(source_text), count_units(hypothesis_text)))

    return max(0.0, min(100.0, 100.0 - ((penalty / length_unit) * 100.0)))


def compute_objective_penalty(
    chrf_score: float,
    comet_score: float,
    omega_1: float,
    omega_2: float,
    threshold: float,
    alpha: float,
) -> float:
    objective = (omega_1 * chrf_score) + (omega_2 * comet_score)
    return alpha * max(0.0, threshold - objective)


def compute_s_final(
    s_mqm: float,
    p_obj: float,
    e_term: float,
    delta: float,
) -> float:
    raw = (s_mqm - p_obj) * (1.0 - delta * e_term)
    return max(0.0, min(100.0, raw))
