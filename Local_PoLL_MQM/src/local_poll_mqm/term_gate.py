from __future__ import annotations

import re
from typing import Any, Mapping


def _use_word_boundary(language: str) -> bool:
    return str(language).lower() in {"en", "de", "fr", "it", "es", "pt", "nl", "sv", "no", "da", "fi"}


def _contains_token(text: str, token: str, language: str) -> bool:
    if not token:
        return False

    lowered_text = text.lower()
    lowered_token = token.lower()
    if _use_word_boundary(language):
        pattern = r"\b" + re.escape(lowered_token) + r"\b"
        return re.search(pattern, lowered_text) is not None
    return lowered_token in lowered_text


def _normalize_term_rules(raw_rules: Mapping[str, Any] | None) -> dict[str, Any]:
    # Keep logic self-contained for the new local pipeline.
    defaults = {
        "is_active": False,
        "llm_instruction": "",
        "veto_validation": {
            "expected_keywords": [],
            "forbidden_keywords": [],
            "multilingual_expected": {},
            "multilingual_forbidden": {},
        },
    }
    if not isinstance(raw_rules, Mapping):
        return defaults

    rules = dict(defaults)
    rules["is_active"] = bool(raw_rules.get("is_active", False))
    rules["llm_instruction"] = str(raw_rules.get("llm_instruction", "")).strip()

    veto = raw_rules.get("veto_validation", {})
    if isinstance(veto, Mapping):
        rules["veto_validation"] = {
            "expected_keywords": list(veto.get("expected_keywords", [])),
            "forbidden_keywords": list(veto.get("forbidden_keywords", [])),
            "multilingual_expected": dict(veto.get("multilingual_expected", {})),
            "multilingual_forbidden": dict(veto.get("multilingual_forbidden", {})),
        }
    return rules


def evaluate_term_gate(term_rules: Mapping[str, Any] | None, hypothesis: str, language: str) -> dict[str, Any]:
    rules = _normalize_term_rules(term_rules)
    if not rules.get("is_active"):
        return {
            "active": False,
            "passed": True,
            "hit_rate": 1.0,
            "e_term": 0.0,
            "expected_hits": [],
            "missing_expected": [],
            "forbidden_hits": [],
            "veto_tier": "[INACTIVE]",
        }

    veto = rules.get("veto_validation", {})
    expected = list(veto.get("expected_keywords", []))
    forbidden = list(veto.get("forbidden_keywords", []))

    multilingual_expected = veto.get("multilingual_expected", {})
    multilingual_forbidden = veto.get("multilingual_forbidden", {})
    if isinstance(multilingual_expected, Mapping) and language in multilingual_expected:
        expected = list(multilingual_expected.get(language, []))
    if isinstance(multilingual_forbidden, Mapping) and language in multilingual_forbidden:
        forbidden = list(multilingual_forbidden.get(language, []))

    expected = [str(x) for x in expected if str(x).strip()]
    forbidden = [str(x) for x in forbidden if str(x).strip()]

    expected_hits = [tok for tok in expected if _contains_token(hypothesis, tok, language)]
    missing_expected = [tok for tok in expected if tok not in expected_hits]
    forbidden_hits = [tok for tok in forbidden if _contains_token(hypothesis, tok, language)]

    has_fatal = bool(forbidden_hits)
    if expected:
        hit_rate = len(expected_hits) / max(1, len(expected))
    else:
        hit_rate = 1.0

    e_term = 1.0 if has_fatal else 1.0 - hit_rate
    passed = (not has_fatal) and (not missing_expected)
    tier = "[FATAL_VETO]" if has_fatal else ("[WARNING]" if missing_expected else "[PASSED]")

    return {
        "active": True,
        "passed": passed,
        "hit_rate": round(hit_rate, 4),
        "e_term": round(e_term, 4),
        "expected_hits": expected_hits,
        "missing_expected": missing_expected,
        "forbidden_hits": forbidden_hits,
        "veto_tier": tier,
    }
