# @title --- MODULE: audit_evaluator ---

import difflib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .config import B2_SCHEMA_VERSION, SCORE_SPEC_VERSION
    from .schemas import AuditMetrics, AuditReport, AuditScoreItem, normalize_term_rules, validate_dataset_bundle
except ImportError:
    from config import B2_SCHEMA_VERSION, SCORE_SPEC_VERSION
    from schemas import AuditMetrics, AuditReport, AuditScoreItem, normalize_term_rules, validate_dataset_bundle


ObjectiveMetricProvider = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, float]]
ObjectiveMetricBatchProvider = Callable[
    [Mapping[str, Mapping[str, Any]], list[Mapping[str, Any]]],
    Mapping[str, Mapping[str, float]],
]
MQMProvider = Callable[[Mapping[str, Any], Mapping[str, Any]], float]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)



class VetoResult:
    """Holds the result of a regex-based term verification."""
    def __init__(
        self,
        passed: bool,
        hit_rate: float,
        penalty_score: float,
        details: dict[str, Any]
    ) -> None:
        self.passed = passed
        self.hit_rate = hit_rate
        self.penalty_score = penalty_score
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "hit_rate": self.hit_rate,
            "penalty_score": self.penalty_score,
            "details": self.details
        }


class AuditEvaluator:
    """Compute S_final from normalized raw inference results.
    
    Structure:
    1. Veto Gate (Regex/Glossary) - Fast, rule-based.
    2. Objective Metrics (chrf/COMET) - Heavy, embedding-based.
    3. Final Score Fusion.
    """

    def __init__(
        self,
        omega_1: float = 0.4,
        omega_2: float = 0.6,
        threshold: float = 60.0,
        alpha: float = 0.2,
        delta: float = 0.5,
    ) -> None:
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.threshold = threshold
        self.alpha = alpha
        self.delta = delta

    def compute_objective_penalty(self, chrf_score: float, comet_score: float) -> float:
        objective = (self.omega_1 * chrf_score) + (self.omega_2 * comet_score)
        return self.alpha * max(0.0, self.threshold - objective)

    def compute_s_final(self, s_mqm: float, p_obj: float, e_term: float) -> float:
        gated = (s_mqm - p_obj) * (1.0 - self.delta * e_term)
        return max(0.0, gated)

    @staticmethod
    def _use_word_boundary(language: str) -> bool:
        lang = str(language).lower()
        return lang in {"en", "de", "fr", "it", "es", "pt", "nl", "sv", "no", "da", "fi"}

    @staticmethod
    def _contains_token(text: str, token: str, language: str) -> bool:
        if not token:
            return False

        lowered_text = text.lower()
        lowered_token = token.lower()

        if AuditEvaluator._use_word_boundary(language):
            return re.search(rf"\\b{re.escape(lowered_token)}\\b", lowered_text) is not None
        return lowered_token in lowered_text

    @staticmethod
    def _extract_term_checks(term_rules: Mapping[str, Any], language: str) -> tuple[list[str], list[str]]:
        veto = term_rules.get("veto_validation", {})
        
        # 默认使用通用列表
        expected = list(veto.get("expected_keywords", []))
        forbidden = list(veto.get("forbidden_keywords", []))

        # 如果有特定语言的映射，则覆盖
        multilingual_expected = veto.get("multilingual_expected", {})
        multilingual_forbidden = veto.get("multilingual_forbidden", {})
        if isinstance(multilingual_expected, Mapping) and language in multilingual_expected:
            expected = list(multilingual_expected.get(language, []))
        if isinstance(multilingual_forbidden, Mapping) and language in multilingual_forbidden:
            forbidden = list(multilingual_forbidden.get(language, []))

        return [str(x) for x in expected], [str(x) for x in forbidden]

    def evaluate_veto_gate(self, term_rules: Mapping[str, Any], hypothesis: str, language: str) -> VetoResult:
        """Stage 1: Pure Logic/Regex Veto Gate."""
        rules = normalize_term_rules(term_rules)
        if not rules.get("is_active"):
            return VetoResult(True, 1.0, 0.0, {"active": False, "reason": "Glossary rules not active"})

        expected, forbidden = self._extract_term_checks(rules, language)
        expected_hits = [token for token in expected if self._contains_token(hypothesis, token, language)]
        forbidden_hits = [token for token in forbidden if self._contains_token(hypothesis, token, language)]

        missing_expected = [token for token in expected if token not in expected_hits]
        has_fatal = bool(forbidden_hits)
        passed = not has_fatal and not missing_expected

        if expected:
            hit_rate = len(expected_hits) / max(len(expected), 1)
        else:
            hit_rate = 1.0

        # penalty_score (e_term): 1.0 if fatal, otherwise proportional to missing expected
        e_term = 1.0 if has_fatal else 1.0 - hit_rate

        details = {
            "active": True,
            "expected_hits": expected_hits,
            "missing_expected": missing_expected,
            "forbidden_hits": forbidden_hits,
            "fatal_violations": [f"Forbidden '{t}' found" for t in forbidden_hits],
            "warnings": [f"Missing '{t}'" for t in missing_expected],
            "checked_language": language,
            "veto_tier": "[FATAL_VETO]" if has_fatal else ("[WARNING]" if missing_expected else "[PASSED]")
        }
        return VetoResult(passed, round(hit_rate, 4), round(e_term, 4), details)

    def run_full_audit(
        self,
        raw_row: Mapping[str, Any],
        dataset_item: Mapping[str, Any],
        preferred_language: str,
        metric_provider: ObjectiveMetricProvider | None = None
    ) -> dict[str, Any]:
        """Orchestrate the full audit pipeline for a single row."""
        test_id = str(raw_row.get("test_id", ""))
        hypotheses = raw_row.get("hypotheses", {})
        hyp = str(hypotheses.get(preferred_language, "")).strip()
        
        # 1. Veto Step
        term_rules = dataset_item.get("term_rules", {})
        veto = self.evaluate_veto_gate(term_rules, hyp, preferred_language)
        
        # 2. Objective Metric Step
        metrics = {"chrf_score": 0.0, "comet_score": 0.0}
        if metric_provider:
            metrics.update(metric_provider(dataset_item, raw_row))
        
        # 3. Penalty & Final Calculation
        # Assume s_mqm (LLM eval) defaults to 100 if not present, pending stage 2 refactor
        s_mqm = float(raw_row.get("mqm_score", 100.0))
        p_obj = self.compute_objective_penalty(metrics["chrf_score"], metrics["comet_score"])
        s_final = self.compute_s_final(s_mqm, p_obj, veto.penalty_score)
        
        return {
            "test_id": test_id,
            "status": "AUDITED",
            "s_final": round(s_final, 4),
            "s_mqm": s_mqm,
            "p_obj": round(p_obj, 4),
            "e_term": veto.penalty_score,
            "metrics": metrics,
            "veto": veto.to_dict(),
            "timestamp": _utc_now_iso()
        }

    @staticmethod
    def _default_objective_metrics(dataset_item: Mapping[str, Any], raw_row: Mapping[str, Any]) -> dict[str, float]:
        references = dataset_item.get("reference_translations", {})
        reference_text = ""
        if isinstance(references, Mapping):
            if isinstance(references.get("en"), str) and references.get("en", "").strip():
                reference_text = str(references.get("en", "")).strip()
            else:
                for value in references.values():
                    if isinstance(value, str) and value.strip():
                        reference_text = value.strip()
                        break

        hypotheses = raw_row.get("hypotheses", {})
        hypothesis_text = ""
        if isinstance(hypotheses, Mapping):
            if isinstance(hypotheses.get("en"), str) and hypotheses.get("en", "").strip():
                hypothesis_text = str(hypotheses.get("en", "")).strip()
            else:
                for value in hypotheses.values():
                    if isinstance(value, str) and value.strip():
                        hypothesis_text = value.strip()
                        break

        status = str(raw_row.get("status", ""))
        if status != "SUCCESS" or not hypothesis_text:
            return {"chrf_score": 0.0, "comet_score": 0.0}

        if not reference_text:
            # For no-reference blocks, keep a neutral baseline instead of full score.
            return {"chrf_score": 65.0, "comet_score": 65.0}

        ratio = difflib.SequenceMatcher(a=reference_text, b=hypothesis_text).ratio() * 100.0
        comet_proxy = max(0.0, min(100.0, ratio * 0.95 + 2.0))
        return {"chrf_score": round(ratio, 4), "comet_score": round(comet_proxy, 4)}

    @staticmethod
    def _default_mqm(dataset_item: Mapping[str, Any], raw_row: Mapping[str, Any]) -> float:
        status = str(raw_row.get("status", ""))
        if status != "SUCCESS":
            return 0.0

        references = dataset_item.get("reference_translations", {})
        reference_text = ""
        if isinstance(references, Mapping):
            if isinstance(references.get("en"), str) and references.get("en", "").strip():
                reference_text = str(references.get("en", "")).strip()
            else:
                for value in references.values():
                    if isinstance(value, str) and value.strip():
                        reference_text = value.strip()
                        break

        hypotheses = raw_row.get("hypotheses", {})
        hypothesis_text = ""
        if isinstance(hypotheses, Mapping):
            hypothesis_text = str(hypotheses.get("en", "")).strip() or ""
            if not hypothesis_text:
                for value in hypotheses.values():
                    if isinstance(value, str) and value.strip():
                        hypothesis_text = value.strip()
                        break

        if not hypothesis_text:
            return 0.0
        if not reference_text:
            return 75.0

        length_ratio = abs(len(hypothesis_text) - len(reference_text)) / max(len(reference_text), 1)
        penalty = min(60.0, length_ratio * 100.0)
        return max(0.0, 100.0 - penalty)

    def audit_block(
        self,
        raw_inference_path: Path,
        dataset_path: Path,
        output_path: Path,
        objective_metric_provider: ObjectiveMetricProvider | None = None,
        objective_metric_batch_provider: ObjectiveMetricBatchProvider | None = None,
        mqm_provider: MQMProvider | None = None,
        score_spec_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw_report = _load_json(raw_inference_path)
        dataset = _load_json(dataset_path)

        errors = validate_dataset_bundle(dataset)
        if errors:
            raise ValueError(f"Invalid dataset schema at {dataset_path}: {errors}")

        by_test_id = {str(item["test_id"]): item for item in dataset["items"]}
        score_rows: list[AuditScoreItem] = []

        objective_provider = objective_metric_provider or self._default_objective_metrics
        mqm_score_provider = mqm_provider or self._default_mqm
        objective_metric_map: dict[str, Mapping[str, float]] = {}

        if objective_metric_batch_provider is not None:
            try:
                batch_values = objective_metric_batch_provider(by_test_id, list(raw_report.get("results", [])))
                objective_metric_map = {str(k): v for k, v in dict(batch_values).items()}
            except Exception as exc:
                objective_metric_map = {}
                print(f"[audit] objective_metric_batch_provider failed, fallback to row provider: {exc}")

        active_term_items = 0
        passed_term_items = 0
        fatal_term_items = 0
        warning_term_items = 0

        for row in raw_report.get("results", []):
            test_id = str(row.get("test_id", ""))
            dataset_item = by_test_id.get(test_id, {})
            hypotheses = row.get("hypotheses", {})
            if not isinstance(hypotheses, Mapping):
                hypotheses = {}

            language = "en"
            hypothesis_text = str(hypotheses.get(language, ""))
            if not hypothesis_text and hypotheses:
                language = next(iter(hypotheses.keys()))
                hypothesis_text = str(hypotheses.get(language, ""))

            term_rules = normalize_term_rules(dataset_item.get("term_rules"))
            # 修复方法名不匹配问题：底层方法是 evaluate_veto_gate，而循环中使用的是 _term_gate
            veto_res = self.evaluate_veto_gate(term_rules, hypothesis_text, language)
            term_details = veto_res.to_dict()
            term_gate_details = term_details.get("details", {}) if isinstance(term_details, Mapping) else {}
            if not isinstance(term_gate_details, Mapping):
                term_gate_details = {}

            term_passed = veto_res.passed
            term_hit_rate = veto_res.hit_rate
            e_term = veto_res.penalty_score
            
            if bool(term_gate_details.get("active")):
                active_term_items += 1
                if term_passed:
                    passed_term_items += 1
                elif term_gate_details.get("veto_tier") == "[FATAL_VETO]":
                    fatal_term_items += 1
                elif term_gate_details.get("veto_tier") == "[WARNING]":
                    warning_term_items += 1

            objective_metrics = objective_metric_map.get(test_id) or objective_provider(dataset_item, row)
            chrf_score = float(objective_metrics.get("chrf_score", 0.0))
            comet_score = float(objective_metrics.get("comet_score", 0.0))
            p_obj = self.compute_objective_penalty(chrf_score=chrf_score, comet_score=comet_score)
            s_mqm = float(mqm_score_provider(dataset_item, row))
            s_final = self.compute_s_final(s_mqm=s_mqm, p_obj=p_obj, e_term=e_term)

            # --- [Pass-through Context for Audited Reports] ---
            term_rules_raw = dataset_item.get("term_rules", {})
            term_context = ""
            if isinstance(term_rules_raw, Mapping):
                term_context = str(term_rules_raw.get("llm_instruction", "")).strip()

            score_rows.append(
                AuditScoreItem(
                    test_id=test_id,
                    metrics=AuditMetrics(
                        chrf_score=round(chrf_score, 4),
                        comet_score=round(comet_score, 4),
                        objective_penalty_pobj=round(p_obj, 4),
                        term_veto_passed=term_passed,
                        term_hit_rate=round(term_hit_rate, 4),
                        mqm_score=round(s_mqm, 4),
                        s_final=round(s_final, 4),
                    ),
                    veto_details=term_details,
                    hypotheses={str(k): str(v) for k, v in hypotheses.items()},
                    source_text=str(dataset_item.get("source_text", "")),
                    reference_translations={str(k): str(v) for k, v in dict(dataset_item.get("reference_translations", {})).items()},
                    term_context=term_context,
                    audit_tags=list(dataset_item.get("audit_tags", [])),
                )
            )

        avg_s_final = 0.0
        avg_chrf_score = 0.0
        avg_comet_score = 0.0
        avg_mqm_score = 0.0
        avg_term_hit_rate = 0.0
        if score_rows:
            avg_s_final = sum(row.metrics.s_final for row in score_rows) / len(score_rows)
            avg_chrf_score = sum(row.metrics.chrf_score for row in score_rows) / len(score_rows)
            avg_comet_score = sum(row.metrics.comet_score for row in score_rows) / len(score_rows)
            avg_mqm_score = sum(row.metrics.mqm_score for row in score_rows) / len(score_rows)
            avg_term_hit_rate = sum(row.metrics.term_hit_rate for row in score_rows) / len(score_rows)

        if active_term_items > 0:
            overall_veto_pass_rate = passed_term_items / active_term_items
            fatal_veto_rate = fatal_term_items / active_term_items
            warning_veto_rate = warning_term_items / active_term_items
        else:
            overall_veto_pass_rate = 1.0
            fatal_veto_rate = 0.0
            warning_veto_rate = 0.0

        score_spec: dict[str, Any] = {
            "version": SCORE_SPEC_VERSION,
            "objective_provider": getattr(objective_provider, "__name__", str(objective_provider)),
            "objective_batch_provider": (
                getattr(objective_metric_batch_provider, "__name__", str(objective_metric_batch_provider))
                if objective_metric_batch_provider
                else ""
            ),
            "mqm_provider": getattr(mqm_score_provider, "__name__", str(mqm_score_provider)),
        }
        if isinstance(score_spec_overrides, Mapping):
            for key, value in score_spec_overrides.items():
                if value is None:
                    continue
                score_spec[str(key)] = value

        payload = AuditReport(
            audit_meta={
                "model_id": raw_report.get("run_meta", {}).get("model_id", "unknown"),
                "test_block": raw_report.get("run_meta", {}).get("test_block", "unknown"),
                "timestamp": _utc_now_iso(),
                "schema_version": B2_SCHEMA_VERSION,
                "score_spec": score_spec,
                "formula": {
                    "omega_1": self.omega_1,
                    "omega_2": self.omega_2,
                    "threshold": self.threshold,
                    "alpha": self.alpha,
                    "delta": self.delta,
                },
            },
            scores=score_rows,
            block_summary={
                "total_samples": len(score_rows),
                "avg_s_final": round(avg_s_final, 4),
                "avg_chrf_score": round(avg_chrf_score, 4),
                "avg_comet_score": round(avg_comet_score, 4),
                "avg_mqm_score": round(avg_mqm_score, 4),
                "avg_term_hit_rate": round(avg_term_hit_rate, 4),
                "overall_veto_pass_rate": round(overall_veto_pass_rate, 4),
                "fatal_veto_rate": round(fatal_veto_rate, 4),
                "warning_veto_rate": round(warning_veto_rate, 4),
            },
        )

        as_dict = payload.to_dict()
        as_dict["scores"] = [
            {
                "test_id": row.test_id,
                "metrics": asdict(row.metrics),
                "veto_details": row.veto_details,
            }
            for row in score_rows
        ]
        _save_json(output_path, as_dict)
        return as_dict

# %%