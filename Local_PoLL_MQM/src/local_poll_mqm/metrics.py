from __future__ import annotations

import difflib
import importlib
from typing import Any, Mapping


class ObjectiveMetricsEngine:
    def __init__(
        self,
        metric_mode: str,
        preferred_language: str,
        comet_model_name: str,
        no_reference_chrf: float,
        no_reference_comet: float,
        comet_batch_size: int = 16,
    ) -> None:
        self.metric_mode = str(metric_mode).lower()
        self.preferred_language = preferred_language
        self.comet_model_name = comet_model_name
        self.no_reference_chrf = float(no_reference_chrf)
        self.no_reference_comet = float(no_reference_comet)
        self.comet_batch_size = max(1, int(comet_batch_size))

        self._sacrebleu = None
        self._comet_model = None
        self._torch = None
        self._comet_ready = False

        if self.metric_mode not in {"auto", "proxy", "real"}:
            raise ValueError(f"Unsupported metric_mode: {metric_mode}")

        if self.metric_mode != "proxy":
            try:
                self._sacrebleu = importlib.import_module("sacrebleu")
            except Exception:
                self._sacrebleu = None
                if self.metric_mode == "real":
                    raise

    @staticmethod
    def _torch_cuda_available(torch_module: Any) -> bool:
        cuda_module = getattr(torch_module, "cuda", None)
        is_available = getattr(cuda_module, "is_available", None)
        if callable(is_available):
            try:
                return bool(is_available())
            except Exception:
                return False
        return False

    def _ensure_comet(self) -> None:
        if self._comet_model is not None:
            return
        
        # If we already tried and failed, we can try again if it's not "real" mode, 
        # but let's use a simple throttler to avoid spamming.
        if hasattr(self, "_comet_last_fail_time") and self._comet_ready:
            import time
            if time.time() - self._comet_last_fail_time < 30: # 30s gap
                return

        self._comet_ready = True
        if self.metric_mode not in {"auto", "real"}:
            return

        try:
            comet_module = importlib.import_module("comet")
            self._torch = importlib.import_module("torch")
            download_model = getattr(comet_module, "download_model")
            load_from_checkpoint = getattr(comet_module, "load_from_checkpoint")
            
            import os
            if os.path.exists(self.comet_model_name) and os.path.isfile(self.comet_model_name):
                ckpt = self.comet_model_name
                model_display_name = os.path.basename(ckpt)
            else:
                ckpt = download_model(self.comet_model_name)
                model_display_name = self.comet_model_name
                
            model = load_from_checkpoint(ckpt)
            
            device = "cuda" if self._torch_cuda_available(self._torch) else "cpu"
            self._comet_model = model.to(device)
            print(f"[metrics]: COMET 模型加载成功: {model_display_name} (设备: {device})")
        except Exception as exc:
            self._comet_model = None
            self._torch = None
            self._comet_last_fail_time = __import__("time").time()
            if self.metric_mode == "real":
                raise RuntimeError(f"[metrics]: 实测模式下 COMET 加载失败: {exc}") from exc
            print(f"[metrics]: COMET 暂时不可用，降级至 chrF/proxy 模式: {exc}")

    def _simple_ratio(self, hyp: str, ref: str) -> float:
        return difflib.SequenceMatcher(a=ref, b=hyp).ratio() * 100.0

    def score(
        self,
        status: str,
        source_text: str,
        hypothesis_text: str,
        reference_text: str,
    ) -> dict[str, float]:
        if not hypothesis_text.strip():
            return {"chrf_score": 0.0, "comet_score": 0.0}

        if not reference_text.strip():
            return {
                "chrf_score": round(self.no_reference_chrf, 4),
                "comet_score": round(self.no_reference_comet, 4),
            }

        chrf_score = 0.0
        if self._sacrebleu is not None:
            try:
                chrf_score = float(self._sacrebleu.sentence_chrf(hypothesis_text, [reference_text]).score)
            except Exception:
                chrf_score = self._simple_ratio(hypothesis_text, reference_text)
        else:
            chrf_score = self._simple_ratio(hypothesis_text, reference_text)

        comet_score = 0.0
        if self.metric_mode in {"auto", "real"}:
            self._ensure_comet()

        if self._comet_model is not None and self._torch is not None:
            try:
                pred = self._comet_model.predict(
                    [{"src": source_text, "mt": hypothesis_text, "ref": reference_text}],
                    batch_size=1,
                    gpus=1 if self._torch_cuda_available(self._torch) else 0,
                )
                comet_score = float(pred.scores[0]) * 100.0
            except Exception:
                comet_score = chrf_score
        else:
            comet_score = chrf_score

        return {
            "chrf_score": round(chrf_score, 4),
            "comet_score": round(comet_score, 4),
        }

    def score_batch(
        self,
        batch: list[dict[str, str]],
    ) -> list[dict[str, float]]:
        results = [{"chrf_score": 0.0, "comet_score": 0.0} for _ in range(len(batch))]
        
        valid_indices = []
        comet_inputs = []
        
        for i, item in enumerate(batch):
            hyp = item.get("hypothesis_text", "")
            ref = item.get("reference_text", "")
            src = item.get("source_text", "")
            
            if not hyp.strip():
                continue
                
            if not ref.strip():
                results[i] = {
                    "chrf_score": round(self.no_reference_chrf, 4),
                    "comet_score": round(self.no_reference_comet, 4),
                }
                continue
                
            chrf_score = 0.0
            if self._sacrebleu is not None:
                try:
                    chrf_score = float(self._sacrebleu.sentence_chrf(hyp, [ref]).score)
                except Exception:
                    chrf_score = self._simple_ratio(hyp, ref)
            else:
                chrf_score = self._simple_ratio(hyp, ref)
                
            results[i]["chrf_score"] = round(chrf_score, 4)
            results[i]["comet_score"] = round(chrf_score, 4)
            
            valid_indices.append(i)
            comet_inputs.append({"src": src, "mt": hyp, "ref": ref})
            
        if not comet_inputs:
            return results
            
        if self.metric_mode in {"auto", "real"}:
            self._ensure_comet()
            
        if self._comet_model is not None and self._torch is not None:
            try:
                pred = self._comet_model.predict(
                    comet_inputs,
                    batch_size=self.comet_batch_size,
                    gpus=1 if self._torch_cuda_available(self._torch) else 0,
                )
                for idx, idx_in_batch in enumerate(valid_indices):
                    comet_score = float(pred.scores[idx]) * 100.0
                    results[idx_in_batch]["comet_score"] = round(comet_score, 4)
            except Exception as e:
                print(f"[metrics]: 批次 COMET 推断失败: {e}。将降级使用 chrF 指标。")
                
        return results
