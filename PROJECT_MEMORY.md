# PROJECT_MEMORY — L-Station MT Evaluation (Expanded)

This document is an expanded memory map of Stage 0–3 data, scripts, outputs, and **what each file does**. It is intended to support audits, re-entry, and Stage 3 planning.

---

## Repository Roots (Windows)
- `LinuxDo` (project root)

---

# Stage 0 — 原初数据采集与清洗 (Metadatas)
**Purpose:** ingest raw L-Station community data, extract UI text & slang contexts, build baseline datasets, glossary, and slang gold set.  
**Outcome:** canonical dataset + glossary + slang benchmark + reference translations + schema map.

## Primary folders
- Snapshot (archived input): `Metadatas-20260316T020430Z-3-001/Metadatas`
- Working directory: `Metadatas/`

## Key data files (outputs)
- `Metadatas/Benchmark_V1_Dataset.json`
  - **What it is:** canonical benchmark dataset; merged records used across Stage 1–3.
  - **Data structure (high-level):** list of items with `id`, `source_text`, `target_lang`, `domain`, `tags`, `metadata`, plus optional `slang_context` fields.
- `Metadatas/Benchmark_Reference_Translations.json`
  - **What it is:** reference translations for evaluation (gold translations).
- `Metadatas/Benchmark_Slang_Context.json`
  - **What it is:** slang contexts and surface forms extracted from posts.
- `Metadatas/Benchmark_Slang_Golden_V1_Final.json`
  - **What it is:** validated slang gold set; disambiguation mapping for slang.
- `Metadatas/L_Station_Glossary.json`
  - **What it is:** key terms + preferred translations (terminology enforcement).
- `Metadatas/Slang_Filtered_Benchmark_Raw.json`
  - **What it is:** filtered slang candidate pool before finalization.
- `Metadatas/extracted_keywords.json`
  - **What it is:** keywords extracted from posts for sampling & filter logic.
- `Metadatas/l_station_posts.jsonl`
  - **What it is:** raw collected community posts (full).
- `Metadatas/l_station_posts_pure.jsonl`
  - **What it is:** cleaned / de-noised posts used by extraction.

## Core scripts (Stage 0)
- `Metadatas/l_station_collector.py`
  - **Role:** raw post collection (source data ingestion).
- `Metadatas/extract_reference_keys.py`
  - **Role:** pulls UI / key phrases for reference mapping.
- `Metadatas/build_reversible_html_tree.py`
  - **Role:** builds reversible HTML tree for UI text normalization.
- `Metadatas/apply_html_translations.py`
  - **Role:** applies translations back into HTML structure.
- `Metadatas/build_benchmark_v1.py`
  - **Role:** merges all inputs into `Benchmark_V1_Dataset.json`.
- `Metadatas/test_apply_and_schema.py`
  - **Role:** sanity checks for HTML round-trip and schema.

## Schema map
- `Metadatas/data_files_schema_tree.md`
  - **Role:** schema outline for Stage 0 outputs (data shape reference).

---

# Stage 1 — 模型推理 + 初审 (arena_core → Benchmarks)
**Purpose:** run model inference, collect raw outputs, compute initial audits, build leaderboard.  
**Outcome:** per-model raw outputs + per-block audit + global leaderboard metrics.

## Core logic (code)
Location: `IPYNB-PY/arena_core/`

- `config.py`
  - **Role:** test block list, target languages, model list.
- `filesystem.py`
  - **Role:** path conventions / standardized output structure.
- `schemas.py`
  - **Role:** dataset and report JSON schemas.
- `inference_runner.py`
  - **Role:** runs model inference, handles JSON repair, and writes raw outputs.
- `audit_evaluator.py`
  - **Role:** computes objective metrics, term gate, and `S_final`.
- `aggregator.py`
  - **Role:** builds global leaderboard summary.

## Stage 1 outputs (Benchmarks)
- `Benchmarks-20260316T020328Z-3-001/Benchmarks/01_Datasets/*.json`
  - **Role:** dataset snapshots used by the run.
- `Benchmarks-20260316T020328Z-3-001/Benchmarks/02_Experiment_Runs/<model>/raw_inference/*_raw.json`
  - **Role:** raw model outputs per block.
- `Benchmarks-20260316T020328Z-3-001/Benchmarks/02_Experiment_Runs/<model>/audited_reports/*_audit.json`
  - **Role:** per-block audit results (metrics + term gate).
- `Benchmarks-20260316T020328Z-3-001/Benchmarks/03_Leaderboard/Global_Metrics_Summary.json`
  - **Role:** aggregated leaderboard (objective metrics + `S_final`).
- `Benchmarks-20260316T020328Z-3-001/Benchmarks/run_manifest.json`
  - **Role:** run metadata, configurations, and trace.

---

# Stage 2 — 本地 PoLL + MQM 多裁判评审 (Local_PoLL_MQM)
**Purpose:** use 5 LLM judges for MQM-style error extraction and consensus arbitration.  
**Outcome:** consensus MQM scores + per-block audits + final leaderboards.

## Core logic (code)
Location: `Local_PoLL_MQM/src/local_poll_mqm/`

- `pipeline.py`
  - **Role:** end-to-end pipeline (inference → audit → scoring).
- `mqm.py`
  - **Role:** consensus arbitration, severity weights, `S_mqm` and `S_final`.
- `metrics.py`
  - **Role:** chrF / COMET integration.
- `term_gate.py`
  - **Role:** terminology veto logic.
- `io_utils.py`
  - **Role:** loading, extraction, and IO helpers.
- `config.py`
  - **Role:** `PipelineConfig` definitions and defaults.

## Main entry points
- `Local_PoLL_MQM/run_local_poll_mqm.py`
  - **Role:** standard CLI run for Stage 2.
- `Local_PoLL_MQM/src/local_poll_mqm/cli.py`
  - **Role:** CLI definitions and flags.

## Configuration
- `Local_PoLL_MQM/configs/elite_five_consensus.json`
  - **Role:** canonical 5-judge config.
  - **Judges:** `gpt_5_01`, `claude_4_6_01`, `qwen_3_5_sf`, `minimax_m2_5_sf`, `deepseek_v3_2_sf`
  - **Target models:** 15 candidate translation models.
  - **output_root:** `output/elite_five_integrated`

## Clean single-judge caches (seed material)
- Archive (source): `Local_PoLL_MQM/archive/original_single_runs/{gpt,claude,qwen,minimax,deepseek}_single_judge_test`
- Working copies (restored to):
  - `Local_PoLL_MQM/output/gpt_single_judge_test`
  - `Local_PoLL_MQM/output/claude_single_judge_test`
  - `Local_PoLL_MQM/output/qwen_single_judge_test`
  - `Local_PoLL_MQM/output/minimax_single_judge_test`
  - `Local_PoLL_MQM/output/deepseek_single_judge_test`

## Merge & audit utilities (Stage 2)
- `Local_PoLL_MQM/scripts/merge_dragon_balls.py`
  - **Role:** merge 5 single-judge caches into integrated checkpoints.
- `Local_PoLL_MQM/scripts/integrity_check.py`
  - **Role:** validate / clean caches (structure and judge slots).
- `Local_PoLL_MQM/scripts/deep_content_audit.py`
  - **Role:** content-level judge checks and discrepancy highlighting.
- `Local_PoLL_MQM/scripts/master_data_miner.py`
  - **Role:** consensus stats / divergence / CoT snippets extraction.

## Stage 2 outputs (integrated)
- `Local_PoLL_MQM/output/elite_five_integrated/checkpoints/<model>/<block>_judge_cache.json`
  - **Role:** judge caches (per-block, per-model) used for consensus.
- `Local_PoLL_MQM/output/elite_five_integrated/audited_reports/<model>/<block>_poll_mqm_audit.json`
  - **Role:** per-block MQM audit results (severity + consensus).
- `Local_PoLL_MQM/output/elite_five_integrated/leaderboard/Global_PoLL_MQM_Summary.json`
  - **Role:** aggregated MQM leaderboard.
- `Local_PoLL_MQM/output/elite_five_integrated/scheduler_diagnostic.log`
  - **Role:** scheduler diagnostics and errors.

---

# Stage 3 — 数据分析与图表 (Analysis Infra)
**Purpose:** post-hoc analysis and visualization based on Stage 2 outputs.  
**Outcome:** normalized datasets, deep analytics, static plots, and interactive wall.

## Existing analysis data & outputs
- `Local_PoLL_MQM/analysis_infra/archive/processed_visual_data.csv` (**archived legacy output**)
  - **Role:** legacy normalized table from `prepare_visual_data.py` (superseded by `dim_data`).
- `Local_PoLL_MQM/analysis_infra/dim_data/` (**Stage 3 normalized primitives**)
  - `dim_global_capability.csv` — per model & block averages (`avg_s_final`, `avg_s_mqm`, `avg_chrf`, `avg_comet`, `avg_p_obj`).
  - `dim_error_topology.csv` — per model & block error counts by `category` + `final_severity`.
  - `dim_slang_matrix.csv` — term-gate hits/misses (per model, per test_id, per term).
  - `dim_divergence_cases.json` — marginal pass/fail cases (votes=3 or 2) for audit & drill-down.
  - `dim_judge_consensus.json` — **derived diagnostics only**:
    - **vote_distribution** counts are aggregated from `accepted_errors` + `rejected_errors` (per error’s `votes` field).
    - **judge_stats** counts are aggregated from each error’s `judge_ids` list:
      - `proposed` = times a judge appears in any error (accepted or rejected).
      - `accepted` = times a judge appears in accepted errors.
      - `reliability_rate` = `accepted / proposed` (heuristic, **not a scientific judge reliability**; not for ranking).
- `Local_PoLL_MQM/analysis_infra/raw_stats/`
  - `consensus_summary.json` — consensus distribution and agreement rates.
  - `divergence_cases.json` — high-disagreement cases.
  - `cot_snippets.json` — representative CoT snippets for insight.
  - `structural_fidelity.csv` — structure fidelity metrics.
  - `slang_s_final_matrix.csv` — slang coverage by model.
- `Local_PoLL_MQM/analysis_infra/archive/` (**legacy artifacts, archived**)
  - `ANALYSIS_ENGINE_DESIGN.md`
  - `ANALYSIS_MEMO.md`
  - `Interactive_Compare_Wall.html`
  - `processed_visual_data.csv`
  - `plots/`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/` — canonical 10-plot set (01–09, with 08a/08b).
- `Local_PoLL_MQM/analysis_infra/archive/plots/` — **archived legacy plots** from older scripts.

## Analysis scripts
**Active (current pipeline)**
- `Local_PoLL_MQM/scripts/analysis_data_factory.py`
  - **Role:** builds Stage 3 `dim_data` primitives directly from Stage 2 `audited_reports` and leaderboard summary.
  - **Notes:** `dim_judge_consensus.json` is diagnostic-only; derived from per-error `votes` + `judge_ids` in `accepted_errors`/`rejected_errors`.
- `Local_PoLL_MQM/scripts/build_interactive_wall.py`
  - **Role:** builds V2 interactive wall (V1-style compare wall + charts driven by `dim_data`).
- `Local_PoLL_MQM/scripts/generate_canonical_plots.py`
  - **Role:** generates canonical 10-plot set (01–09, with 08a/08b).
- `Local_PoLL_MQM/scripts/generate_ultimate_slang_matrix.py`
  - **Role:** slang matrix analysis (source for heatmap logic).
- `Local_PoLL_MQM/scripts/calculate_structural_fidelity.py`
  - **Role:** measures structural fidelity.

**Stage 2 utility scripts (still kept)**
- `Local_PoLL_MQM/scripts/merge_dragon_balls.py`
- `Local_PoLL_MQM/scripts/integrity_check.py`
- `Local_PoLL_MQM/scripts/deep_content_audit.py`
- `Local_PoLL_MQM/scripts/audit_judge_failures.py`
- `Local_PoLL_MQM/scripts/judge_health_check.py`
- `Local_PoLL_MQM/scripts/elite_five_health_check.py`
- `Local_PoLL_MQM/scripts/final_four_audit.py`

**Archived (legacy / debug / old logic)**
- `Local_PoLL_MQM/scripts/archive/prepare_visual_data.py`
- `Local_PoLL_MQM/scripts/archive/generate_9_dimensions.py`
- `Local_PoLL_MQM/scripts/archive/generate_deep_analysis_plots.py`
- `Local_PoLL_MQM/scripts/archive/generate_interactive_wall.py`
- `Local_PoLL_MQM/scripts/archive/master_data_miner.py`
- `Local_PoLL_MQM/scripts/archive/analyze_scheduler_logs.py`
- `Local_PoLL_MQM/scripts/archive/peek_metadata.py`
- `Local_PoLL_MQM/scripts/archive/archive_clutter.py`
- `Local_PoLL_MQM/scripts/archive/simulate_gemini_concurrency.py`
- `Local_PoLL_MQM/scripts/archive/debug_*`
- `Local_PoLL_MQM/scripts/archive/test_*`

## Interactive wall
- `Local_PoLL_MQM/analysis_infra/Interactive_Compare_Wall_V2.html`
  - **Status:** current V2 (V1-style compare wall + charts driven by `dim_data`).
- `Local_PoLL_MQM/analysis_infra/archive/Interactive_Compare_Wall.html`
  - **Status:** archived legacy wall (pre-V2).

---

# Stage 3 — Deep Analysis Plan (审计 + 设计方向)
**Goal:** define a complete, multi-dimensional analysis system.  
**Focus:** not only static charts, but multi-view analytics and interactive comparisons.

## Proposed analysis dimensions (multi-level)
1. **Global performance layer**
   - Overall `S_final`, `S_mqm`, term gate rates, chrF/COMET.
   - Leaderboard stability across blocks.
2. **Judge consensus layer**
   - Agreement rate, judge reliability, divergence statistics.
   - Error category distributions and severity weights.
3. **Block robustness layer**
   - Per-block variance; identify fragile blocks.
4. **Slang & domain layer**
   - Slang disambiguation accuracy; slang coverage matrix.
   - Domain-specific performance shifts.
5. **Structural fidelity layer**
   - HTML / structure preservation; format compliance.
6. **Error topology layer**
   - MQM error types by model; severity distribution.
7. **Interaction & exploration layer**
   - Interactive drill-down: model ↔ block ↔ error ↔ term gate.

## Planned outputs (static + interactive)
- **Static plots:** updated `01–09` series from normalized tables.
- **Interactive wall:** aggregated comparisons, filters, and per-dimension slices.
- **Diagnostics views:** divergence cases, judge conflicts, and consensus maps.

## Data normalization requirements
- unify `S_final`, `S_mqm`, chrF/COMET columns
- ensure judge slots are exactly 5
- ensure all block IDs, model names, and dataset keys are consistent

---

# Recommended Re-entry Workflow
1. Verify Stage 2 output integrity (exactly 5 judge slots).
2. Rebuild Stage 2 audit + scoring if necessary.
3. Rebuild Stage 3 analysis outputs using normalized fields.
4. Regenerate `01–09` plots.
5. Update `Interactive_Compare_Wall.html` based on the new normalized tables.

---

# Stage 3 — Audit Report (Post judge_id Dedup Fix)
**Scope:** Stage 2 audit/scoring re-run only (no inference, no judge re-generation). Stage 3 primitives + plots + wall updated accordingly.

## Fix Applied
- `Local_PoLL_MQM/src/local_poll_mqm/mqm.py`: de-duplicate `judge_ids` per cluster and re-derive `votes` from unique judge IDs to prevent votes > 5.

## Audit Results (dim_data integrity)
- `dim_global_capability.csv`: **45 rows** (15 models × 3 blocks), model_id ↔ model_folder mapping consistent with leaderboard.
- `dim_error_topology.csv`: **248 rows**, **exact match** to accepted_errors aggregation by (model, block, category, severity).
- `dim_slang_matrix.csv`: **8,310 rows**, **exact match** to term_gate expected_hits + missing_expected; no duplicates.
- `dim_divergence_cases.json`: **4,605 rows**, **exact match** to accepted_errors votes=3 (marginal_pass) + rejected_errors votes=2 (marginal_fail).
- `dim_judge_consensus.json`: vote_distribution now strictly within **1–5** (no out-of-range votes).

## Ranking Shift Check (vs `output/archive/elite_five_integrated_backup`)
- **S_final mean delta:** +2.01 (min +0.33, max +3.87)
- **S_mqm mean delta:** +2.39 (min +0.83, max +5.22)
- **Top rank gains:** `google/translategemma-12b-it` (+2), `CohereLabs/tiny-aya-global` (+2), `google/gemma-3-12b-it` (+1)
- **Top rank drops:** `Qwen/Qwen2.5-7B-Instruct` (-2), `Qwen/Qwen3.5-9B` (-1), `google/translategemma-4b-it` (-1), `google/gemma-3-4b-it` (-1)

## Interactive Wall V2 (V1-style rebuilt)
- `Local_PoLL_MQM/scripts/build_interactive_wall.py` now rebuilds V2 with V1-style card wall (search + per-test compare table) as the **default tab**.
- Data source for compare wall is `output/elite_five_integrated/audited_reports` (full fidelity), while charts still use `dim_data`.

# Known Issues
- Stage 2/3 scripts still assume some missing fields from legacy runs.
- Plot set `01–05` needs regeneration after Stage 3 normalization.
- `Interactive_Compare_Wall.html` must be updated to support interactive drill-down and new dimensions.
- `dim_judge_consensus.json` includes a heuristic `reliability_rate` (accepted/proposed) that is **not a scientifically grounded judge reliability metric** and should never be used for ranking; treat it as diagnostic-only.