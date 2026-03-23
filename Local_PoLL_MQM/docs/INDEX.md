# Documentation Index

## Purpose

This document provides a concise index of the project documentation, stage boundaries, and selected visual references. It is intended to keep the repository homepage and stage notes lightweight while preserving a clear navigation path to the detailed materials.

## Stage Notes

### Stage 1: Colab-based inference and baseline audit
Stage 1 is the primary Colab execution stage. It runs model inference on the benchmark dataset and produces baseline audit outputs. The local repository keeps the core code paths and documentation needed to reproduce or inspect the run logic.

Primary references:
- `IPYNB-PY/arena_core/`
- `PROJECT_MEMORY.md`
- `DATA_STRUCTURE.md`

### Stage 2: Local PoLL + MQM consensus audit
Stage 2 is the local multi-judge auditing stage. It consumes exported inference artifacts and produces consensus MQM judgments, block-level reports, and the integrated leaderboard.

Primary references:
- `Local_PoLL_MQM/README.md`
- `Local_PoLL_MQM/docs/BLUEPRINT.md`
- `Local_PoLL_MQM/docs/ALGORITHM_DRAFT.md`
- `Local_PoLL_MQM/scripts/`
- `Local_PoLL_MQM/output/elite_five_integrated/`

### Stage 3: Analysis and visualization
Stage 3 is the analysis layer built on top of Stage 2 outputs. It generates normalized tables, plots, and the interactive comparison wall.

Primary references:
- `Local_PoLL_MQM/analysis_infra/FINAL_POST_DRAFT.md`
- `Local_PoLL_MQM/analysis_infra/dim_data/`
- `Local_PoLL_MQM/analysis_infra/raw_stats/`
- `Local_PoLL_MQM/analysis_infra/Interactive_Compare_Wall_V2.html`

## Canonical Plot Set

The repository maintains a canonical 10-figure plot set for Stage 3 analysis. These are the recommended references when a visual summary is needed in documentation:

- `Local_PoLL_MQM/analysis_infra/plots_canonical/01_Global_S_Final_Ranking.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/02_Global_S_MQM_Ranking.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/03_Performance_Across_Blocks.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/04_Error_Consensus_Distribution.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/05_Metrics_Divergence.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/06_Slang_Hit_Rate.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/07_Slang_Disambiguation_S_Final_Heatmap.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/08a_Radar_Top5.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/08b_Radar_All.png`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/09_Error_Severity_Distribution.png`

Recommended usage:
- Prefer tables and textual summaries for the repository homepage.
- Use 1 to 3 representative figures only when a visual explanation is necessary.
- Keep the full plot set in the Stage 3 analysis area rather than duplicating it across documentation.

## Recommended Representative Figures

For README-level or summary documentation, the following figures are the most representative:

1. `01_Global_S_Final_Ranking.png`
2. `03_Performance_Across_Blocks.png`
3. `05_Metrics_Divergence.png`

These three figures provide:
- a global ranking summary,
- a block-level comparison,
- and the metric divergence view.

If only one figure is needed, prefer `01_Global_S_Final_Ranking.png`.

## Core Documentation Files

- `README.md`
  - repository homepage and execution guide
- `DATA_STRUCTURE.md`
  - logical data hierarchy and storage policy
- `PROJECT_MEMORY.md`
  - stage-by-stage project memory map
- `Local_PoLL_MQM/README.md`
  - local Stage 2 execution guide
- `Local_PoLL_MQM/docs/BLUEPRINT.md`
  - pipeline boundaries and runtime design
- `Local_PoLL_MQM/docs/ALGORITHM_DRAFT.md`
  - scoring logic and algorithm notes
- `Local_PoLL_MQM/analysis_infra/FINAL_POST_DRAFT.md`
  - report-style narrative draft
- `Local_PoLL_MQM/docs/WATERMARK_EXECUTION.md`
  - invisible watermark execution and verification runbook

## External Data Policy

Large data artifacts are stored outside the repository and are referenced through public links or mirrored metadata. The repository is intended to track:
- core source code,
- configuration,
- documentation,
- hierarchy definitions,
- and lightweight manifest or summary files.

Generated artifacts that should generally remain external or ignored include:
- large dataset archives,
- analysis exports,
- canonical plots,
- watermarked plots,
- and interactive HTML outputs.

## Notes for Maintenance

- Update this document whenever the stage layout changes.
- If a new canonical figure replaces an older one, revise the list above rather than adding a second redundant reference.
- If a new docs file is added, place it in the appropriate section and keep the index concise.