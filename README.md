# LinuxDo MT Evaluation Stack

## Overview

This repository contains the core logic and documentation for a multi-stage machine translation evaluation system focused on LinuxDo community terminology and slang disambiguation. The pipeline is structured into data preparation, model inference, multi-judge MQM auditing, and analysis and visualization.

## Scope and Goals

- Build a benchmark that emphasizes slang and community-specific terminology.
- Apply multi-judge MQM auditing to reduce single-evaluator bias.
- Provide standardized analysis outputs and visual artifacts for reporting.
- Keep large datasets external and track only the core logic, documentation, and hierarchy definitions in Git.

## Repository Layout

- `IPYNB-PY/arena_core/`: Stage 1 inference engine and auditing utilities.
- `Local_PoLL_MQM/`: Stage 2 MQM pipeline, evaluation reports, analysis infra, and Stage 3 analysis outputs.
- `Metadatas-*/` and `Benchmarks-*/`: archived datasets and historical run artifacts.
- `DATA_STRUCTURE.md`: logical asset hierarchy and storage policy.
- `PROJECT_MEMORY.md`: project memory map and stage-by-stage file roles.

## Stage Boundaries and Execution Guide

### Stage 1: Colab-based inference and baseline audit

Stage 1 remains the primary Colab execution stage. Its purpose is to run model inference on the benchmark dataset and produce baseline audit outputs. The local repository keeps only the code paths and documentation required to reproduce or inspect the run logic.

Core responsibilities:
- Load the benchmark dataset snapshots.
- Run model inference in the Colab runtime.
- Produce raw inference outputs and baseline audit reports.
- Aggregate the baseline leaderboard.

Primary code location:
- `IPYNB-PY/arena_core/`

Primary outputs:
- `Benchmarks/01_Datasets/`
- `Benchmarks/02_Experiment_Runs/<model>/raw_inference/`
- `Benchmarks/02_Experiment_Runs/<model>/audited_reports/`
- `Benchmarks/03_Leaderboard/Global_Metrics_Summary.json`
- `Benchmarks/run_manifest.json`

Stage 1 data assets are not committed to the repository and should be stored externally.

### Stage 2: Local PoLL + MQM consensus audit

Stage 2 is the local multi-judge auditing stage. It consumes exported inference artifacts and produces consensus MQM judgments, block-level reports, and the integrated leaderboard.

Core responsibilities:
- Load exported Stage 1 artifacts.
- Build per-model and per-block judge caches.
- Perform PoLL judgment and MQM arbitration.
- Apply objective metric fallback and terminology gate logic.
- Aggregate the integrated leaderboard.
- Keep the Stage 2 output tree reproducible through `run_manifest`, `scheduler_diagnostic.log`, and the block-level audited reports.

Primary code location:
- `Local_PoLL_MQM/src/local_poll_mqm/`
- `Local_PoLL_MQM/scripts/`

Primary outputs:
- `Local_PoLL_MQM/output/elite_five_integrated/checkpoints/`
- `Local_PoLL_MQM/output/elite_five_integrated/audited_reports/`
- `Local_PoLL_MQM/output/elite_five_integrated/leaderboard/Global_PoLL_MQM_Summary.json`
- `Local_PoLL_MQM/output/elite_five_integrated/scheduler_diagnostic.log`
- `Local_PoLL_MQM/output/elite_five_integrated/profiling_report.json`

### Stage 3: Analysis and visualization

Stage 3 is the analysis layer built on top of Stage 2 outputs. It generates normalized tables, plots, and the interactive comparison wall.

Core responsibilities:
- Normalize audited reports into analysis tables.
- Generate canonical plots.
- Produce interactive comparison artifacts.
- Support drill-down and audit review.
- Keep the canonical plots and derived tables outside Git if they are regenerated locally.

Primary code location:
- `Local_PoLL_MQM/scripts/`
- `Local_PoLL_MQM/analysis_infra/`

Primary outputs:
- `Local_PoLL_MQM/analysis_infra/dim_data/`
- `Local_PoLL_MQM/analysis_infra/raw_stats/`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/`
- `Local_PoLL_MQM/analysis_infra/Interactive_Compare_Wall_V2.html`

## Chart Policy

The repository keeps the chart policy explicit:

- The canonical plot set is the authoritative visual summary for Stage 3.
- The current canonical set contains 10 figures:
  - `01_Global_S_Final_Ranking.png`
  - `02_Global_S_MQM_Ranking.png`
  - `03_Performance_Across_Blocks.png`
  - `04_Error_Consensus_Distribution.png`
  - `05_Metrics_Divergence.png`
  - `06_Slang_Hit_Rate.png`
  - `07_Slang_Disambiguation_S_Final_Heatmap.png`
  - `08a_Radar_Top5.png`
  - `08b_Radar_All.png`
  - `09_Error_Severity_Distribution.png`
- For the repository, charts may be referenced in documentation, but the preferred source of truth for the final report is the data table and the derived leaderboard unless a figure is specifically needed for illustration.
- Large binary image assets are treated as generated outputs and should remain external or ignored unless a release specifically requires bundling them.
- Watermarked variants are treated as derived assets and should not be committed unless explicitly requested.

## Data Policy

Large data artifacts and datasets are not tracked in this repository. They are stored externally, for example in Google Drive, and referenced via public links or mirrored metadata. This repository focuses on core logic, scripts, documentation, and reproducible structure definitions.

The following directories are treated as external or generated assets and should remain untracked unless explicitly required:
- `Metadatas-*/`
- `Benchmarks-*/`
- `Local_PoLL_MQM/output/`
- `Local_PoLL_MQM/analysis_infra/dim_data/`
- `Local_PoLL_MQM/analysis_infra/raw_stats/`
- `Local_PoLL_MQM/analysis_infra/plots_canonical/`
- `Local_PoLL_MQM/analysis_infra/plots_canonical_watermarked/`
- `Local_PoLL_MQM/analysis_infra/plots_canonical_watermarked_svd/`

## Reproducibility

- Use Stage 0 assets to rebuild benchmarks.
- Use `arena_core` for inference and baseline auditing.
- Use `Local_PoLL_MQM` for multi-judge MQM consensus and analysis.
- Use the asset hierarchy documented in `DATA_STRUCTURE.md` to restore the expected directory layout.
- Stage 1 logic runs in Colab; Stage 2 and Stage 3 are the local execution and analysis layers.

## Documentation Index

- `Local_PoLL_MQM/docs/INDEX.md`: centralized documentation navigator.
- `PROJECT_MEMORY.md`: project memory map and stage-by-stage file roles.
- `DATA_STRUCTURE.md`: logical hierarchy of datasets, outputs, and tracked code.
- `Local_PoLL_MQM/docs/BLUEPRINT.md`: Stage 2 pipeline design and execution boundaries.
- `Local_PoLL_MQM/docs/ALGORITHM_DRAFT.md`: algorithmic notes and score composition details.
- `Local_PoLL_MQM/analysis_infra/FINAL_POST_DRAFT.md`: final narrative draft and report text.

## Status

This repository is under active documentation and audit updates.

## License

Not specified. Add a license file if distribution terms are required.
