# Data Structure and Asset Hierarchy

## Artifact Policy
- Large datasets, generated plots, and run outputs are treated as external artifacts and should not be committed by default.
- The repository should keep only source code, configuration, documentation, and hierarchy definitions.
- If a generated artifact must be shipped for publication, it should be explicitly whitelisted rather than added broadly.

## Purpose
This document defines the repository data hierarchy, core assets, and storage policy. Large datasets and run artifacts are stored externally and referenced by public links. This repository contains core logic, scripts, and documentation.

## Storage Policy
- Large data files are not tracked in Git.
- Public download links are stored in this document once available.
- Only core code, configuration, and documentation are tracked.
- Watermarked image variants, derived charts, and analysis exports are treated as generated artifacts unless explicitly required.

## External Data Links
- Stage 0 raw data (Metadatas): TO_BE_ADDED
- Stage 1 run artifacts (Benchmarks): TO_BE_ADDED
- Stage 2 integrated outputs (Local_PoLL_MQM/output): TO_BE_ADDED
- Stage 3 analysis artifacts (analysis_infra outputs): TO_BE_ADDED
- Canonical plot bundle: TO_BE_ADDED
- Watermarked plot bundle: TO_BE_ADDED

## Repository Hierarchy (Logical View)

### Root
- `README.md`
- `DATA_STRUCTURE.md`
- `PROJECT_MEMORY.md`
- `Metadatas-*/`
- `Benchmarks-*/`
- `IPYNB-PY/`
- `Local_PoLL_MQM/`

### Stage 0: Metadatas
- `Metadatas-*/Metadatas/`
  - `l_station_posts.jsonl`
  - `l_station_posts_pure.jsonl`
  - `Benchmark_V1_Dataset.json`
  - `Benchmark_Reference_Translations.json`
  - `Benchmark_Slang_Context.json`
  - `Benchmark_Slang_Golden_V1_Final.json`
  - `L_Station_Glossary.json`
  - `Slang_Filtered_Benchmark_Raw.json`
  - `extracted_keywords.json`

### Stage 1: Inference and Baseline Audit
- `IPYNB-PY/arena_core/`
  - `config.py`
  - `filesystem.py`
  - `schemas.py`
  - `inference_runner.py`
  - `audit_evaluator.py`
  - `aggregator.py`
- `Benchmarks-*/Benchmarks/`
  - `01_Datasets/`
  - `02_Experiment_Runs/<model>/`
    - `raw_inference/`
    - `audited_reports/`
  - `03_Leaderboard/Global_Metrics_Summary.json`
  - `run_manifest.json`

### Stage 2: PoLL MQM Consensus
- `Local_PoLL_MQM/`
  - `src/local_poll_mqm/`
  - `configs/`
  - `scripts/`
  - `output/`
    - `elite_five_integrated/`
      - `checkpoints/`
      - `audited_reports/`
      - `leaderboard/Global_PoLL_MQM_Summary.json`
      - `scheduler_diagnostic.log`

### Stage 3: Analysis and Visualization
- `Local_PoLL_MQM/analysis_infra/`
  - `dim_data/`
    - `dim_global_capability.csv`
    - `dim_error_topology.csv`
    - `dim_slang_matrix.csv`
    - `dim_divergence_cases.json`
    - `dim_judge_consensus.json`
  - `raw_stats/`
  - `plots_canonical/`
  - `plots_canonical copy/`
  - `plots_canonical_watermarked/`
  - `plots_canonical_watermarked_svd/`
  - `Interactive_Compare_Wall_V2.html`

## Notes
- The hierarchy above is a logical view of core assets and is updated after each audit pass.
- Any deviations or new assets should be recorded in this document before publication.
- Test directories, scratch outputs, and helper scripts should be added to the repository ignore policy rather than tracked directly.