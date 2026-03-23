# Blueprint: Local PoLL + MQM Pipeline

## 1. Scope and Boundaries

This pipeline is a local rebuild for PoLL + MQM scoring. It is intentionally decoupled from previous Colab runtime code.

Input artifacts:
- Exported `Benchmarks/01_Datasets`
- Exported `Benchmarks/02_Experiment_Runs`
- Optional `Metadatas` for diagnostics

Output artifacts:
- Per-block audited report with row-level PoLL/MQM details
- Global leaderboard
- Run manifest and diagnostics

## 2. Runtime Stages

1. Load Config
- Read local JSON config.
- Resolve benchmark paths and model list.
- Build Gemini judge slots (7 slots default).

2. Load Data
- Build dataset map by block (`Baseline_Standard`, `Jargon_Tech`, `Slang_Ambiguous`).
- For each model/block, load raw inference file and index dataset by `test_id`.

3. PoLL Judging
- For each row, send source/hypothesis/reference context to each judge slot.
- Parse strict JSON outputs (`errors[]` with span/category/severity).
- Retry on transient API/JSON failures.

4. Arbitration
- Cluster similar errors across judges by span overlap + category.
- Keep errors with votes >= threshold (default 4/7).
- Resolve severity conflicts with deterministic rule.

5. MQM + Objective + Term Gate
- Compute `S_mqm` by accepted MQM penalties and length normalization.
- Compute objective fallback or real metrics (chrF++, COMET).
- Compute terminology gate penalty from expected/forbidden terms.
- Produce final score using the nonlinear formula.

6. Aggregation
- Summarize by block and by model.
- Rank models by overall average `S_final`.
- Persist reports and manifest.

## 3. Project Structure

```text
Local_PoLL_MQM/
  configs/
    local_poll_mqm.example.json
  docs/
    BLUEPRINT.md
    ALGORITHM_DRAFT.md
  src/local_poll_mqm/
    __init__.py
    cli.py
    config.py
    io_utils.py
    judge_client.py
    metrics.py
    mqm.py
    pipeline.py
    term_gate.py
    types.py
  run_local_poll_mqm.py
```

## 4. Design Choices Frozen for v1

- Local-only execution, no Colab dependency.
- Gemini-only PoLL panel slots.
- Smoke mode defaults: 3 judges, repeat 2, limited samples.
- Full mode defaults: 7 judges, vote threshold 4/7.
- Keep original nonlinear final-score formula.

## 5. Reliability Controls

- Retry and timeout per judge call.
- JSON extraction and schema validation.
- Minimum valid judge responses per row.
- Deterministic arbitration output.
- Run manifest with settings snapshot.

## 6. Future Extensions

- Judge weighting and calibration.
- Better span clustering with embedding similarity.
- Language-specific tokenization for MQM normalization.
- Confidence interval and rank robustness report.
