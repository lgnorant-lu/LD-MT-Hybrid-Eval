# Local PoLL + MQM Evaluator

This folder is a fully local implementation for the next-stage evaluation pipeline.

- Does not reuse the Colab runner/evaluator code path.
- Uses exported `Benchmarks` + `Metadatas` as data source.
- Supports mixed OpenAI-compatible judge endpoints.
- Supports smoke mode (`2/3 judges + repeat 2`) and full mode.

## Quick Start

1. Copy `configs/local_poll_mqm.example.json` to `configs/local_poll_mqm.json`.
2. Fill Gemini endpoint and API key env variables.
3. Run smoke test (single judge, small sample):

```bash
python run_local_poll_mqm.py --config configs/local_poll_mqm.json --smoke --smoke-judges 1 --smoke-repeat 1 --smoke-max-items 2 --only-judge-slot gemini_3_flash_01
```

4. Single triple test (one model + one block + one test_id):

```bash
python run_local_poll_mqm.py --config configs/local_poll_mqm.json --single-model CohereLabs--tiny-aya-global --single-block Jargon_Tech --single-test-id slang_test_001 --only-judge-slot gemini_3_flash_01 --min-valid-judges 1 --vote-threshold 1
```

5. Run full panel:

```bash
python run_local_poll_mqm.py --config configs/local_poll_mqm.json
```

## Request Design Notes

- The evaluator sends row-level triples: `source_text`, `hypothesis_text`, `reference_text(optional)`.
- `poll.request_batch_size` controls how many triples are packed into one LLM request.
- `runtime.block_sample_limits` controls per-block sampling size (for example Baseline 80, Jargon 34, Slang 207).
- `runtime.require_reference_blocks` ensures selected blocks only keep rows with non-empty reference text.

## Outputs

- Per-model, per-block audit reports:
  - `output/audited_reports/<model_folder>/<block>_poll_mqm_audit.json`
- Global leaderboard:
  - `output/leaderboard/Global_PoLL_MQM_Summary.json`
- Run manifest:
  - `output/run_manifest.json`

## Core Docs

- Blueprint: `docs/BLUEPRINT.md`
- Algorithm draft: `docs/ALGORITHM_DRAFT.md`
