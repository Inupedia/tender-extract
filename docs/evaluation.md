# Extraction quality evaluation

`tender-extract eval` runs the real extraction pipeline against a JSONL Gold Dataset and reports field-level and aggregate quality metrics.

## Offline baseline

```bash
uv run tender-extract eval eval/gold.jsonl \
  --fail-under 1.0 \
  --report artifacts/baseline-eval.json
```

The command reports:

- per-field Precision / Recall / F1
- micro F1 and macro F1
- exact-case accuracy
- expected vs predicted values for failed fields
- LLM provider/model and call count

`--fail-under` exits with code `2` when micro F1 is lower than the required threshold, so the same dataset can be used as a CI regression gate.

## SiliconFlow evaluation

Use an environment variable or GitHub Actions Secret. Do not commit an API key or place it directly in a reusable command/script.

```bash
export SILICONFLOW_API_KEY="..."
uv run tender-extract eval eval/gold-siliconflow.jsonl \
  --llm siliconflow \
  --model Qwen/Qwen2.5-7B-Instruct \
  --fail-under 1.0 \
  --report artifacts/siliconflow-eval.json
```

`eval/gold-siliconflow.jsonl` contains a field that is not covered by the built-in rule patterns, so a successful run verifies real LLM missing-field recovery rather than merely initialising an LLM client.

## Gold Dataset format

Each non-empty JSONL line is one case:

```json
{"id":"basic-001","document":"fixtures/basic.md","tags":["basic-info"],"expected":{"project_number":["TEST-2026-001"],"bidder":["四川测试有限公司"]}}
```

`document` is resolved relative to the JSONL file. `expected` may contain a string or `list[str]` for each annotated field. Only explicitly annotated fields are scored, allowing the dataset to grow incrementally without treating unlabelled fields as false positives.

## Recommended workflow

1. Add a representative document fixture and human-verified expected values.
2. Run the offline baseline before changing extraction logic.
3. Make the extraction change.
4. Re-run the same dataset and inspect failures, not only the aggregate score.
5. Add the newly fixed failure as a permanent regression case.
6. Use the SiliconFlow dataset for changes that depend on LLM routing or recovery.
