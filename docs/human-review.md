# Human review feedback loop

`tender-review` turns uncertain extraction results into a durable human-review queue and then exports reviewed decisions back into the Gold Dataset format used by `tender-extract eval`.

## What enters the queue

A field becomes reviewable when at least one of these conditions is true:

- confidence is below the configured extraction threshold;
- the field still has unresolved extraction conflicts;
- the selected primary candidate was recovered by an LLM.

Each review item stores the source document, field name, candidate values, confidence, reason codes, conflicts and a compact evidence snapshot. Item IDs are deterministic, so running the same extraction repeatedly does not create duplicate pending work. A resolved item is not overwritten by a later identical extraction.

## Run extraction and create review items

```bash
tender-review run ./examples/example.md \
  --queue .review/queue.jsonl \
  --config ./config/example.yaml
```

For SiliconFlow, keep the key in the environment rather than the command history:

```bash
export SILICONFLOW_API_KEY=...
tender-review run ./examples/example.md \
  --llm siliconflow \
  --model Qwen/Qwen2.5-7B-Instruct \
  --queue .review/queue.jsonl
```

The normal extraction result can optionally be retained with `--result out/result.json`.

## Collect from an existing extraction result

```bash
tender-review collect ./out/example.md.json \
  --document ./examples/example.md \
  --queue .review/queue.jsonl \
  --threshold 0.7
```

## Inspect and resolve work

```bash
tender-review list --queue .review/queue.jsonl --status pending
```

Accept the current primary value:

```bash
tender-review resolve REVIEW_ID \
  --queue .review/queue.jsonl \
  --action accept \
  --reviewer robin
```

Correct a value:

```bash
tender-review resolve REVIEW_ID \
  --queue .review/queue.jsonl \
  --action correct \
  --value "修正后的字段值" \
  --reviewer robin \
  --note "人工核对原文后修正"
```

Reject a false-positive extraction:

```bash
tender-review resolve REVIEW_ID \
  --queue .review/queue.jsonl \
  --action reject
```

A rejection is intentionally exported as an empty expected value for that field. This allows the evaluator to treat a repeated false positive as a regression rather than silently dropping the feedback.

## Export reviewed decisions into evaluation data

```bash
tender-review export \
  --queue .review/queue.jsonl \
  --out eval/gold-reviewed.jsonl

tender-extract eval eval/gold-reviewed.jsonl
```

The exported dataset is partially labelled: it only contains fields that humans explicitly resolved. This matches the evaluator's existing semantics and lets the feedback dataset grow incrementally without requiring every document to be exhaustively annotated.

## Storage model

The queue is a local JSONL file with atomic rewrites. It is intentionally storage-agnostic: the review domain models and deterministic IDs can later be placed behind a database/API without changing the review semantics. PR2 does not add a web UI or multi-user locking; those belong in the later service/product layer.
