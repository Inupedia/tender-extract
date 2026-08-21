<div align="center">

# tender-extract

**Turn Chinese procurement and tender documents into structured data you can trace back to the source.**

Rules handle the fast path · LLMs review only uncertain fields · Every result keeps evidence

English · [中文](README.md)

`PDF` · `DOCX` · `Markdown` · `TXT` · `CLI` · `HTTP API` · `Docker` · `Python 3.12+`

</div>

<p align="center">
  <img src="./assets/acceptance-benchmark.svg" width="100%" alt="tender-extract real-document acceptance: 13 real PDFs, 911 pages, 26.31 seconds, 34.62 pages per second, 14 of 14 semantic checks passed, 0 failures">
</p>

## What does it actually produce?

The committed [`examples/example.pdf`](examples/example.pdf) is a 10-page government procurement document. The current pipeline extracts results such as:

```text
project_name    合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目
project_number  2024BFFFZ01583
tenderer        合肥市公安局瑶海分局
bid_amount      437.677万元
bid_date        2024年7月1日17时30分
contact_info    055166223642
```

A field is more than a `key: value`. It can retain source evidence:

```json
{
  "project_number": {
    "primary_value": "2024BFFFZ01583",
    "confidence": 0.99,
    "values": [
      {
        "location": {
          "document_id": "example.pdf",
          "page": 1,
          "source_text": "项目编号：2024BFFFZ01583"
        }
      }
    ]
  }
}
```

When PDF text can be located reliably, the evidence can also include page coordinates, enabling flows such as:

```text
field → document → page → source text → PDF bbox / highlight
```

## Fastest start: Docker

Use the latest published image without setting up Python first:

```bash
IMAGE=ghcr.io/inupedia/tender-extract-server:latest

docker pull "$IMAGE"
docker run --rm \
  -p 8000:8000 \
  -v tender-extract-cache:/data/cache \
  "$IMAGE"
```

Upload a PDF:

```bash
curl -s \
  -F "file=@example.pdf" \
  "http://localhost:8000/v1/extract?llm_provider=none"
```

Open `http://localhost:8000/docs` for the generated OpenAPI documentation.

> `latest` is convenient for evaluation and following the newest release. For reproducible production deployments, pin `ghcr.io/inupedia/tender-extract-server:<version>`. Git tag `vX.Y.Z` publishes Docker tag `:X.Y.Z` and updates `:latest`.

[Browse GHCR image versions](https://github.com/Inupedia/tender-extract/pkgs/container/tender-extract-server)

## Why not send the entire PDF to an LLM?

```text
PDF / DOCX / Markdown / TXT
            │
            ▼
       parse + chunk
            │
            ▼
     rules / structure
        │           │
   high confidence  low confidence / conflict / missing
        │           │
        │           ▼
        │        LLM review
        │           │
        └─────┬─────┘
              ▼
        merge + evidence
              │
              ▼
        structured JSON
```

Deterministic fields stay on the local fast path. Only uncertain fields are routed to a model, preserving speed and cost control while keeping semantic recovery available where it helps.

## Core capabilities

| Capability | What it does |
|---|---|
| **Hybrid extraction** | Combines rules, dictionaries, NER and optional LLM review |
| **Evidence tracing** | Keeps file, page and source text; PDF evidence can also include bbox coordinates |
| **Project-level packages** | Handles tender, amendments, clarifications and multiple bidder files while preserving revisions and bidder boundaries |
| **Human review** | Accept, correct or reject uncertain output and feed reviewed data back into Gold datasets |
| **Quality evaluation** | Precision / Recall / F1 with CI quality gates |
| **Multiple entry points** | Python CLI, HTTP API and Docker / GHCR |

### Current support

| Area | Support |
|---|---|
| Formats | PDF, DOCX, Markdown, TXT |
| Scanned files | Optional OCR; the lean server image does not bundle PaddleOCR |
| LLM providers | SiliconFlow, OpenAI, DeepSeek, Qwen/DashScope, Claude, Gemini, Ollama, OpenAI-compatible |
| Privacy | Sensitive personal information is excluded by default; HTTP API can require an API key |

## HTTP server

```text
GET  /healthz      health check
GET  /v1/info      version and capabilities
POST /v1/extract   upload a document and return structured output
```

<details>
<summary><strong>Protect the server with an API key</strong></summary>

```bash
IMAGE=ghcr.io/inupedia/tender-extract-server:latest

docker run --rm -p 8000:8000 \
  -e TENDER_SERVER_API_KEY=your-secret \
  "$IMAGE"
```

Send the key as:

```text
X-API-Key: your-secret
```

</details>

<details>
<summary><strong>Enable SiliconFlow inside the container</strong></summary>

```bash
IMAGE=ghcr.io/inupedia/tender-extract-server:latest

docker run --rm -p 8000:8000 \
  -e SILICONFLOW_API_KEY=your-key \
  -e TENDER_SERVER_LLM_PROVIDER=siliconflow \
  -e TENDER_SERVER_LLM_MODEL=Qwen/Qwen3-8B \
  -v tender-extract-cache:/data/cache \
  "$IMAGE"
```

</details>

## Local CLI

Requires Python **3.12+**.

```bash
git clone https://github.com/Inupedia/tender-extract.git
cd tender-extract

uv sync --extra pdf
uv run tender-extract extract examples/example.pdf --out out
```

Batch extraction:

```bash
uv run tender-extract extract ./documents --pattern "*.pdf" --out out
```

## LLMs are an optional enhancement, not a prerequisite

SiliconFlow `Qwen/Qwen3-8B` has been exercised against the real API: the cold acceptance run completed **4/4 network calls successfully**; the second pass reused **4 cache entries with 0 new network calls**. The current live Gold acceptance case reports **Micro F1 / Macro F1 = 1.000**.

> That F1 belongs to the current acceptance case. It is not a claim of universal accuracy across all tender documents.

```bash
export SILICONFLOW_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm siliconflow \
  --model Qwen/Qwen3-8B \
  --out out
```

List available providers:

```bash
uv run tender-extract providers
```

## One project can contain many files

Real tender projects rarely consist of a single PDF:

```text
project/
├── tender.pdf
├── amendment-01.pdf
├── clarification-01.pdf
├── bidder-a.pdf
└── bidder-b.pdf
```

`tender-package` processes them as one project while preserving revision relationships and bidder isolation:

```bash
uv run tender-package validate package.yaml
uv run tender-package extract package.yaml --out out/package.json
```

An amendment can supersede an older value without deleting historical evidence, and bidder A cannot overwrite bidder B.

See [`docs/tender-package.md`](docs/tender-package.md).

## Human review and quality loop

```bash
uv run tender-review run examples/example.pdf --queue .review/queue.jsonl
uv run tender-review export --queue .review/queue.jsonl --out eval/gold-reviewed.jsonl
uv run tender-extract eval eval/gold-reviewed.jsonl
```

CI can enforce an F1 floor directly:

```bash
uv run tender-extract eval eval/gold.jsonl --fail-under 0.95
```

## Reproduce the real-document acceptance run

The repository commits a real procurement/tender acceptance corpus covering Anhui, Beijing, Shaanxi, Henan and Shanghai: **13 documents / 911 pages**. The current GitHub Actions benchmark with LLM disabled records:

| Metric | Result |
|---|---:|
| Documents | **13** |
| Pages | **911** |
| Wall time | **26.31 s** |
| Throughput | **34.62 pages/s** |
| Semantic checks | **14 / 14** |
| Failures | **0** |

> Throughput is measured in GitHub Actions. Hardware, OCR, storage and PDF layout can change real-world performance.

Source URLs, page counts and SHA256 values are recorded in [`examples/public-corpus.lock.json`](examples/public-corpus.lock.json). Reproduce the run with:

```bash
uv run python scripts/acceptance_corpus.py \
  --examples examples \
  --min-pdfs 13 \
  --report artifacts/real-pdf-acceptance.json
```

## Documentation

- [Evidence tracing](docs/evidence.md)
- [Project-level multi-file handling](docs/tender-package.md)
- [Human review](docs/human-review.md)
- [Evaluation](docs/evaluation.md)

## License

MIT
