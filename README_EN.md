# tender-extract · Structured extraction for Chinese tender documents

English | [中文](README.md)

> **Do not send the whole PDF to an LLM.** `tender-extract` handles deterministic fields with a fast, reproducible rule/router layer and only routes low-confidence, conflicting, or missing fields to an LLM. Results retain page-aware evidence and PDF bounding boxes for audit and review.

![Real PDF acceptance benchmark](assets/acceptance-benchmark.svg)

## 🚀 Why use it?

- ⚡ **Fast deterministic path** — 13 committed public PDFs, 911 pages, measured at **26.31 s / 34.62 pages per second** in GitHub Actions.
- 🧠 **LLM only when needed** — live-tested against SiliconFlow `Qwen/Qwen3-8B`.
- 🔎 **Auditable evidence** — `document_id + page + section + line + source_text + bbox` when the source format supports it.
- 📦 **Multi-file tender packages** — tender, amendment, clarification and bidder submissions with revisions and bidder isolation.
- 👀 **Human review loop** — accept/correct/reject uncertain fields and export decisions back into a Gold Dataset.
- 📊 **Quality gates** — built-in Precision / Recall / F1 evaluation for CI.

## 📊 Real PDF acceptance

Measured on GitHub-hosted Ubuntu 24.04 with Python 3.12 and LLM disabled:

| Metric | Result |
|---|---:|
| PDFs | **13** |
| Pages | **911** |
| Corpus size | **9.26 MB** |
| Extracted fields | **72** |
| Wall time | **26.31 s** |
| Throughput | **34.62 pages/s** |
| Average per PDF | **2.02 s** |
| Semantic checks | **14 / 14** |
| Failures | **0** |
| LLM calls | **0** |

This is a measured CI run, not an SLA. OCR, hardware, storage and document layout can change throughput. The **34.62 pages/s number does not include OCR or network LLM time**.

### `examples/example.pdf`

The committed 10-page procurement document completes in roughly **0.35 s** on the deterministic path with six core fields and average confidence around **0.94**:

```text
project_name    合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目  0.99  page 1
project_number  2024BFFFZ01583                                             0.95  page 1
tenderer        合肥市公安局瑶海分局                                        0.95  page 1
bid_amount      437.677万元                                                0.90  page 3
bid_date        2024年7月1日17时30分                                       0.90  page 5
contact_info    055166223642                                               0.95  page 7
```

The real-world corpus uncovered and fixed issues that synthetic tests did not catch: wrapped project names, project-name over-capture, procurement-agent false positives, URL noise, unrelated historical dates being selected as deadlines, and unlabelled large amounts being promoted to bid amounts.

## 🧠 Live SiliconFlow validation

```bash
export SILICONFLOW_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm siliconflow \
  --model Qwen/Qwen3-8B \
  --out out
```

Measured live acceptance:

| Metric | Result |
|---|---:|
| Model | `Qwen/Qwen3-8B` |
| Cold logical LLM calls | **4** |
| Cold network calls | **4 / 4 succeeded** |
| Cold wall time | **423.31 s** |
| Warm cache hits | **4** |
| New network calls on warm run | **0** |
| Warm wall time | **0.27 s** |
| Gold Micro F1 | **1.000** |
| Gold Macro F1 | **1.000** |
| Exact case accuracy | **1.000** |

Cold LLM latency depends on the external provider, queueing, model inference and network conditions. That is exactly why the pipeline is designed as **rules first → LLM only for uncertainty → persistent cache**. High-confidence rule fields remain protected from unnecessary model overrides.

For fast batch extraction without model calls:

```bash
uv run tender-extract extract ./documents --llm none --out out
```

## 🔎 Evidence-aware JSON

A simplified real result looks like this:

```json
{
  "metadata": {
    "filename": "example.pdf",
    "total_pages": 10,
    "processing_time": 0.35
  },
  "fields": {
    "project_name": {
      "primary_value": "合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目",
      "confidence": 0.99,
      "values": [
        {
          "source": "regex_enhanced",
          "location": {
            "document_id": "example.pdf",
            "page": 1,
            "source_text": "合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目"
          }
        }
      ]
    }
  }
}
```

For PDF text that can be located reliably, evidence can also carry a real `bbox`. Markdown/TXT do not receive fabricated physical pages or coordinates; they keep character spans, lines and section paths instead.

See [`docs/evidence.md`](docs/evidence.md).

## 🛠️ Quick start

Requires Python **3.12+**.

```bash
git clone https://github.com/Inupedia/tender-extract.git
cd tender-extract

uv sync --extra pdf
uv run tender-extract extract examples/example.pdf --out out
```

Batch mode:

```bash
uv run tender-extract extract examples/ --pattern "*.pdf" --llm none --out out
```

List configured LLM providers:

```bash
uv run tender-extract providers
```

The provider registry includes SiliconFlow, OpenAI, Azure OpenAI, Anthropic, Gemini, DeepSeek, DashScope/Qwen, Kimi, Zhipu, Volcengine, Ollama, OpenRouter, Groq, Together, Mistral, xAI and generic OpenAI-compatible endpoints.

## 📦 Tender Package

Real projects contain multiple documents, not one PDF:

```text
project/
├── tender.pdf
├── amendment-01.pdf
├── clarification-01.pdf
├── bidder-a.pdf
└── bidder-b.pdf
```

`tender-package` models document roles, revisions and supersedes relationships. Procurement-side documents create an effective view, while bidder submissions remain isolated by bidder.

```bash
uv run tender-package validate package.yaml
uv run tender-package inspect package.yaml
uv run tender-package extract package.yaml --out out/package.json
```

See [`docs/tender-package.md`](docs/tender-package.md).

## 👀 Human review → Gold Dataset

```bash
uv run tender-review run examples/example.pdf --queue .review/queue.jsonl
uv run tender-review list --queue .review/queue.jsonl
uv run tender-review resolve REVIEW_ID --action correct --value "correct value"
uv run tender-review export --queue .review/queue.jsonl --out eval/gold-reviewed.jsonl
```

The feedback loop is:

**extract → review → human decision → Gold Dataset → regression / F1 gate**

See [`docs/human-review.md`](docs/human-review.md).

## 📈 Evaluation

```bash
uv run tender-extract eval eval/gold.jsonl
uv run tender-extract eval eval/gold.jsonl --fail-under 0.95
```

Live SiliconFlow evaluation:

```bash
export SILICONFLOW_API_KEY=your-key
uv run tender-extract eval eval/gold-siliconflow.jsonl \
  --llm siliconflow \
  --model Qwen/Qwen3-8B \
  --fail-under 1.0
```

Reports include per-field Precision / Recall / F1, Micro F1, Macro F1, exact-case accuracy, failures and LLM call counts.

See [`docs/evaluation.md`](docs/evaluation.md).

## 📚 Committed acceptance corpus

`examples/` contains the historical 10-page sample plus 12 public tender/procurement PDFs from Beijing, Shaanxi, Henan and Shanghai. The 13 files total **911 pages**.

Official source URLs, original sizes, page counts and SHA256 values for the public corpus are recorded in [`examples/public-corpus.lock.json`](examples/public-corpus.lock.json). Candidate source metadata is stored in [`examples/public-sources.json`](examples/public-sources.json).

Normal CI does **not** depend on those government sites. It runs entirely against the committed corpus:

```bash
uv run python scripts/acceptance_corpus.py \
  --examples examples \
  --min-pdfs 13 \
  --report artifacts/real-pdf-acceptance.json
```

## ⚙️ Pipeline

```text
PDF / DOCX / MD / TXT
        │
        ▼
  Document Parser ───── OCR fallback
        │
        ▼
  Chunk + Module Router
        │
        ▼
 Regex / Dictionary / NER
        │
        ├── high confidence ───────────────┐
        │                                  │
        └── low/conflict/missing → LLM ───┤
                                           ▼
                                  Conflict Resolution
                                           │
                                           ▼
                                  Evidence Locator
                                  page / bbox / section
                                           │
                                           ▼
                                      Pydantic JSON
```

The principle is simple: **do not spend model latency and cost on deterministic facts, and do not pretend a rule is certain when it is not.**

## 🧪 CI

The repository validates:

```text
unit
integration
E2E
legacy regression
package build
combined coverage
real PDF acceptance (13 PDFs / 911 pages)
SiliconFlow live acceptance
```

Real PDF acceptance is fully offline. SiliconFlow live validation uses a GitHub Actions secret and does not commit the API key to source or artifacts.

## ⚠️ Boundaries

- The 34.62 pages/s benchmark excludes OCR and network LLM calls.
- OCR on scanned documents is substantially slower than text-layer PDF parsing.
- LLM latency depends on provider/model/network; persistent cache is strongly recommended.
- PDF can provide physical page and bbox provenance; non-paginated formats do not receive fake coordinates.
- Chinese tender/procurement templates vary by industry and region, so Gold Datasets should keep growing with real reviewed cases.

## 📄 License

MIT License
