# tender-extract

English | [中文](README.md)

> A **hybrid extraction** pipeline for **thousand-page** level Chinese tender documents (converted to Markdown): First use **rules/dictionaries/NER** to extract deterministic fields, then route only **low-confidence/conflicting** small fragments to **LLM**, ensuring auditability while significantly reducing costs and improving efficiency.

## 🚀 Core Advantages

- **Cost Control**: Rule layer covers 60-90% of hard fields, dramatically reducing LLM API calls
- **High Performance**: 5 documents processed in only 2.31 seconds, averaging 0.46 seconds per document
- **Zero LLM Calls**: In your tests, the rule layer completely covered all fields, requiring no LLM calls
- **Auditability**: Each extraction result preserves original text evidence fragments for traceability and verification
- **Detailed Progress**: Real-time display of LLM processing progress and content for debugging and monitoring

## ✨ Features

- **Markdown Structure Parsing + Chapter-Priority Slicing**: Based on `markdown-it-py`, builds chapter tree by `# / ## / ###`, then performs recursive character splitting (with minimal overlap).
- **High-Throughput Rule Layer**: Regex + keyword line heuristics; extracts amounts/dates/deposits/contact info/addresses/postal codes in one pass.
- **Ultra-Fast Large Dictionary Matching**: Aho–Corasick (`pyahocorasick`) batch phrase scanning (e.g., "qualification requirements/qualification conditions/evaluation methods/consortium" and similar phrases).
- **Near-Duplicate & Template Recognition**: `RapidFuzz` string similarity + `datasketch` **MinHash LSH**, avoiding duplicate LLM queries.
- **On-Demand LLM**: Only when rule layer has **low confidence or conflicts**, send **minimal evidence fragments** to LLM; supports **OpenAI Structured Outputs** and local **Ollama**.
- **Strict Structured Output**: Pydantic/JSONSchema validation, preserving `evidence_spans` (field values + reference locations) for auditing.
- **Detailed Progress Monitoring**: Real-time display of LLM call progress, sent content, and responses, with debug mode support.

## 📊 Actual Performance

<img src="./assets/1.jpg" alt="Performance Statistics Chart" style="width:300px; height:auto;" />

**Field Extraction Statistics**:
- 26 different types of fields successfully extracted
- Average of 24.4 fields extracted per document
- High-frequency fields (appearing 5 times): project name, bidder, contact info, dates, etc.
- Medium-frequency fields (appearing 3-4 times): business scope, bid amount, business license, etc.
- Low-frequency fields (appearing 1-2 times): registered capital, shareholder info, project manager, etc.

---

## 🛠️ Installation Guide

### System Requirements
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- Optional: Ollama (for local LLM inference)

### Quick Installation

```bash
# Clone the project
git clone <repository-url>
cd tender-extract

# Install basic dependencies + CLI
uv sync --extra cli

# Install Chinese NER (optional)
uv sync --extra ner

# Set Ollama address (if using local LLM)
export OLLAMA_BASE_URL=http://your-ollama-server:11434
```

### Verify Installation

```bash
# Check if CLI is available
uv run tender-extract --help

# Run simple test
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm none --verbose
```

---

## 🎯 Usage Guide

### Basic Usage

```bash
# Rule-based extraction only (fastest, recommended)
uv run tender-extract extract ./examples/sample.md --out ./out --use-ner --llm none

# Batch process entire directory
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm none --verbose
```

### Advanced Usage

```bash
# Enable detailed progress display
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm ollama --verbose

# Enable LLM debug mode (show complete prompts and responses)
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm ollama --verbose --debug

# Use OpenAI (requires API key)
export OPENAI_API_KEY=your-api-key
uv run tender-extract extract ./examples/ --out ./out --llm openai --model gpt-4o-mini

# Local Ollama
uv run tender-extract extract ./examples/ --out ./out --llm ollama --model deepseek-r1:32b
```

### 📁 Batch Process Your `.md` Files:

```bash
# Rules/dictionary only (fastest)
uv run tender-extract extract /path/to/md_dir --pattern "*.md" --out ./out --llm none

# Enable OpenAI (strict JSON output)
uv run tender-extract extract /path/to/md_dir --out ./out --llm openai --model gpt-4o-mini

# Local Ollama
uv run tender-extract extract /path/to/md_dir --out ./out --llm ollama --model deepseek-r1:32b
```

### 🔧 Command Line Parameters

```bash
uv run tender-extract --help
uv run tender-extract extract --help
```

**Main Parameters**:
- `input_path`: Input file or directory (Markdown)
- `--pattern`: Pattern matching when input_path is a directory (default "*.md")
- `--out`: Output directory (default ./out)
- `--config`: Rules/dictionary YAML (default ./config/example.yaml)
- `--use-ner`: Enable Chinese NER (requires foolnltk)
- `--llm`: none | ollama | openai
- `--model`: LLM model name (e.g., deepseek-r1:32b)
- `--cache-dir`: Cache directory (default ./.cache)
- `--verbose`: Show detailed processing information
- `--debug`: LLM debug mode, show complete prompts and responses

---

## 📂 Project Structure

```bash
tender-extract/
├── pyproject.toml                # uv/dependencies/script entry
├── README.md
├── config/
│   └── example.yaml              # Regex and dictionary configuration
├── data/dicts/
│   └── keywords_zh.txt           # Keyword dictionary
├── examples/                     # Sample documents
│   └── example.md
├── out/                          # Output directory
│   └── example.md.json
└── src/tender_extract/
    ├── cli.py                    # CLI (Typer)
    ├── preprocess.py             # Markdown cleaning + chapter tree
    ├── chunker.py                # Chapter priority + recursive slicing
    ├── rules.py                  # Regex + keyword heuristic extraction
    ├── ner.py                    # Optional: foolnltk
    ├── dedupe.py                 # RapidFuzz + MinHash
    ├── llm_router.py             # OpenAI / Ollama adapter
    ├── merge.py                  # Field merging strategy
    └── schema.py                 # Pydantic output model
```

---

## ⚙️ Configuration & Extension

### Rules/Dictionary Configuration

Edit `config/example.yaml` and `data/dicts/keywords_zh.txt`:

```yaml
# config/example.yaml
patterns:
  date:
    - pattern: r'(\d{4}年\d{1,2}月\d{1,2}日)'
      confidence: 0.9
  amount:
    - pattern: r'人民币[壹贰叁肆伍陆柒捌玖拾佰仟万亿]+元'
      confidence: 0.8

synonyms:
  - [评标办法, 资格条件, 联合体]
  - [法定代表人, 法人代表, 负责人]
```

### NER Configuration

Used with `--use-ner` to supplement organization/person/location entity candidates, can be fused with rule layer voting.

### LLM Routing Configuration

`src/tender_extract/llm_router.py` supports ollama; only triggered when fields have low confidence or conflicts.

---

## 🔍 How It Works

1. **Preprocessing & Slicing**: Parse Markdown → chapter tree; recursive character splitting for long paragraphs (~600–800 tokens level)
2. **Rule Layer Extraction**: Extract "hard fields" like amounts/dates/numbers/deposits/contact info first; multi-keyword phrases use Aho–Corasick linear scanning
3. **Deduplication & Template Recognition**: RapidFuzz + MinHash LSH, reuse parsing results from template paragraphs
4. **On-Demand LLM**: Only send "minimal evidence fragments" to LLM when low confidence/conflicts, using Structured Outputs/function calls for strict JSON

---

## 📈 Performance Optimization Tips

1. **Rules First, Models Later**: Rules/dictionary layer typically covers 60–90% of hard fields; LLM only supplements difficult cases
2. **Control Fragment Length**: Chapter priority + minimal overlap recursive slicing
3. **Build Cache**: Create fingerprints for fragment text (like MinHash), directly reuse extraction results for same/similar paragraphs
4. **Parallelization**: Preprocessing, rule extraction, and similarity detection can be multi-processed; LLM uses small batch concurrency with rate limiting

---

## 🎯 Use Cases

- **Bidding Agencies**: Batch process tender documents, extract key information
- **Evaluation Experts**: Quickly obtain core tender information, assist evaluation decisions
- **Regulatory Bodies**: Automate tender compliance review
- **Research Institutions**: Analyze tender data for market research
- **Enterprise Bidding**: Quickly analyze competitor tender information

---

## 📊 Output Format

Each document generates a corresponding JSON file containing:

```json
{
  "metadata": {
    "filename": "example.md",
    "file_size": 12345,
    "total_lines": 500,
    "total_chunks": 10,
    "processing_time": 2.31,
    "extraction_stats": {
      "total_fields": 24,
      "avg_confidence": 0.85
    }
  },
  "fields": {
    "project_name": {
      "field_type": "project_name",
      "primary_value": "Test Engineering Project",
      "confidence": 0.95,
      "values": [
        {
          "value": "Test Engineering Project",
          "confidence": 0.95,
          "source": "rules",
          "start": 100,
          "end": 110
        }
      ]
    }
  },
  "chunks_processed": 10,
  "llm_calls": 3,
  "cache_hits": 2
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Installation Failure**
```bash
# Ensure correct Python version
python --version  # Should be 3.9+

# Reinstall dependencies
uv sync --reinstall
```

**2. Ollama Connection Failure**
```bash
# Check Ollama service status
curl http://your-ollama-server:11434/api/tags

# Set correct environment variables
export OLLAMA_BASE_URL=http://your-ollama-server:11434
```

**3. NER Module Error**
```bash
# Reinstall NER dependencies
uv sync --extra ner --reinstall
```

**4. Insufficient Memory**
```bash
# Reduce slice size
uv run tender-extract extract ./examples/ --out ./out --llm none
```

### Debugging Tips

1. **Use Verbose Mode**: Add `--verbose` parameter to view detailed logs
2. **Enable Debug Mode**: Add `--debug` parameter to view complete LLM interactions
3. **Check Configuration**: Ensure `config/example.yaml` format is correct
4. **Review Output Files**: Check JSON files in `out/` directory

---

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

--- 