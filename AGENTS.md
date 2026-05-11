# AGENTS.md

## Cursor Cloud specific instructions

### Overview

**tender-extract** is a Python CLI tool for extracting structured information from Chinese tender/bid documents. It uses a hybrid pipeline: rules/dictionary/NER for deterministic fields, with optional LLM routing for low-confidence extractions.

### Running the application

```bash
# Rules-only extraction (no external services needed)
uv run tender-extract extract ./examples/ --out ./out --llm none --verbose

# Show file info
uv run tender-extract info ./examples/example.md

# Run built-in component test
uv run tender-extract test ./examples/example.md
```

### Key notes

- **Python 3.12+** is required. The VM has Python 3.12.3.
- **Package manager**: `uv` (installed to `~/.local/bin`). Always use `uv run` to execute commands within the project virtualenv.
- **No automated test suite**: There is no `tests/` directory or pytest config. The built-in `tender-extract test` command exercises core components (preprocessing, chunking, rule extraction, NER).
- **No lint config**: There is no ruff/flake8/mypy configuration in `pyproject.toml`. Basic Python syntax checking can be done with `uv run python -m py_compile src/tender_extract/<file>.py`.
- **jieba SyntaxWarnings**: The `jieba` NER library emits `SyntaxWarning` about invalid escape sequences on Python 3.12. These are harmless warnings from the third-party library, not from project code.
- **uv.toml mirrors**: The `uv.toml` points to Chinese PyPI mirrors (Aliyun/Tsinghua) as primary index, with `pypi.org` as fallback. This works fine from cloud VMs.
- **LLM modes**: Use `--llm none` for fully self-contained testing. `--llm ollama` or `--llm openai` require external services/API keys respectively.
- **Working directory**: The CLI uses relative paths for config (`config/example.yaml`) and data (`data/dicts/`), so always run from the repository root (`/workspace`).
