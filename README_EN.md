# tender-extract

English | [中文](README.md)

Hybrid extraction for Chinese tender documents: rules first, LLM only for low-confidence or conflicting spans. Evidence snippets are kept for audit. ID numbers are masked unless `--include-pii` is set.

```
PDF/DOCX/MD/TXT → parse → chunk → module route → regex extract → on-demand LLM → JSON
```

## Install

Python 3.12+.

```bash
uv sync --extra all
uv run tender-extract --help
uv run tender-extract providers
```

## Usage

```bash
uv run tender-extract extract ./examples/example.md --out ./out

export OPENAI_API_KEY=sk-...
uv run tender-extract extract ./docs --out ./out --llm openai --model gpt-4o-mini

uv run tender-extract extract ./docs --out ./out --llm deepseek
uv run tender-extract extract ./docs --out ./out --llm anthropic
uv run tender-extract extract ./docs --out ./out --llm ollama --model qwen2.5:14b
```

`extract-v2` is an alias of `extract`. `--config` is actually loaded. `--include-pii` writes full ID numbers.

OpenAI-compatible providers include OpenAI, Azure, Gemini, DeepSeek, Qwen, Kimi, GLM, Doubao, Groq, OpenRouter, Mistral, xAI, and more. Anthropic uses the native Messages API. See `tender-extract providers`.

## Develop

```bash
uv sync --extra all
uv run pytest
```

MIT license.
