# 标书信息提取工具

[English](README_EN.md) | 中文

## 项目介绍

`tender-extract` 是面向中文招投标文档的混合抽取工具：规则层先抽取高置信字段，仅在低置信或冲突时把最小证据片段交给大模型。每个字段保留原文证据，默认对身份证号脱敏。

流水线：

```
输入(PDF/DOCX/MD/TXT) → 解析 → 切块 → 模块路由 → 规则抽取 → 按需 LLM → JSON
```

## 安装

```bash
git clone <repository-url>
cd tender-extract
uv sync --extra all          # 含 PDF / DOCX / OCR
# 或最小安装
uv sync
uv sync --extra pdf --extra docx
```

需要 Python 3.12+。

```bash
uv run tender-extract --help
uv run tender-extract providers
```

## 用法

```bash
# 规则抽取（默认不调模型）
uv run tender-extract extract ./examples/example.md --out ./out

# 目录批量，支持 PDF / DOCX / MD / TXT
uv run tender-extract extract ./招标文件/ --out ./out --verbose

# 低置信字段走大模型
export OPENAI_API_KEY=sk-...
uv run tender-extract extract ./examples --out ./out --llm openai --model gpt-4o-mini

export DEEPSEEK_API_KEY=...
uv run tender-extract extract ./examples --out ./out --llm deepseek

export ANTHROPIC_API_KEY=...
uv run tender-extract extract ./examples --out ./out --llm anthropic --model claude-sonnet-4-5

# 本地 Ollama（OpenAI 兼容 /v1）
uv run tender-extract extract ./examples --out ./out --llm ollama --model qwen2.5:14b

# 任意 OpenAI 兼容网关
uv run tender-extract extract ./examples --out ./out \
  --llm openai_compat --base-url https://your-gateway/v1 --api-key "$LLM_API_KEY" --model gpt-4o-mini
```

`extract-v2` 是 `extract` 的兼容别名。

### 常用参数

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置，CLI 会真正读取并允许覆盖 |
| `--llm` | 提供商 ID，见 `tender-extract providers` |
| `--model` | 模型名；Azure 填部署名 |
| `--base-url` | 覆盖 API 地址 |
| `--include-pii` | 输出完整身份证号（默认掩码） |
| `--modules / --no-modules` | 是否按章节路由到专业模块 |
| `--use-ner` | jieba NER 补充 |

## 支持的 LLM

通过 OpenAI Chat Completions 兼容协议接入（Anthropic 为原生 Messages API）：

OpenAI、Azure OpenAI、Anthropic Claude、Google Gemini、Ollama、DeepSeek、通义千问 / DashScope、Moonshot Kimi、智谱 GLM、火山方舟 Doubao、腾讯混元、百川、MiniMax、零一万物、阶跃星辰、硅基流动、OpenRouter、Groq、Together、Mistral、xAI Grok、Fireworks、Perplexity。

密钥走对应环境变量，例如 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`ARK_API_KEY`。完整列表：`uv run tender-extract providers`。

## 输出

每个输入文件对应一个 JSON，人员与证书绑定在**该文件**的结果里，不会串到其它文档。

```json
{
  "metadata": { "filename": "example.md", "processing_time": 0.2 },
  "fields": {
    "bid_amount": {
      "primary_value": "1000000.00元",
      "confidence": 0.95,
      "values": [{ "value": "1000000.00元", "unit": "元", "normalized_value": "1000000.00", "ref": "..." }]
    }
  },
  "personnel": [],
  "certificates": []
}
```

金额保留单位，并给出换算成人民币元的 `normalized_value`，用于保证金与投标金额交叉校验。

## 开发

```bash
uv sync --extra all
uv run pytest
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
