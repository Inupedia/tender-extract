# LLM Provider 配置

tender-extract 的 LLM 只用于低置信、冲突或缺失字段复核。主抽取流程不依赖某一家模型厂商。

## 支持方式

Provider 分为四类：

1. **OpenAI-compatible**：绝大多数云厂商共用 `chat/completions` 适配器。
2. **Anthropic**：使用原生 Messages API。
3. **Azure OpenAI**：使用 Azure endpoint + deployment name。
4. **本地 / 自托管**：Ollama、vLLM、LM Studio，以及任意 OpenAI-compatible endpoint。

内置 Provider 包括：

- OpenAI
- Azure OpenAI
- Anthropic Claude
- Google Gemini
- DeepSeek
- 阿里云通义千问 / DashScope
- Moonshot / Kimi
- 智谱 GLM
- 火山方舟 / Doubao
- 腾讯混元
- 百川
- MiniMax
- 零一万物
- 阶跃星辰
- SiliconFlow
- OpenRouter
- Groq
- Together AI
- Mistral
- xAI / Grok
- Fireworks
- Perplexity
- NVIDIA NIM
- Ollama
- vLLM
- LM Studio
- 任意 OpenAI-compatible 服务

运行下面的命令可以查看当前版本实际注册的 Provider、默认模型和环境变量：

```bash
uv run tender-extract providers
```

HTTP Server 可通过下面的接口做 Provider discovery：

```bash
curl http://localhost:8000/v1/providers
```

## CLI 示例

### DeepSeek

```bash
export DEEPSEEK_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm deepseek \
  --model deepseek-chat \
  --out out
```

### OpenAI

```bash
export OPENAI_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm openai \
  --out out
```

### Gemini

```bash
export GEMINI_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm gemini \
  --out out
```

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm anthropic \
  --out out
```

### SiliconFlow

```bash
export SILICONFLOW_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm siliconflow \
  --model Qwen/Qwen3-8B \
  --out out
```

## 本地模型

### Ollama

```bash
uv run tender-extract extract examples/example.pdf \
  --llm ollama \
  --model qwen2.5:14b
```

### vLLM

```bash
uv run tender-extract extract examples/example.pdf \
  --llm vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B
```

### LM Studio

```bash
uv run tender-extract extract examples/example.pdf \
  --llm lmstudio \
  --model your-loaded-model
```

本地地址、localhost、私网 IP 会被识别为本地 endpoint，不会因为云端 LLM 隐私策略而自动对发送文本做 PII 脱敏。

## 任意 OpenAI-compatible endpoint

对于 vLLM、TGI、企业模型网关、反向代理以及其他兼容 OpenAI Chat Completions 的服务，不需要新增代码：

```bash
export LLM_BASE_URL=https://llm.example.com/v1
export LLM_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm openai_compat \
  --model your-model
```

也可以直接通过参数传入 endpoint：

```bash
uv run tender-extract extract examples/example.pdf \
  --llm openai_compat \
  --base-url http://127.0.0.1:9000/v1 \
  --model your-model
```

## HTTP Server

服务器级默认 Provider：

```bash
docker run --rm -p 8000:8000 \
  -e DEEPSEEK_API_KEY=your-key \
  -e TENDER_SERVER_LLM_PROVIDER=deepseek \
  -e TENDER_SERVER_LLM_MODEL=deepseek-chat \
  ghcr.io/inupedia/tender-extract-server:latest
```

单次请求可以覆盖 Provider / model / base URL：

```bash
curl -F "file=@example.pdf" \
  "http://localhost:8000/v1/extract?llm_provider=gemini&llm_model=gemini-3.8-flash"
```

如果不希望把上游 LLM API Key 固定在服务器环境变量中，可以使用请求头：

```bash
curl \
  -H "X-LLM-API-Key: your-upstream-key" \
  -F "file=@example.pdf" \
  "http://localhost:8000/v1/extract?llm_provider=openrouter&llm_model=openai/gpt-4o-mini"
```

`X-LLM-API-Key` 只用于当前请求，不会出现在响应中。生产部署仍建议优先使用服务端 Secret / 环境变量，并通过 `TENDER_SERVER_API_KEY` 保护抽取接口。

## 环境变量解析顺序

API Key：

1. CLI `--api-key` / HTTP `X-LLM-API-Key`
2. Server 默认 `TENDER_SERVER_LLM_API_KEY`（仅默认 Provider）
3. Provider 专属环境变量，例如 `DEEPSEEK_API_KEY`
4. 通用 `LLM_API_KEY`

Base URL：

1. CLI `--base-url` / HTTP `llm_base_url`
2. Server 默认 `TENDER_SERVER_LLM_BASE_URL`（仅默认 Provider）
3. Provider 专属环境变量
4. `LLM_BASE_URL`
5. Provider 内置默认 endpoint

## 不直接适配的平台

AWS Bedrock、Google Vertex AI 等平台使用独立的云身份认证与请求协议，目前没有伪装成“原生支持”。如果企业已经通过 LiteLLM、vLLM gateway、API gateway 等方式暴露 OpenAI-compatible endpoint，可以直接使用 `openai_compat` 接入；后续也可以在 Provider Registry 中增加独立 adapter，而无需改抽取流水线。
