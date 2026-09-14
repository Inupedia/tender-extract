<div align="center">

# tender-extract

**面向中文采购 / 招投标文档的结构化抽取工具。**

规则与结构抽取优先，低置信字段可选 LLM 复核；输出保留页码、原文与 PDF 坐标证据。

<p>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/FastAPI-HTTP_API-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI HTTP API">
  <img src="https://img.shields.io/badge/Docker-GHCR-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker GHCR">
  <img src="https://img.shields.io/badge/GitHub_Actions-tested-2088FF?style=flat-square&logo=githubactions&logoColor=white" alt="GitHub Actions tested">
  <img src="https://img.shields.io/badge/Formats-PDF%20%7C%20DOCX%20%7C%20Markdown%20%7C%20TXT-57606A?style=flat-square" alt="PDF DOCX Markdown TXT">
</p>

</div>

<p align="center">
  <img src="./assets/acceptance-benchmark.svg" width="100%" alt="tender-extract 真实文档验收：13 份真实 PDF、911 页、26.31 秒、34.62 页每秒、运行失败文档 0">
</p>

## 先看它能得到什么

仓库里的 [`examples/example.pdf`](examples/example.pdf) 是一份 10 页政府采购文件。当前流水线可以从中得到类似下面的结构化结果：

```text
项目名称    合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目
项目编号    2024BFFFZ01583
采购人      合肥市公安局瑶海分局
金额        437.677万元
日期        2024年7月1日17时30分
联系电话    055166223642
```

它不是简单的 `key: value`。字段可以同时保留来源证据：

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

对于能够可靠定位的 PDF 文本，还可以保留页面坐标，因此上层系统可以继续实现：

```text
字段 → 来源文件 → 页码 → 原文片段 → PDF 坐标 / 高亮
```

## 最快开始：Docker

直接使用最新发布镜像，不需要先准备 Python 环境：

```bash
IMAGE=ghcr.io/inupedia/tender-extract-server:latest

docker pull "$IMAGE"
docker run --rm \
  -p 8000:8000 \
  -v tender-extract-cache:/data/cache \
  "$IMAGE"
```

上传一份 PDF：

```bash
curl -s \
  -F "file=@example.pdf" \
  "http://localhost:8000/v1/extract?llm_provider=none"
```

启动后可直接打开 `http://localhost:8000/docs` 查看 OpenAPI 文档。

> `latest` 适合快速体验和跟随最新正式版本。生产环境需要可复现部署时，建议固定到 `ghcr.io/inupedia/tender-extract-server:<version>`。Git tag `vX.Y.Z` 会发布 Docker tag `:X.Y.Z`，并同步更新 `:latest`。

[查看 GHCR 镜像版本](https://github.com/Inupedia/tender-extract/pkgs/container/tender-extract-server)

## 抽取流程

```text
PDF / DOCX / Markdown / TXT
            │
            ▼
      文档解析与分块
            │
            ▼
      规则 / 结构抽取
        │           │
   高置信直接输出   低置信 / 冲突 / 缺失
        │           │
        │           ▼
        │        LLM 复核
        │           │
        └─────┬─────┘
              ▼
        合并 + 证据定位
              │
              ▼
        结构化 JSON
```

默认先用规则与结构化方法完成可确定字段；只有低置信、冲突或缺失字段才进入可选的 LLM 复核流程。

## 核心能力

| 能力 | 做什么 |
|---|---|
| **混合抽取** | 规则、词典、NER 与可选 LLM 复核组合使用 |
| **证据定位** | 保留文件、页码、原文片段，PDF 可进一步保留 bbox |
| **项目级多文件** | 招标文件、补遗、澄清、多个投标人文件统一处理，同时保留版本与投标人边界 |
| **人工复核** | 接受、修改、驳回低置信结果，并回流为标注数据 |
| **质量评测** | Precision / Recall / F1 与 CI 质量门禁 |
| **多种入口** | Python CLI、HTTP API、Docker / GHCR |

### 当前支持

| 项目 | 支持情况 |
|---|---|
| 文档格式 | PDF、DOCX、Markdown、TXT |
| 扫描件 | 可选 OCR；轻量 Server 镜像默认不内置 PaddleOCR |
| LLM | OpenAI、Azure OpenAI、Claude、Gemini、DeepSeek、通义千问、Kimi、GLM、Doubao、SiliconFlow、OpenRouter、Groq、Together、Mistral、xAI、NVIDIA NIM、Ollama、vLLM、LM Studio，以及任意 OpenAI-compatible endpoint |
| 隐私 | 默认不返回敏感个人信息；HTTP 服务可选 API Key |

## HTTP 服务

```text
GET  /healthz        健康检查
GET  /v1/info        版本与能力
GET  /v1/providers   Provider 发现与配置元数据
POST /v1/extract     上传文档并返回结构化结果
```

<details>
<summary><strong>给服务增加 API Key</strong></summary>

```bash
IMAGE=ghcr.io/inupedia/tender-extract-server:latest

docker run --rm -p 8000:8000 \
  -e TENDER_SERVER_API_KEY=your-secret \
  "$IMAGE"
```

调用时增加：

```text
X-API-Key: your-secret
```

</details>

<details>
<summary><strong>在容器中启用 LLM Provider</strong></summary>

以 DeepSeek 为例：

```bash
IMAGE=ghcr.io/inupedia/tender-extract-server:latest

docker run --rm -p 8000:8000 \
  -e DEEPSEEK_API_KEY=your-key \
  -e TENDER_SERVER_LLM_PROVIDER=deepseek \
  -e TENDER_SERVER_LLM_MODEL=deepseek-chat \
  -v tender-extract-cache:/data/cache \
  "$IMAGE"
```

SiliconFlow、OpenAI、Gemini、Claude 等 Provider 只需替换 provider/model 与对应环境变量。也可以通过 `X-LLM-API-Key` 为单次 HTTP 请求提供上游模型密钥。

</details>

## 本地 CLI

要求 Python **3.12+**。

```bash
git clone https://github.com/Inupedia/tender-extract.git
cd tender-extract

uv sync --extra pdf
uv run tender-extract extract examples/example.pdf --out out
```

批量处理目录：

```bash
uv run tender-extract extract ./documents --pattern "*.pdf" --out out
```

## LLM 复核（可选）

LLM Provider 不绑定 SiliconFlow。内置 Registry 覆盖主流云模型、本地运行时和任意 OpenAI-compatible endpoint：

```bash
uv run tender-extract providers
```

例如 DeepSeek：

```bash
export DEEPSEEK_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm deepseek \
  --model deepseek-chat \
  --out out
```

任意 OpenAI-compatible 服务：

```bash
uv run tender-extract extract examples/example.pdf \
  --llm openai_compat \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-8B \
  --out out
```

SiliconFlow `Qwen/Qwen3-8B` 已完成真实接口验收：冷启动测试 **4/4 网络调用成功**；同一文档第二次运行 **4 次全部命中缓存、0 次新增网络调用**。当前这组 live Gold acceptance case 的 **Micro F1 / Macro F1 均为 1.000**。

> 这里的 F1 是当前验收样本结果，不代表所有招投标文档上的通用准确率，也不代表其他 Provider 已完成同等 live acceptance。

更完整的 Provider、环境变量、HTTP 和本地模型配置见 [`docs/llm-providers.md`](docs/llm-providers.md)。

## 一个项目有很多文件

真实项目经常不只有一份 PDF：

```text
某项目/
├── 招标文件.pdf
├── 补遗01.pdf
├── 澄清01.pdf
├── A公司投标文件.pdf
└── B公司投标文件.pdf
```

`tender-package` 可以把它们作为一个项目处理，并管理版本替代关系与投标人隔离：

```bash
uv run tender-package validate package.yaml
uv run tender-package extract package.yaml --out out/package.json
```

补遗中的新值可以覆盖对应旧值，但历史证据仍然保留；A 公司的字段也不会覆盖 B 公司。

详见 [`docs/tender-package.md`](docs/tender-package.md)。

## 人工复核与质量闭环

```bash
uv run tender-review run examples/example.pdf --queue .review/queue.jsonl
uv run tender-review export --queue .review/queue.jsonl --out eval/gold-reviewed.jsonl
uv run tender-extract eval eval/gold-reviewed.jsonl
```

CI 也可以直接设置最低 F1：

```bash
uv run tender-extract eval eval/gold.jsonl --fail-under 0.95
```

## 真实文档验收如何复现

仓库提交了一套真实采购 / 招投标 PDF 验收集，覆盖安徽、北京、陕西、河南、上海等地区，共 **13 份 / 911 页**。当前 GitHub Actions 基准在关闭 LLM 时记录为：

| 指标 | 结果 |
|---|---:|
| 文档 | **13 份** |
| 页数 | **911 页** |
| 总耗时 | **26.31 秒** |
| 吞吐 | **34.62 页/秒** |
| 运行失败文档 | **0** |

> 速度来自 GitHub Actions。硬件、OCR、存储和 PDF 版式都会影响实际吞吐。

来源 URL、页数与 SHA256 记录在 [`examples/public-corpus.lock.json`](examples/public-corpus.lock.json)。复现：

```bash
uv run python scripts/acceptance_corpus.py \
  --examples examples \
  --min-pdfs 13 \
  --report artifacts/real-pdf-acceptance.json
```

## 文档

- [LLM Provider 配置](docs/llm-providers.md)
- [证据定位](docs/evidence.md)
- [项目级多文件处理](docs/tender-package.md)
- [人工复核](docs/human-review.md)
- [质量评测](docs/evaluation.md)

## 许可证

MIT
