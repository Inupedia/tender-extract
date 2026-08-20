# tender-extract · 中文标书结构化提取

[English](README_EN.md) | 中文

> **把几十到几百页的采购 / 招投标文档，快速变成可核查的结构化数据。**
> 规则负责快，大模型只处理不确定项；结果不仅有值，还能回到原文页码和位置核查。

![真实招投标文档验收结果](assets/acceptance-benchmark.svg)

## 为什么用它

- ⚡ **快**：一套 **13 份、911 页**的真实采购 / 招投标文档验收集，不启用大模型时实测 **26.31 秒，34.62 页/秒**。
- 🎯 **结果可靠**：项目名称、编号、采购人、金额、日期、联系人、人员、证书等字段都带置信度。
- 🔎 **能回原文**：PDF 可保留 **文件、页码、原文片段和坐标**，方便点击定位和高亮。
- 🧠 **大模型按需使用**：只把低置信、冲突或缺失字段交给大模型，避免整本 PDF 全量调用。
- 📦 **一个项目可以多份文件一起处理**：招标文件、补遗、澄清、多个投标人的文件统一建模，同时保留版本关系和投标人隔离。
- ✅ **能持续变准**：人工复核结果可以回流成标注数据，并通过 F1 评测和持续集成防止回归。

## 最省事：直接启动 HTTP 服务

不需要先配 Python 环境，直接拉取服务镜像：

```bash
docker pull ghcr.io/inupedia/tender-extract-server:0.1.0

docker run --rm \
  -p 8000:8000 \
  -v tender-extract-cache:/data/cache \
  ghcr.io/inupedia/tender-extract-server:0.1.0
```

上传一份 PDF：

```bash
curl -s \
  -F "file=@example.pdf" \
  "http://localhost:8000/v1/extract?llm_provider=none"
```

接口文档启动后直接打开 `http://localhost:8000/docs`。

服务提供：

```text
GET  /healthz      健康检查
GET  /v1/info      版本与能力
POST /v1/extract   上传 PDF / DOCX / Markdown / TXT 并返回结构化结果
```

默认不返回敏感个人信息。需要给服务加访问密钥时：

```bash
docker run --rm -p 8000:8000 \
  -e TENDER_SERVER_API_KEY=your-secret \
  ghcr.io/inupedia/tender-extract-server:0.1.0
```

调用时增加 `X-API-Key` 请求头即可。

需要大模型时把对应密钥传入容器，例如 SiliconFlow：

```bash
docker run --rm -p 8000:8000 \
  -e SILICONFLOW_API_KEY=your-key \
  -e TENDER_SERVER_LLM_PROVIDER=siliconflow \
  -e TENDER_SERVER_LLM_MODEL=Qwen/Qwen3-8B \
  -v tender-extract-cache:/data/cache \
  ghcr.io/inupedia/tender-extract-server:0.1.0
```

镜像只在创建版本标签时发布：例如 Git tag `v0.1.0` 会发布 `:0.1.0`，并同步更新 `:latest`。普通提交到 `main` 不会发布镜像。

## 本地命令行

要求 Python **3.12+**。

```bash
git clone https://github.com/Inupedia/tender-extract.git
cd tender-extract

uv sync --extra pdf
uv run tender-extract extract examples/example.pdf --out out
```

处理完成后得到：

```text
out/example.pdf.json
```

批量处理目录也一样简单：

```bash
uv run tender-extract extract ./documents --pattern "*.pdf" --out out
```

## 实际会得到什么

仓库中的 `examples/example.pdf` 是这套验收集中的一份 10 页政府采购文件，当前可以稳定提取：

```text
项目名称    合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目
项目编号    2024BFFFZ01583
采购人      合肥市公安局瑶海分局
金额        437.677万元
日期        2024年7月1日17时30分
联系电话    055166223642
```

输出不是简单的 `key: value`。例如项目名称会同时保留证据位置：

```json
{
  "project_name": {
    "primary_value": "合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目",
    "confidence": 0.99,
    "values": [
      {
        "location": {
          "document_id": "example.pdf",
          "page": 1,
          "source_text": "合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目"
        }
      }
    ]
  }
}
```

PDF 能可靠定位时还会返回真实坐标，因此上层系统可以继续实现：

**字段 → 来源文件 → 页码 → 原文位置 → PDF 高亮**

## 真实文档验收

`examples/` 本身就是一套完整的真实采购 / 招投标 PDF 验收集，覆盖 **安徽、北京、陕西、河南、上海等地区，共 13 份、911 页**。整套文档统一用于验证解析、字段抽取和证据定位。

| 实测指标 | 结果 |
|---|---:|
| 文档数量 | **13 份** |
| 总页数 | **911 页** |
| 总耗时 | **26.31 秒** |
| 处理速度 | **34.62 页/秒** |
| 字段语义校验 | **14 / 14 通过** |
| 运行失败 | **0** |

> 上述速度来自 GitHub Actions，且**不启用大模型**。不同机器、PDF 版式和 OCR 情况会影响速度。

文件来源、页数和 SHA256 记录在 [`examples/public-corpus.lock.json`](examples/public-corpus.lock.json)，正常测试直接使用仓库内文件，不依赖外部网站在线状态。

复现验收：

```bash
uv run python scripts/acceptance_corpus.py \
  --examples examples \
  --min-pdfs 13 \
  --report artifacts/real-pdf-acceptance.json
```

## 为什么不是“整本 PDF 全丢给大模型”

```text
PDF / DOCX / Markdown / TXT
            │
            ▼
      文档解析与分块
            │
            ▼
       规则与结构抽取
        │           │
   高置信直接输出   低置信 / 冲突 / 缺失
        │           │
        │           ▼
        │        大模型复核
        │           │
        └─────┬─────┘
              ▼
        合并 + 证据定位
              │
              ▼
        结构化 JSON
```

这样确定字段不用等待模型，也不会产生额外调用成本；难字段仍然可以交给模型理解。

SiliconFlow 已使用真实 `Qwen/Qwen3-8B` 接口完成验收：**4/4 网络调用成功**，同一文档第二次运行 **4 次全部命中缓存、0 次新增网络调用**，对应标注集的 **微平均 F1 / 宏平均 F1 均为 1.000**。外部模型耗时受网络和服务端排队影响，因此不与上面的本地处理速度混在一起计算。

启用 SiliconFlow：

```bash
export SILICONFLOW_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm siliconflow \
  --model Qwen/Qwen3-8B \
  --out out
```

也支持 OpenAI、DeepSeek、通义千问、Claude、Gemini、Ollama 以及其他 OpenAI 兼容接口：

```bash
uv run tender-extract providers
```

## 一个项目有很多文件，也可以一起处理

现实项目通常同时包含招标文件、补遗、澄清和多个投标人的文件。`tender-package` 可以把它们作为**一个项目**处理，并管理版本、替代关系和投标人边界。

```text
某项目/
├── 招标文件.pdf
├── 补遗01.pdf
├── 澄清01.pdf
├── A公司投标文件.pdf
└── B公司投标文件.pdf
```

```bash
uv run tender-package validate package.yaml
uv run tender-package extract package.yaml --out out/package.json
```

补遗中的新值可以覆盖对应旧值，但历史来源仍然保留；A 公司的字段也不会覆盖 B 公司。

详见 [`docs/tender-package.md`](docs/tender-package.md)。

## 人工复核和质量评测

低置信、冲突或模型恢复字段可以进入人工复核队列，修改结果直接回流成标注集：

```bash
uv run tender-review run examples/example.pdf --queue .review/queue.jsonl
uv run tender-review export --queue .review/queue.jsonl --out eval/gold-reviewed.jsonl
uv run tender-extract eval eval/gold-reviewed.jsonl
```

持续集成也可以直接设置最低 F1：

```bash
uv run tender-extract eval eval/gold.jsonl --fail-under 0.95
```

详见 [`docs/human-review.md`](docs/human-review.md) 和 [`docs/evaluation.md`](docs/evaluation.md)。

## 支持能力

| 能力 | 当前支持 |
|---|---|
| 使用方式 | Python 命令行、HTTP 服务、Docker / GHCR 镜像 |
| 文档格式 | PDF、DOCX、Markdown、TXT |
| 扫描件 | 可选 OCR；轻量服务镜像默认不内置 PaddleOCR |
| 字段抽取 | 规则、词典、命名实体识别、大模型复核 |
| 证据定位 | 文件、页码、章节、行号、原文、PDF 坐标 |
| 项目级处理 | 多文件、版本、补遗/澄清、投标人隔离 |
| 人工复核 | 接受、修改、驳回、导出标注集 |
| 质量评测 | 精确率、召回率、F1、持续集成质量门禁 |
| 隐私 | 默认脱敏敏感个人信息；服务可选 API Key |

更多细节：

- [证据定位](docs/evidence.md)
- [项目级多文件处理](docs/tender-package.md)
- [人工复核](docs/human-review.md)
- [质量评测](docs/evaluation.md)

## 许可证

MIT
