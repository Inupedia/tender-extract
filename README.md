# tender-extract · 中文标书结构化提取

[English](README_EN.md) | 中文

> **把几十到几百页的采购 / 招投标文档，快速变成可核查的结构化数据。**
> 规则负责快，大模型只处理不确定项；结果不仅有值，还能回到原文页码和位置核查。

![真实招投标文档验收结果](assets/acceptance-benchmark.svg)

## 为什么用它

- ⚡ **快**：真实验收集共 **13 份文档、911 页**，不启用大模型时实测 **26.31 秒，34.62 页/秒**。
- 🎯 **不是只“抽到就算”**：项目名称、编号、采购人、金额、日期、联系人、人员、证书等结果都带置信度。
- 🔎 **能回原文**：PDF 可保留 **文件、页码、原文片段和坐标**，方便后续做点击定位和高亮。
- 🧠 **大模型按需使用**：低置信、冲突或缺失字段才交给大模型，避免整本 PDF 全量调用。
- 📦 **支持一个项目多份文件**：招标文件、补遗、澄清、多个投标人的文件可以统一处理，同时保留版本关系和投标人隔离。
- ✅ **能持续变准**：人工复核结果可以回流成标注数据，并通过 F1 评测和 CI 防止回归。

## 30 秒跑起来

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

也可以直接批量处理目录：

```bash
uv run tender-extract extract ./documents --pattern "*.pdf" --out out
```

## 实际会得到什么

仓库中的 `examples/example.pdf` 是一份 10 页政府采购文件，当前可以稳定提取出这些核心信息：

```text
项目名称    合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目
项目编号    2024BFFFZ01583
采购人      合肥市公安局瑶海分局
金额        437.677万元
日期        2024年7月1日17时30分
联系电话    055166223642
```

输出不是简单的 `key: value`。例如项目名称会保留证据位置：

```json
{
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
```

PDF 能可靠定位时还会返回真实坐标，因此上层系统可以继续实现：

**字段 → 来源文件 → 页码 → 原文位置 → PDF 高亮**

## 真实文档验收

`examples/` 现在是一套完整的真实采购 / 招投标 PDF 验收集，覆盖 **安徽、北京、陕西、河南、上海等地区，共 13 份、911 页**。这些文件作为同一套回归语料持续验证解析、字段抽取和证据定位。

| 实测指标 | 结果 |
|---|---:|
| 文档数量 | **13 份** |
| 总页数 | **911 页** |
| 总耗时 | **26.31 秒** |
| 处理速度 | **34.62 页/秒** |
| 字段语义校验 | **14 / 14 通过** |
| 运行失败 | **0** |

> 上述速度来自 GitHub Actions，且**不启用大模型**。不同机器、PDF 版式和 OCR 情况会影响速度。

公开文件来源、页数和 SHA256 已记录在 [`examples/public-corpus.lock.json`](examples/public-corpus.lock.json)，正常测试直接使用仓库内文件，不依赖外部网站在线状态。

复现这套验收：

```bash
uv run python scripts/acceptance_corpus.py \
  --examples examples \
  --min-pdfs 13 \
  --report artifacts/real-pdf-acceptance.json
```

## 为什么不是“整本 PDF 全丢给大模型”

`tender-extract` 的处理方式是：

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

这样大部分确定字段不需要等待模型，也不会产生额外调用成本；难字段仍然可以交给模型理解。

SiliconFlow 已使用真实 `Qwen/Qwen3-8B` 接口完成验收：**4/4 网络调用成功**，同一文档第二次运行 **4 次全部命中缓存、0 次新增网络调用**，对应标注集的 **Micro F1 / Macro F1 均为 1.000**。外部模型耗时受网络和服务端排队影响，因此不与上面的 34.62 页/秒本地基线混在一起计算。

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

真实项目往往同时有招标文件、补遗、澄清和多个投标人的文件。`tender-package` 可以把它们作为**一个项目**处理，并明确管理版本、替代关系和投标人边界。

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
uv run tender-package inspect package.yaml
uv run tender-package extract package.yaml --out out/package.json
```

这样补遗中的新值可以覆盖对应旧值，但历史来源仍然保留；A 公司的投标字段也不会覆盖 B 公司。

详见 [`docs/tender-package.md`](docs/tender-package.md)。

## 人工复核和质量评测

低置信、冲突或模型恢复的字段可以进入人工复核队列：

```bash
uv run tender-review run examples/example.pdf --queue .review/queue.jsonl
uv run tender-review list --queue .review/queue.jsonl
```

人工修改后的结果可以直接导出为标注集，再进入下一轮自动评测：

```bash
uv run tender-review export --queue .review/queue.jsonl --out eval/gold-reviewed.jsonl
uv run tender-extract eval eval/gold-reviewed.jsonl
```

也可以在 CI 中设置最低 F1：

```bash
uv run tender-extract eval eval/gold.jsonl --fail-under 0.95
```

详见 [`docs/human-review.md`](docs/human-review.md) 和 [`docs/evaluation.md`](docs/evaluation.md)。

## 支持能力

| 能力 | 当前支持 |
|---|---|
| 文档格式 | PDF、DOCX、Markdown、TXT |
| 扫描件 | 可选 OCR |
| 字段抽取 | 规则、词典、NER、大模型复核 |
| 证据定位 | 文件、页码、章节、行号、原文、PDF 坐标 |
| 项目级处理 | 多文件、版本、补遗/澄清、投标人隔离 |
| 人工复核 | 接受、修改、驳回、导出标注集 |
| 质量评测 | Precision、Recall、F1、CI 质量门禁 |
| 隐私 | 默认脱敏敏感个人信息 |

更多细节：

- [证据定位](docs/evidence.md)
- [项目级多文件处理](docs/tender-package.md)
- [人工复核](docs/human-review.md)
- [质量评测](docs/evaluation.md)

## License

MIT
