# tender-extract · 中文标书结构化抽取

[English](README_EN.md) | 中文

> **不是把整本 PDF 丢给大模型。** `tender-extract` 先用高速、可复现的规则/路由处理确定性字段，只把低置信、冲突或缺失项交给 LLM；最终结果保留页码、原文和 PDF bbox，能够回到证据位置核查。

![Real PDF acceptance benchmark](assets/acceptance-benchmark.svg)

## 🚀 为什么值得用

招标文件经常几十到几百页。真正难的不只是“抽到一个值”，而是同时做到：**快、可验证、能处理版本变化、能持续纠错**。

`tender-extract` 目前已经形成完整闭环：

- ⚡ **高速规则基线**：13 份真实公开 PDF、911 页，在 GitHub Actions 上一次实测 **26.31 秒，34.62 页/秒**。
- 🧠 **按需 LLM**：只处理低置信、冲突和缺失字段；已用 SiliconFlow `Qwen/Qwen3-8B` 做真实 API 验收。
- 🔎 **证据可追溯**：字段可携带 `document_id + page + section + line + source_text + bbox`。
- 📦 **多文件项目包**：招标文件、补遗、澄清、投标文件可以组成一个 Tender Package，支持 revision / supersedes / bidder 隔离。
- 👀 **人工复核闭环**：低置信结果进入 review queue，人工 accept / correct / reject 后可直接导出 Gold Dataset。
- 📊 **质量可量化**：内置 Precision / Recall / F1 评测和 CI quality gate，不再只看“有没有输出 JSON”。

---

## 📊 真实 PDF 验收结果

PR5 不再只使用合成样本。仓库 `examples/` 中已经提交 **13 份 PDF**：原有合肥示例 + 12 份来自北京、陕西、河南、上海等公开采购/招标文件。

### 离线确定性基线

运行环境：GitHub-hosted Ubuntu 24.04、Python 3.12、PDF 解析使用 PyMuPDF；LLM 关闭。

| 指标 | 实测结果 |
|---|---:|
| PDF 数量 | **13** |
| 总页数 | **911** |
| 总文件大小 | **9.26 MB** |
| 抽取字段数 | **72** |
| 总耗时 | **26.31 s** |
| 吞吐 | **34.62 pages/s** |
| 平均每份 PDF | **2.02 s** |
| 字段级语义检查 | **14 / 14** |
| 运行失败 | **0** |
| LLM 调用 | **0** |

> 这是一次真实 CI 测量，不是 SLA。机器、PDF 版式、OCR 与磁盘环境不同，速度会变化。**34.62 pages/s 只代表不启用 LLM 的确定性基线。**

### `examples/example.pdf` 实测

仓库自带的 10 页政府采购文件现在会得到 6 个有效核心字段，平均置信度约 **0.94**，离线路径一次实测约 **0.35 秒**：

```text
project_name    合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目  0.99  page 1
project_number  2024BFFFZ01583                                             0.95  page 1
tenderer        合肥市公安局瑶海分局                                        0.95  page 1
bid_amount      437.677万元                                                0.90  page 3
bid_date        2024年7月1日17时30分                                       0.90  page 5
contact_info    055166223642                                               0.95  page 7
```

这轮真实语料验收也直接发现并修复了多类问题，包括：PDF 项目名称跨行截断、项目名称吞掉下一字段、采购代理误识别为采购人、政府采购网址误识别为采购人、旧制度日期误识别为投标截止时间，以及无标签大金额误识别为报价等。

---

## 🧠 SiliconFlow / LLM 真实验收

LLM 不是 README 里“理论支持”。当前 CI 使用真实 SiliconFlow API 验证：

```bash
export SILICONFLOW_API_KEY=your-key

uv run tender-extract extract examples/example.pdf \
  --llm siliconflow \
  --model Qwen/Qwen3-8B \
  --out out
```

一次真实验收结果：

| 路径 | 结果 |
|---|---:|
| 模型 | `Qwen/Qwen3-8B` |
| 冷运行逻辑 LLM 调用 | **4** |
| 冷运行真实网络 API 调用 | **4 / 4 成功** |
| 冷运行耗时 | **423.31 s** |
| 第二次运行缓存命中 | **4** |
| 第二次新增网络调用 | **0** |
| 第二次运行耗时 | **0.27 s** |
| SiliconFlow Gold Micro F1 | **1.000** |
| SiliconFlow Gold Macro F1 | **1.000** |
| Exact case accuracy | **1.000** |

冷运行时间受外部模型排队、推理和网络影响，这也是本项目坚持 **Rule first → LLM only when needed → persistent cache** 的原因。高置信规则字段不会因为启用了 LLM 就被随意覆盖；本次验收中 `project_name`、`project_number`、`tenderer` 均被完整保留，`bid_date` 的最终主证据由 LLM 复核得到。

如果你只需要高速批处理：

```bash
uv run tender-extract extract ./documents --llm none --out out
```

---

## 🔎 输出不是只有 value：可以回到原文

简化后的真实输出结构类似：

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
            "section_path": [],
            "source_text": "合肥市公安局瑶海分局雪亮工程支网一期、二期、三期运维服务采购项目"
          }
        }
      ]
    }
  }
}
```

PDF 文本能够可靠定位时，`location` 还会包含真实 `bbox`；Markdown/TXT 没有物理页面概念时不会伪造页码和坐标，而是保留字符区间、行号和章节路径。

这意味着上层系统可以继续实现：

**字段 → 来源文件 → 页码 → bbox → PDF Viewer 高亮**

详细说明见 [`docs/evidence.md`](docs/evidence.md)。

---

## 🛠️ 快速开始

### 1. 安装

要求 Python **3.12+**。

```bash
git clone https://github.com/Inupedia/tender-extract.git
cd tender-extract

# 常用：包含 PDF 支持
uv sync --extra pdf

# 或安装全部可选能力（PDF / DOCX / OCR / test）
uv sync --extra all
```

### 2. 跑一份真实 PDF

```bash
uv run tender-extract extract examples/example.pdf --out out
```

输出：

```text
out/example.pdf.json
```

### 3. 批量处理目录

```bash
uv run tender-extract extract examples/ --pattern "*.pdf" --llm none --out out
```

### 4. 查看支持的 LLM

```bash
uv run tender-extract providers
```

除 SiliconFlow 外，Provider Registry 还包含 OpenAI、Azure OpenAI、Anthropic、Gemini、DeepSeek、DashScope/Qwen、Kimi、智谱、火山方舟、Ollama、OpenRouter、Groq、Together、Mistral、xAI 等，以及任意 OpenAI-compatible endpoint。

---

## 📦 一个项目不止一份文件：Tender Package

现实里的招投标项目通常不是一个 PDF：

```text
某项目/
├── 招标文件.pdf
├── 补遗01.pdf
├── 澄清01.pdf
├── A公司投标文件.pdf
└── B公司投标文件.pdf
```

`tender-package` 对这些文件建立明确的项目级语义：

- `tender / amendment / clarification / bid / attachment / other`
- `revision`
- `supersedes`
- 招标侧 effective view
- 旧值与冲突保留
- 投标文件按 bidder 严格隔离
- A 投标人的字段不会覆盖 B 投标人

```bash
uv run tender-package validate package.yaml
uv run tender-package inspect package.yaml
uv run tender-package extract package.yaml --out out/package.json
```

示例 manifest：

```yaml
package_id: demo-water-project
project_name: 某水利工程

documents:
  - id: tender-v1
    path: ./招标文件.pdf
    role: tender
    revision: 1

  - id: amendment-1
    path: ./补遗01.pdf
    role: amendment
    revision: 1
    supersedes: []

  - id: bidder-a
    path: ./A公司投标文件.pdf
    role: bid
    bidder: A公司
    revision: 1
```

详细说明见 [`docs/tender-package.md`](docs/tender-package.md)。

---

## 👀 Human Review：把人工修正变成下一轮测试数据

低置信、冲突或 LLM 恢复字段可以进入本地 review queue：

```bash
uv run tender-review run examples/example.pdf --queue .review/queue.jsonl
uv run tender-review list --queue .review/queue.jsonl
```

人工可以：

```bash
uv run tender-review resolve REVIEW_ID --action accept
uv run tender-review resolve REVIEW_ID --action correct --value "正确值"
uv run tender-review resolve REVIEW_ID --action reject
```

处理完成后直接回流成 Gold Dataset：

```bash
uv run tender-review export \
  --queue .review/queue.jsonl \
  --out eval/gold-reviewed.jsonl
```

所以流程不是“模型抽错 → 人改一下 → 下次继续错”，而是：

**抽取 → 待复核 → 人工决策 → Gold Dataset → Regression / F1 Gate**

详细说明见 [`docs/human-review.md`](docs/human-review.md)。

---

## 📈 用 F1 验证质量

```bash
uv run tender-extract eval eval/gold.jsonl
```

CI 可以设置质量底线：

```bash
uv run tender-extract eval eval/gold.jsonl --fail-under 0.95
```

也可以评测真实 LLM：

```bash
export SILICONFLOW_API_KEY=your-key

uv run tender-extract eval eval/gold-siliconflow.jsonl \
  --llm siliconflow \
  --model Qwen/Qwen3-8B \
  --fail-under 1.0
```

输出包含 per-field Precision / Recall / F1、Micro F1、Macro F1、exact case accuracy、失败字段和 LLM 调用次数。

详见 [`docs/evaluation.md`](docs/evaluation.md)。

---

## 📚 `examples/` 真实验收语料

| 文件 | 地区 | 页数 |
|---|---|---:|
| `example.pdf` | 安徽 | 10 |
| `public-01-beijing-landscape-lighting-2024.pdf` | 北京 | 89 |
| `public-02-shaanxi-xianyang-water-smart-2026.pdf` | 陕西 | 146 |
| `public-03-henan-tanghe-2026.pdf` | 河南 | 64 |
| `public-04-shanghai-single-source-2026.pdf` | 上海 | 37 |
| `public-05-shanghai-open-bid-20312273.pdf` | 上海 | 56 |
| `public-06-henan-zhengzhou-2026-312.pdf` | 河南 | 69 |
| `public-07-shanghai-digital-ops-2026.pdf` | 上海 | 119 |
| `public-08-shaanxi-baoji-books-2025.pdf` | 陕西 | 80 |
| `public-09-henan-shangqiu-2026.pdf` | 河南 | 58 |
| `public-10-shanghai-exam-equipment-2026.pdf` | 上海 | 75 |
| `public-11-shanghai-population-service-2026.pdf` | 上海 | 55 |
| `public-12-shanghai-xuhui-municipal-2026.pdf` | 上海 | 53 |

12 份新增公开文件的官方来源 URL、原始文件大小、页数和 SHA256 记录在 [`examples/public-corpus.lock.json`](examples/public-corpus.lock.json)；候选来源清单在 [`examples/public-sources.json`](examples/public-sources.json)。正常 CI **不访问这些政府网站**，直接使用仓库内已锁定的 PDF，因此验收可重复。

复现实验：

```bash
uv run python scripts/acceptance_corpus.py \
  --examples examples \
  --min-pdfs 13 \
  --report artifacts/real-pdf-acceptance.json
```

---

## ⚙️ 工作原理

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

设计原则很简单：**确定的事情不要花模型的钱，不确定的事情不要假装规则一定正确。**

---

## ✨ 主要能力

- PDF / DOCX / TXT / Markdown 统一解析
- PyMuPDF 文本页解析，扫描页可选 PaddleOCR
- 章节切块 + 模块化路由
- 中文项目名、项目编号、采购人、报价、日期、联系方式等规则抽取
- 人员、证书、身份证等专项抽取（PII 默认脱敏）
- RapidFuzz + MinHash 去重
- 低置信 / conflict / missing-field LLM routing
- LLM persistent cache
- structured evidence：page / line / section / bbox / source span
- Gold Dataset + Precision / Recall / F1
- Human Review → Gold feedback loop
- Tender Package 多文件版本和投标人隔离

---

## 🧪 测试与 CI

项目 CI 分层执行：

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

其中 Real PDF Acceptance 完全离线；SiliconFlow Live 使用 GitHub Actions secret，不会把 API key 写入仓库或 artifact。

本地：

```bash
uv sync --extra dev --extra pdf
uv run pytest -q
```

---

## ⚠️ 边界说明

- **34.62 pages/s 不包含 OCR 和网络 LLM 时间。** 扫描件 OCR 会明显更慢。
- LLM 延迟取决于模型、服务商、排队和网络；建议保留缓存，并让规则层先处理明确字段。
- PDF 可以提供真实物理页和 bbox；DOCX/Markdown/TXT 无法可靠推断物理分页时不会伪造坐标。
- 当前抽取规则针对中文招投标/政府采购文本进行了较多适配，但行业、地区和模板差异仍然需要 Gold Dataset 持续校准。

---

## 📄 License

MIT License
