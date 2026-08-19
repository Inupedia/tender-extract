# 标书信息提取工具

[English](README_EN.md) | 中文

## 📖 项目介绍

### 项目背景

在招投标行业中，标书文档通常包含数百页甚至上千页的复杂信息。传统人工提取方式存在效率低下、成本高昂、准确性不足等问题。

### 项目目的

`tender-extract` 通过**混合抽取技术**（规则引擎 + 大语言模型）实现智能化标书信息提取：

- **自动化提取**：从大量文档中自动识别关键字段
- **成本优化**：规则层覆盖60-90%字段，大幅降低大语言模型调用成本
- **高精度保证**：结合确定性规则和智能推理
- **可审计追溯**：保留原文证据，支持结果验证
- **标准化输出**：统一结构化数据格式

> 面向**千页**级别中文标书的**混合抽取**流水线：先用**规则/词典/命名实体识别**处理确定性字段，仅将**低置信/冲突**片段路由给**大语言模型**，在保证可审计的同时显著降本增效。

## 🚀 核心优势

- **高效处理**：5个文档仅需2.31秒，平均每个文档0.46秒
- **成本控制**：规则层覆盖60-90%硬字段，大语言模型调用次数大幅降低
- **可审计性**：每个抽取结果都保留原文证据片段
- **详细监控**：实时显示处理进度，便于调试

## ✨ 功能特性

- **多格式文档解析**：直接输入 PDF/DOCX/TXT/Markdown，自动转换为统一中间格式
- **模块化路由**：按章节内容路由到9个专业模块（基础信息/财务/资质/评标等）
- **增强正则引擎**：分层置信度模式库，覆盖中文金额/日期/证照/联系方式等
- **高吞吐规则层**：正则表达式+Aho–Corasick关键词+NER多方法竞争
- **智能去重**：RapidFuzz+MinHash局部敏感哈希，避免重复处理
- **按需大语言模型**：仅低置信/冲突时路由最小证据片段；支持 OpenAI / Azure / Anthropic / Gemini / Ollama / DeepSeek / 通义 / Kimi / 智谱 等主流厂商
- **结构化输出**：Pydantic数据校验，保留证据片段便于审计
- **表格感知**：从 Markdown 表格中提取金额/日期/人员信息

### v0.2.0 新增功能

| 功能 | 说明 |
|------|------|
| PDF/DOCX 解析 | 通过 PyMuPDF/python-docx 解析，基于字体大小推断标题，提取表格 |
| 扫描件 OCR | PaddleOCR 智能识别扫描页（自动检测文字页 vs 扫描页） |
| 模块化路由 | 不全文直抽，按标题切块后路由到专业模块，每个模块独立定义输出 Schema |
| 增强正则模式库 | 50+ 精确模式，分层置信度(0.95/0.8/0.6)，大写金额、统一社会信用代码等 |
| 人员信息抽取 | 身份证号、姓名+职务、学历、专业证书编号、有效期（支持表格和散落文本） |
| 多方法竞争 | 同一字段多种抽取方式并行，按置信度选择最佳结果 |
| 跨字段验证 | 保证金不超过投标金额等逻辑校验 |
| 真实示例文档 | 包含公开的政府采购招标文件 PDF 作为示例 |

## 📊 性能表现

### 真实招标文件抽取效果

以公开的《合肥市公安局瑶海分局雪亮工程支网运维服务采购项目》招标文件（PDF，10页）为例：

```bash
$ uv run tender-extract extract ./examples/example.pdf --out ./out

═══ 处理文件 1/1: example.pdf ═══
  ✓ 解析完成: pdf → Markdown (8742 字符, 10 页)
  ✓ 切块完成: 3 个切片
  ✓ 模块路由: [投标递交]: 3, [投标人信息]: 2, [评标办法]: 1
  ✓ 增强正则抽取...

  ✅ 完成: 8 个字段, 平均置信度 0.90, 耗时 0.21s
     project_name: 合肥市公安局瑶海分局雪亮工程支网 (置信度=0.95)
     tenderer: 合肥市公安局瑶海分局 (置信度=0.95)
     bid_amount: 437.677万元 (置信度=0.90)
     bid_date: 2024年7月1日 (置信度=0.90)
     project_number: 2024BFFFZ01583 (置信度=0.95)
     contact_info: 0551-66223642 (置信度=0.95)
```

**核心指标**：
- ⚡ 处理速度：10页PDF仅需 **0.21秒**
- 🎯 平均置信度：**0.90**
- 💰 零LLM成本：规则层覆盖所有字段
- 📄 支持格式：PDF / DOCX / TXT / Markdown

**抽取统计**：
- 26种字段类型，平均每文档24.4个字段
- 高频：项目名称、投标人、联系方式、日期
- 中频：经营范围、投标金额、营业执照
- 低频：注册资本、股东信息、项目经理

### Markdown 示例

仓库还提供一份结构完整的 Markdown 样例，适合快速验证安装：

```bash
uv run tender-extract extract ./examples/example.md --out ./out
```

`examples/` 目录：

| 文件 | 说明 |
|------|------|
| `examples/example.md` | 合成标书，覆盖项目名称/投标人/金额/保证金/人员 |
| `examples/example.pdf` | 真实政府采购招标文件（合肥市雪亮工程项目，10页） |

### 人员信息抽取效果

自动从标书中提取分散在不同位置的人员信息（表格 + 正文 + 扫描件 OCR）：

```
找到 3 人:
1. 张建国 (项目经理) 身份证:420102****2718
2. 李明辉 (技术负责人) 身份证:320106****0035 硕士/结构工程
3. 王晓东 (安全员) 身份证:510103****3456

找到 4 个证书:
  [建造师] 鄂142011203456
  [工程师职称] 苏高工2019-03456
  [安全B证] 鄂建安B(2021)0045678
  [有效期] 2027年12月31日
```

**支持的人员信息类型**：
- 📋 身份证号码（18位/15位，含校验位验证）
- 👤 姓名 + 职务角色（项目经理/技术负责人/安全员等）
- 🎓 学历、毕业院校、专业
- 📜 建造师/工程师/安全员等证书编号
- 📅 证书有效期
- 🏢 从资质表格中批量提取

**OCR 策略**（处理扫描件）：
- 逐页智能检测：文字页用 PyMuPDF（快），扫描页用 PaddleOCR（准）
- 支持身份证、毕业证、资格证书扫描件的文字识别

身份证号默认脱敏写入 JSON；需要完整号码时加 `--include-pii`。

---

## 🛠️ 快速开始

### 安装

```bash
# 克隆并安装（推荐：安装所有可选依赖）
git clone <repository-url>
cd tender-extract
uv sync --extra all

# 或最小安装
uv sync

# 可选：单独安装
uv sync --extra pdf    # PDF 解析 (pymupdf)
uv sync --extra docx   # DOCX 解析 (python-docx)
uv sync --extra ocr    # 扫描件 OCR (paddlepaddle + paddleocr)

# 验证安装
uv run tender-extract --help
```

### 基础用法

```bash
# 推荐：支持 PDF/DOCX/TXT/MD，模块化路由 + 增强正则
uv run tender-extract extract ./examples/example.pdf --out ./out
uv run tender-extract extract ./examples/ --out ./out --verbose

# 仅规则抽取（不调用大模型）
uv run tender-extract extract ./examples/ --out ./out --llm none

# 启用大语言模型（需要对应厂商的 API 密钥）
export OPENAI_API_KEY=your-api-key
uv run tender-extract extract ./examples/ --out ./out --llm openai --model gpt-4o-mini

export DEEPSEEK_API_KEY=your-api-key
uv run tender-extract extract ./examples/ --out ./out --llm deepseek

# 本地 Ollama
uv run tender-extract extract ./examples/ --out ./out --llm ollama --model deepseek-r1:32b

# 查看全部厂商
uv run tender-extract providers
```

`extract-v2` 仍可用，是 `extract` 的兼容别名。

### 主要参数

- `input_path`：输入文件或目录
- `--out`：输出目录（默认./out）
- `--llm`：`none` 或厂商 ID（`openai` / `ollama` / `anthropic` / `deepseek` / `qwen` 等，见 `providers`）
- `--model`：模型名称（Azure 填部署名）
- `--base-url`：覆盖 API 地址
- `--use-ner`：启用中文命名实体识别
- `--include-pii`：输出完整身份证号（默认掩码）
- `--verbose`：显示详细进度
- `--debug`：大语言模型调试模式

---

## 📂 项目结构

```
tender-extract/
├── config/example.yaml           # 规则配置（正则模式、同义词词典）
├── data/dicts/keywords_zh.txt    # 关键词词典
├── examples/
│   ├── example.md                # Markdown 示例文件
│   └── example.pdf               # 真实招标文件（合肥市政府采购项目）
└── src/tender_extract/
    ├── cli.py                    # 命令行接口（extract / extract-v2）
    ├── pipeline.py               # 统一抽取流水线
    ├── document_parser.py        # 文档解析层（PDF/DOCX → Markdown + OCR）
    ├── module_router.py          # 模块化路由层（9个专业模块）
    ├── extraction_engine.py      # 增强抽取引擎（多方法竞争+置信度）
    ├── patterns.py               # 正则模式库（分层置信度）
    ├── personnel_extractor.py    # 人员信息专项抽取（身份证/证书/学历）
    ├── preprocess.py             # Markdown 预处理 + 章节树构建
    ├── chunker.py                # 智能切块（LangChain 分割器）
    ├── rules.py                  # 规则抽取（正则+Aho-Corasick关键词）
    ├── ner.py                    # 中文 NER（jieba）
    ├── dedupe.py                 # 去重引擎（RapidFuzz + MinHash LSH）
    ├── llm_providers.py          # 主流 LLM 厂商预设
    ├── llm_router.py             # LLM 按需路由
    ├── merge.py                  # 结果合并与冲突解决
    └── schema.py                 # Pydantic 数据模型
```

---

## ⚙️ 配置

### 规则配置

编辑 `config/example.yaml`：

```yaml
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

---

## 🔍 工作原理

```
输入(PDF/DOCX/MD) → ① 解析层 → ② 切块层 → ③ 模块路由 → ④ 抽取层 → ⑤ 合并输出
                     (PyMuPDF)  (LangChain)  (关键词匹配)  (Regex+NER+LLM)  (JSON)
```

1. **解析层**：PDF/DOCX/TXT 统一转 Markdown，保留标题层级和表格
2. **切块层**：按章节切分，LangChain递归字符切片
3. **模块路由**：按关键词将切片路由到9个专业模块（基础信息/财务/资质/评标等）
4. **抽取层**：增强正则（分层置信度）+ NER + 按需LLM
5. **合并输出**：去重、冲突检测、跨字段验证

---

## 📊 输出格式

```json
{
  "metadata": {
    "filename": "example.md",
    "processing_time": 2.31,
    "total_fields": 24
  },
  "fields": {
    "project_name": {
      "primary_value": "测试工程项目",
      "confidence": 0.95,
      "values": [{
        "value": "测试工程项目",
        "source": "rules",
        "start": 100,
        "end": 110
      }]
    }
  }
}
```

---

## 🎯 适用场景

- **招标代理**：批量处理投标文件
- **评标专家**：快速获取标书核心信息
- **监管部门**：自动化合规审核
- **研究机构**：标书数据分析
- **企业投标**：竞争对手分析

---

## 🐛 故障排除

### 常见问题

```bash
# 安装失败
python --version  # 确保3.12+
uv sync --reinstall

# Ollama连接失败
curl http://your-ollama-server:11434/api/tags
export OLLAMA_BASE_URL=http://your-ollama-server:11434

# 调试技巧
uv run tender-extract extract ./examples/ --out ./out --verbose --debug
```

---

## 📝 许可证

MIT许可证 - 详见 [LICENSE](LICENSE) 文件。
