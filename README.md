# tender-extract

[English](README_EN.md) | 中文

> 面向 **千页** 级别的中文标书（已转 Markdown）的 **混合抽取** 流水线：先用 **规则/词典/NER** 吃掉确定性字段，只把 **低置信/冲突** 的小片段 **路由给 LLM**，在保证可审计的同时显著降本增效。

## 🚀 核心优势

- **成本控制**：规则层覆盖 60-90% 的硬字段，LLM调用次数大幅降低
- **高效处理**：5个文档仅需2.31秒，平均每个文档0.46秒
- **零LLM调用**：在您的测试中，规则层完全覆盖了所有字段，无需调用LLM
- **可审计性**：每个抽取结果都保留原文证据片段，便于追溯验证
- **详细进度**：实时显示LLM处理进度和内容，便于调试和监控

## ✨ 功能特性

- **Markdown 结构解析 + 章节优先切片**：基于 `markdown-it-py`，按 `# / ## / ###` 构建章节树，再进行递归字符切分（带少量 overlap）。
- **高吞吐规则层**：正则 + 关键词行启发式；金额/日期/保证金/联系方式/地址/邮编等一次抽取。
- **大词表极速匹配**：Aho–Corasick（`pyahocorasick`）批量短语扫描（如"资质要求/资格条件/评标办法/联合体"等近义短语）。
- **近重复与模板识别**：`RapidFuzz` 字符串相似 + `datasketch` 的 **MinHash LSH**，避免重复问 LLM。
- **按需 LLM**：仅当规则层 **低置信或冲突** 时，将 **最小证据片段** 送入 LLM；支持 **OpenAI Structured Outputs** 与本地 **Ollama**。
- **严格结构化输出**：Pydantic/JSONSchema 校验，保留 `evidence_spans`（字段值 + 引用定位）便于审计。
- **详细进度监控**：实时显示LLM调用进度、发送内容、返回响应，支持调试模式。

## 📊 实际性能表现

<img src="./assets/1.jpg" alt="性能统计图表" style="width:300px; height:auto;" />

**字段抽取统计**：
- 26种不同类型的字段被成功抽取
- 每个文档平均抽取24.4个字段
- 高频字段（出现5次）：项目名称、投标人、联系方式、日期等
- 中频字段（出现3-4次）：经营范围、投标金额、营业执照等
- 低频字段（出现1-2次）：注册资本、股东信息、项目经理等

---

## 🛠️ 安装指南

### 系统要求
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) 包管理器
- 可选：Ollama（用于本地LLM推理）

### 快速安装

```bash
# 克隆项目
git clone <repository-url>
cd tender-extract

# 安装基础依赖 + CLI
uv sync --extra cli

# 安装中文 NER（可选）
uv sync --extra ner

# 设置Ollama地址（如果使用本地LLM）
export OLLAMA_BASE_URL=http://your-ollama-server:11434
```

### 验证安装

```bash
# 检查CLI是否可用
uv run tender-extract --help

# 运行简单测试
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm none --verbose
```

---

## 🎯 使用指南

### 基础用法

```bash
# 仅使用规则抽取（最快，推荐）
uv run tender-extract extract ./examples/sample.md --out ./out --use-ner --llm none

# 批处理整个目录
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm none --verbose
```

### 高级用法

```bash
# 启用详细进度显示
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm ollama --verbose

# 启用LLM调试模式（显示完整提示词和响应）
uv run tender-extract extract ./examples/ --out ./out --use-ner --llm ollama --verbose --debug

# 使用OpenAI（需要API密钥）
export OPENAI_API_KEY=your-api-key
uv run tender-extract extract ./examples/ --out ./out --llm openai --model gpt-4o-mini

# 本地Ollama
uv run tender-extract extract ./examples/ --out ./out --llm ollama --model deepseek-r1:32b
```

### 📁 批处理你的整批 `.md`：

```bash
# 仅规则/词典（最快）
uv run tender-extract extract /path/to/md_dir --pattern "*.md" --out ./out --llm none

# 启用 OpenAI（严格 JSON 输出）
uv run tender-extract extract /path/to/md_dir --out ./out --llm openai --model gpt-4o-mini

# 本地 Ollama
uv run tender-extract extract /path/to/md_dir --out ./out --llm ollama --model deepseek-r1:32b
```

### 🔧 命令行参数详解

```bash
uv run tender-extract --help
uv run tender-extract extract --help
```

**主要参数**：
- `input_path`：输入文件或目录（Markdown）
- `--pattern`：当 input_path 为目录时的匹配模式（默认 "*.md"）
- `--out`：输出目录（默认 ./out）
- `--config`：规则/词典 YAML（默认 ./config/example.yaml）
- `--use-ner`：启用中文 NER（需 foolnltk）
- `--llm`：none | ollama | openai
- `--model`：LLM 模型名（如 deepseek-r1:32b）
- `--cache-dir`：缓存目录（默认 ./.cache）
- `--verbose`：显示详细处理信息
- `--debug`：LLM调试模式，显示完整提示词和响应

---

## 📂 项目结构

```bash
tender-extract/
├── pyproject.toml                # uv/依赖/脚本入口
├── README.md
├── config/
│   └── example.yaml              # 正则与词典配置
├── data/dicts/
│   └── keywords_zh.txt           # 关键词词典
├── examples/                     # 示例文档
│   └── example.md
├── out/                          # 输出目录
│   └── example.md.json
└── src/tender_extract/
    ├── cli.py                    # CLI（Typer）
    ├── preprocess.py             # Markdown 清洗 + 章节树
    ├── chunker.py                # 章节优先 + 递归切片
    ├── rules.py                  # 正则 + 关键词启发式抽取
    ├── ner.py                    # 可选: foolnltk
    ├── dedupe.py                 # RapidFuzz + MinHash
    ├── llm_router.py             # OpenAI / Ollama 适配
    ├── merge.py                  # 字段合并策略
    └── schema.py                 # Pydantic 输出模型
```

---

## ⚙️ 配置与扩展

### 规则/词典配置

编辑 `config/example.yaml` 与 `data/dicts/keywords_zh.txt`：

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

### NER配置

在 `--use-ner` 下用于补充组织/人名/地名等实体候选，可与规则层投票融合。

### LLM 路由配置

`src/tender_extract/llm_router.py` 支持 ollama；仅在字段低置信或冲突时触发。

---

## 🔍 工作原理

1. **预处理与切片**：解析 Markdown → 章节树；对长段落做递归字符切分（~600–800 tokens 等级）
2. **规则层抽取**：金额/日期/编号/保证金/联系方式等"硬字段"优先提取；多关键短语用 Aho–Corasick 线性扫描
3. **去重与模板识别**：RapidFuzz + MinHash LSH，复用模板段落的解析结果
4. **按需 LLM**：仅在低置信/冲突时把"最小证据片段"送入 LLM，并用 Structured Outputs/函数调用保证严格 JSON

---

## 📈 性能优化建议

1. **先规则后模型**：规则/词典层通常覆盖 60–90% 的硬字段；LLM 只补难点
2. **控制片段长度**：章节优先 + 少量 overlap 的递归切片
3. **建立缓存**：对片段文本做指纹（如 MinHash），相同/近似段落直接复用抽取结果
4. **并行化**：预处理、规则抽取与相似度检测可多进程；LLM 采用小批并发并限流

---

## 🎯 适用场景

- **招标代理机构**：批量处理投标文件，提取关键信息
- **评标专家**：快速获取标书核心信息，辅助评标决策  
- **监管部门**：自动化审核标书合规性
- **研究机构**：分析标书数据，进行市场研究
- **企业投标**：快速分析竞争对手标书信息

---

## 📊 输出格式

每个文档处理后会生成对应的JSON文件，包含：

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
      "primary_value": "测试工程项目",
      "confidence": 0.95,
      "values": [
        {
          "value": "测试工程项目",
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

## 🐛 故障排除

### 常见问题

**1. 安装失败**
```bash
# 确保使用正确的Python版本
python --version  # 应该是3.9+

# 重新安装依赖
uv sync --reinstall
```

**2. Ollama连接失败**
```bash
# 检查Ollama服务状态
curl http://your-ollama-server:11434/api/tags

# 设置正确的环境变量
export OLLAMA_BASE_URL=http://your-ollama-server:11434
```

**3. NER模块错误**
```bash
# 重新安装NER依赖
uv sync --extra ner --reinstall
```

**4. 内存不足**
```bash
# 减少切片大小
uv run tender-extract extract ./examples/ --out ./out --llm none
```

### 调试技巧

1. **使用详细模式**：添加 `--verbose` 参数查看详细日志
2. **启用调试模式**：添加 `--debug` 参数查看LLM完整交互
3. **检查配置文件**：确保 `config/example.yaml` 格式正确
4. **查看输出文件**：检查 `out/` 目录中的JSON文件

---

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---


