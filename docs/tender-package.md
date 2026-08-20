# Multi-file Tender Package

`tender-package` 把单文件抽取提升为项目级文档包。一个 package 可以同时包含招标文件、补遗、澄清、投标文件和附件，并保留版本关系与来源。

## Manifest

```yaml
package_id: qingjiang-2026-001
project_name: 青江水库除险加固工程

documents:
  - id: tender-v1
    path: 招标文件.md
    role: tender

  - id: amendment-1
    path: 补遗01.md
    role: amendment

  - id: clarification-v1
    path: 澄清01.md
    role: clarification
    logical_name: clarification
    revision: 1

  - id: clarification-v2
    path: 澄清02.md
    role: clarification
    logical_name: clarification
    revision: 2

  - id: bidder-a
    path: A公司投标文件.md
    role: bid
    bidder: A公司

  - id: bidder-b
    path: B公司投标文件.md
    role: bid
    bidder: B公司
```

相对路径以 manifest 所在目录为基准。

## Roles

- `tender`: 招标文件
- `amendment`: 补遗 / 正式修改文件
- `clarification`: 澄清文件
- `bid`: 投标文件；必须提供 `bidder`
- `attachment`: 附件；有 `bidder` 时归入对应投标人，否则归入招标侧
- `other`: 其它项目文件

## Revision semantics

### `logical_name` + `revision`

同一业务域、同一 role、同一 `logical_name` 的多份文件被视为一个版本序列。最高 revision 为 active，其余版本保留在结果中但标记为 `newer_revision`。

```yaml
- id: clarification-v1
  role: clarification
  logical_name: clarification
  revision: 1

- id: clarification-v2
  role: clarification
  logical_name: clarification
  revision: 2
```

### `supersedes`

用于显式表示整份文件已经被另一份文件替代：

```yaml
- id: bid-a-v2
  role: bid
  bidder: A公司
  path: A公司投标文件-v2.md
  supersedes: [bid-a-v1]
```

`supersedes` 会检查：

- 引用必须存在
- 不能引用自身
- 不允许循环替代
- 不允许跨 scope 替代，例如 A 公司文件替代 B 公司文件

补遗/澄清通常只是修改部分条款，因此**不要**仅因为它是补遗就把整份招标文件写进 `supersedes`。在不显式 supersede 的情况下，两者同时 active，项目级字段会通过 role priority 形成 effective view。

## Effective views

输出包含三个层级：

1. `documents`: 每份文件的完整 ExtractionResult，包含 inactive 历史版本，便于审计。
2. `tender_fields`: 招标侧项目视图。字段来源优先级为 amendment > clarification > tender > attachment/other。
3. `bidder_fields`: 按投标人隔离的视图。A 公司的字段不会参与 B 公司的字段选择。

每个 effective field 都保留：

- `primary_value`
- `selected_from_document`
- `values`
- `conflicts`
- `sources`

因此上层 UI 可以显示“当前有效值是什么、来自哪份文件、旧值/冲突值是什么”。

## Commands

校验项目包：

```bash
tender-package validate ./package.yaml
```

查看 active / inactive 文件：

```bash
tender-package inspect ./package.yaml
```

执行整个项目包抽取：

```bash
tender-package extract ./package.yaml \
  --config ./config/example.yaml \
  --out ./out/package.json
```

使用 SiliconFlow：

```bash
export SILICONFLOW_API_KEY=...

tender-package extract ./package.yaml \
  --llm siliconflow \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out ./out/package.json
```

API key 不应写入 manifest 或提交到仓库。

## Why this matters

后续的页码/BBox 证据定位、Requirement/Compliance Matrix 和项目级风险分析都需要稳定的“项目、文件、版本、投标人、有效值来源”关系。Tender Package 是这些能力的基础数据模型，而不是简单的批量文件处理器。
