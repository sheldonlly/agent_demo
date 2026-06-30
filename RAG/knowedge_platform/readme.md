```
Knowledge Platform（知识平台）
   ├── Connector
   ├── Document Manager
   ├── Parser
   ├── Chunk Manager
   ├── Metadata Manager
   └── Version & ACL
```

# Knowledge Platform

> Knowledge Platform 是整个企业级 RAG 系统的数据中台（Data Platform），负责企业知识的采集、治理、标准化处理以及生命周期管理，为后续的 Index Platform 提供高质量的数据来源。

---

# 整体架构

```text
                  Enterprise Data Source
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
     GitHub           Confluence          Local File
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    Connector Service
                           │
                           ▼
                  Document Manager
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Version Manager   ACL Manager      Metadata Manager
                           │
                           ▼
                        Parser
                           │
                           ▼
                     Chunk Manager
                           │
                           ▼
                  Publish Knowledge Event
                           │
                           ▼
                     Index Platform
```

---

# 核心职责

Knowledge Platform 包括六个核心模块：

| 模块 | 主要职责 |
|------|----------|
| Connector | 从各种数据源同步文档 |
| Document Manager | 管理文档生命周期 |
| Parser | 将不同格式文档解析为统一结构 |
| Chunk Manager | 将文档切分为知识单元 |
| Metadata Manager | 管理文档及 Chunk 元数据 |
| Version & ACL | 管理文档版本和访问权限 |

---

# 1. Connector（数据连接器）

## 职责

Connector 是整个知识平台的数据入口。

它负责从各种外部系统中获取文档，并将文档变化同步到 Knowledge Platform。

Connector 不负责：

- 文档解析
- Chunk 切分
- Embedding
- 检索

它只负责：

> **发现文档发生了变化。**

包括：

- 新增文档
- 修改文档
- 删除文档
- 权限变化

---

## 为什么需要 Connector？

企业中的知识通常分散在多个系统中，例如：

- GitHub
- Confluence
- Notion
- SharePoint
- 飞书文档
- 企业 Wiki
- OSS
- NAS
- MySQL
- PostgreSQL
- 本地文件系统

每种系统：

- API 不同
- 数据格式不同
- 权限体系不同
- 更新方式不同

Connector 的作用就是屏蔽这些差异，为平台提供统一的数据同步接口。

---

## 主要功能

- 全量同步
- 增量同步
- Webhook 监听
- 文件监听
- 权限同步
- 删除同步
- 数据源认证
- Connector 插件管理

---

## 输出

Connector 最终输出统一的 Document Event，例如：

```text
DocumentCreated

DocumentUpdated

DocumentDeleted

PermissionChanged
```

这些事件将交由 Document Manager 处理。

---

# 2. Document Manager（文档管理）

## 职责

Document Manager 是整个 Knowledge Platform 的核心。

它负责维护文档的完整生命周期。

包括：

- 文档注册
- 文档状态
- 文档唯一标识
- Hash 管理
- 生命周期管理

---

## 为什么需要？

假设上传：

```
员工手册.pdf
```

第二天再次上传：

```
员工手册.pdf
```

Document Manager 会：

```
计算 Hash

↓

判断是否发生变化

↓

没有变化

↓

跳过后续流程
```

避免重复解析和重复 Embedding。

---

## 生命周期

一个文档通常具有如下状态：

```text
NEW

↓

PARSING

↓

READY

↓

INDEXING

↓

ACTIVE

↓

ARCHIVED

↓

DELETED
```

Document Manager 负责维护这些状态。

---

## 管理内容

例如：

```text
Document

id

source

name

hash

version

status

owner

created_at

updated_at
```

---

# 3. Parser（文档解析）

## 职责

Parser 的作用是：

> **将各种不同格式的文档统一解析为标准 Document 对象。**

例如：

PDF

↓

Markdown

↓

统一 Document Model

---

## 支持格式

例如：

- PDF
- DOCX
- PPTX
- Excel
- Markdown
- HTML
- CSV
- 图片（OCR）
- XML

---

## 解析内容

Parser 不仅解析正文，还需要解析：

- 标题
- 段落
- 表格
- 图片
- 代码块
- 公式
- 引用

最终输出统一的数据结构。

例如：

```text
Document

├── Paragraph
├── Table
├── Image
├── Code
└── Reference
```

这样后续模块无需关心文档来源。

---

## 技术难点

例如：

- OCR
- 表格识别
- 页面布局分析
- Markdown 转换
- 多语言解析

---

# 4. Chunk Manager（知识切分）

## 职责

Chunk Manager 负责将一个完整文档切分为适合向量检索的知识单元（Chunk）。

它决定了：

> RAG 检索效果。

---

## 输入

Parser 输出：

```
Document
```

---

## 输出

多个 Chunk：

```text
Chunk1

Chunk2

Chunk3
```

每个 Chunk 包含：

```text
chunk_id

doc_id

content

chunk_hash

metadata
```

---

## 支持策略

企业一般不会只支持固定长度切分。

通常支持：

- Fixed Chunk
- Recursive Chunk
- Semantic Chunk
- Hierarchical Chunk
- Parent-Child Chunk
- Sliding Window

不同策略适用于不同业务场景。

---

## 技术难点

包括：

- Chunk 大小选择
- Overlap 控制
- Token 长度控制
- 章节边界识别
- 图片与正文关联

---

# 5. Metadata Manager（元数据管理）

## 职责

Metadata Manager 负责维护文档及 Chunk 的各种元信息。

Metadata 不参与 Embedding，但会影响检索质量。

---

## Metadata 来源

可能来自：

Parser：

```
标题

目录

作者
```

Connector：

```
来源

Owner
```

用户：

```
标签

分类
```

AI 自动生成：

```
Summary

Keywords

Topic
```

---

## Metadata 示例

```json
{
  "department":"AI",
  "language":"zh",
  "project":"Enterprise-RAG",
  "author":"Alice",
  "tag":["RAG","Agent"],
  "version":"v3"
}
```

---

## Metadata 用途

主要用于：

- Filter Search
- 权限过滤
- 标签搜索
- 分类统计
- 多租户隔离
- 数据分析

---

# 6. Version & ACL（版本与权限）

## Version（版本管理）

企业文档通常会不断更新。

例如：

```
员工手册

↓

v1

↓

v2

↓

v3
```

Knowledge Platform 不会直接覆盖旧版本。

而是：

```
v1

inactive

↓

v2

inactive

↓

v3

active
```

这样可以支持：

- 回滚
- 历史查询
- 审计

---

## ACL（Access Control List）

ACL 用于控制：

> 谁可以访问哪些知识。

例如：

```
HR

↓

只能访问 HR 文档
```

Finance：

```
不能检索 HR 数据
```

ACL 通常支持：

- User
- Role
- Department
- Tenant
- Group

例如：

```json
{
  "tenant":"companyA",
  "department":"AI",
  "role":["Developer"]
}
```

检索阶段会根据 ACL 自动过滤文档。

---

# 六个模块之间的关系

```text
Connector
      │
      ▼
Document Manager
      │
      ├───────────────┐
      │               │
      ▼               ▼
Version         Permission(ACL)
      │               │
      └───────┬───────┘
              ▼
           Parser
              │
              ▼
      Metadata Manager
              │
              ▼
        Chunk Manager
              │
              ▼
     Publish Knowledge Event
              │
              ▼
         Index Platform
```

---

# 总结

Knowledge Platform 的目标不是完成检索，而是将企业中分散、异构、不断变化的知识，转换为统一、规范、可治理的知识资产。

各模块职责如下：

| 模块 | 输入 | 输出 | 主要职责 |
|------|------|------|----------|
| Connector | 外部数据源 | Document Event | 同步企业知识，监控文档变化 |
| Document Manager | Document Event | Document | 管理文档生命周期、状态和唯一标识 |
| Parser | 原始文档 | Structured Document | 将不同格式解析为统一文档模型 |
| Chunk Manager | Structured Document | Chunk | 按策略切分知识单元，为索引做准备 |
| Metadata Manager | Document / Chunk | Metadata | 管理标签、作者、语言、部门等元信息 |
| Version & ACL | Document | Version / Permission | 管理文档版本、历史记录及访问控制 |