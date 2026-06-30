```
Index Platform（索引平台）
   ├── Embedding Service
   ├── Chunk Diff
   ├── Vector Index
   ├── BM25 Index
   └── Cache
```

# Index Platform

> Index Platform 是企业级 RAG 系统中的索引构建中心（Index Infrastructure），负责将 Knowledge Platform 输出的标准化知识转换为可检索的索引，并维护索引的生命周期。

Index Platform 不负责：

- 文档采集
- 文档解析
- LLM 调用
- Prompt 构建

它只负责：

> **将知识高效、准确地转换为检索引擎可使用的索引。**

---

# 整体架构

```text
                 Knowledge Platform
                        │
                        ▼
              Structured Document
                        │
                        ▼
                  Chunk Manager
                        │
                        ▼
                  Chunk Diff
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
Embedding Service   BM25 Index   Cache Manager
         │              │
         └──────┬───────┘
                ▼
          Vector Index
                │
                ▼
        Publish Index Event
                │
                ▼
        Retrieval Platform
```

---

# 核心职责

Index Platform 包括五个核心模块：

| 模块 | 主要职责 |
|------|----------|
| Embedding Service | 将 Chunk 转换为向量表示 |
| Chunk Diff | 检测 Chunk 变化，实现增量更新 |
| Vector Index | 构建和维护向量索引 |
| BM25 Index | 构建关键词倒排索引 |
| Cache | 缓存 Embedding、索引和热点数据 |

---

# 1. Embedding Service（向量化服务）

## 职责

Embedding Service 的作用是：

> **将文本 Chunk 转换为高维向量（Embedding），供向量检索使用。**

输入：

```
Chunk
```

输出：

```
Embedding Vector
```

例如：

```
"什么是 RAG？"

↓

[-0.14,0.32,...]
```

---

## 为什么需要？

LLM 无法直接进行语义检索。

Embedding 将自然语言映射到向量空间，使：

- 相似语义距离更近
- 不同语义距离更远

从而支持 ANN（Approximate Nearest Neighbor）搜索。

---

## 主要功能

### 多模型支持

支持：

- BGE
- GTE
- Qwen Embedding
- OpenAI Embedding
- Voyage
- Jina Embedding

---

### 批量 Embedding

避免：

```
一个 Chunk

↓

一次 API
```

而是：

```
100 Chunk

↓

Batch Embedding
```

提高吞吐量。

---

### Embedding Pipeline

```
Chunk

↓

Normalize

↓

Embedding

↓

Vector

↓

Store
```

---

## 技术难点

例如：

- Batch 大小控制
- GPU 调度
- Token 长度限制
- 模型切换
- 多语言支持
- Embedding 一致性

---

# 2. Chunk Diff（增量更新）

## 职责

Chunk Diff 用于：

> **比较文档更新前后的 Chunk，只重新索引发生变化的部分。**

它是企业级 RAG 最重要的优化之一。

---

## 为什么需要？

例如：

一份文档：

```
1000 Chunk
```

今天：

只修改：

```
Chunk 233
Chunk 876
```

如果全部重新 Embedding：

```
1000 次
```

成本极高。

Chunk Diff：

只处理：

```
Chunk233

Chunk876
```

即可。

---

## 工作流程

```
Old Chunk

↓

Hash

↓

New Chunk

↓

Compare

↓

新增

更新

删除
```

---

## 输出

最终生成：

```
Need Insert

Need Update

Need Delete
```

交由：

Embedding Service

和：

Index Builder

处理。

---

## 技术难点

包括：

- Chunk Hash 设计
- Chunk 对齐
- Chunk Merge
- Chunk Split
- 大文档 Diff
- 删除检测

---

# 3. Vector Index（向量索引）

## 职责

Vector Index 用于：

> **构建和维护向量数据库中的 ANN 索引。**

Embedding 只是生成向量。

真正提供语义检索的是：

Vector Index。

---

## 支持数据库

例如：

- Qdrant
- Milvus
- Weaviate
- Pinecone
- pgvector

---

## Index 内容

通常包括：

```
Embedding

Metadata

Chunk ID

Document ID

Version

ACL
```

例如：

```json
{
  "chunk_id":"001",
  "doc_id":"100",
  "vector":[...],
  "metadata":{
      "department":"AI",
      "version":"v3"
  }
}
```

---

## 主要功能

### 插入

新增向量。

---

### 更新

更新已有向量。

---

### 删除

文档删除：

自动删除对应向量。

---

### Filter

支持：

```
Department

Language

Version

Tenant
```

进行 Metadata 检索。

---

## 技术难点

包括：

- HNSW 参数优化
- ANN 精度
- Metadata Filter
- 百万级向量管理
- 多租户隔离
- 分片与副本

---

# 4. BM25 Index（关键词索引）

## 职责

BM25 Index 用于：

> **构建倒排索引，实现关键词检索。**

企业 RAG：

几乎不会：

只使用向量检索。

通常都会：

```
Vector

+

BM25
```

形成：

Hybrid Search。

---

## 为什么需要？

例如：

用户搜索：

```
HTTP 500
```

Embedding：

可能效果不好。

但是：

BM25：

关键词：

```
HTTP

500
```

可以直接命中。

---

## 常见实现

支持：

- Elasticsearch
- OpenSearch
- Lucene

---

## Index 内容

例如：

```
Token

↓

Posting List

↓

Chunk ID
```

---

## 技术难点

包括：

- 中文分词
- 多语言 Tokenizer
- 同义词扩展
- 停用词
- 增量更新
- 倒排索引压缩

---

# 5. Cache（缓存管理）

## 职责

Cache 用于：

> **减少重复计算，提高索引构建效率。**

它不是 Retrieval Cache。

而是：

Index Cache。

---

## Cache 内容

### Embedding Cache

例如：

相同 Chunk：

```
Hash 一样

↓

直接使用已有 Embedding
```

无需重新计算。

---

### Chunk Cache

保存：

```
Chunk Hash
```

用于：

Chunk Diff。

---

### Metadata Cache

缓存：

```
Document Metadata
```

减少数据库访问。

---

### Index Cache

缓存：

```
Index Config

ANN 参数

Collection Info
```

---

## 推荐组件

例如：

- Redis
- RocksDB
- Local Memory Cache

---

## 技术难点

包括：

- Cache 一致性
- Cache 失效策略
- 多节点同步
- 热点数据缓存
- LRU / LFU 策略

---

# 五个模块之间的关系

```text
Knowledge Platform
        │
        ▼
      Chunk
        │
        ▼
    Chunk Diff
        │
 ┌──────┴────────┐
 │               │
 ▼               ▼
Embedding    BM25 Builder
 │               │
 └──────┬────────┘
        ▼
  Vector Index
        │
        ▼
      Cache
        │
        ▼
Retrieval Platform
```

---

# 总结

Index Platform 的目标不是回答问题，而是**构建高质量、高性能、可持续维护的检索索引**。

各模块职责如下：

| 模块 | 输入 | 输出 | 主要职责 |
|------|------|------|----------|
| Embedding Service | Chunk | Embedding Vector | 将文本转换为语义向量，支持语义检索 |
| Chunk Diff | Old Chunk + New Chunk | Diff Result | 比较文档变化，实现增量索引更新 |
| Vector Index | Embedding | ANN Index | 构建和维护向量数据库索引，支持 Metadata Filter |
| BM25 Index | Chunk | Inverted Index | 构建关键词倒排索引，实现 Hybrid Search |
| Cache | Chunk / Embedding / Metadata | Cache Data | 缓存热点数据，减少重复计算，提高索引构建效率 |

---

# Index Platform 在整个企业级 RAG 中的位置

```text
Knowledge Platform
        │
        ▼
=============================
      Index Platform
=============================

Embedding Service
Chunk Diff
Vector Index
BM25 Index
Cache

        │
        ▼
Retrieval Platform
        │
        ▼
AI Platform
```

Index Platform 位于 Knowledge Platform 与 Retrieval Platform 之间，是连接知识治理与检索能力的核心基础设施。它负责将标准化知识加工为高质量索引，并通过增量更新、混合索引和缓存机制，保证企业级 RAG 在准确性、性能和成本之间取得平衡。