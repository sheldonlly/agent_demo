```
Retrieval Platform（检索平台）
   ├── Query Understanding
   ├── Hybrid Retrieval
   ├── Reranker
   ├── Context Builder
   └── Prompt Builder
```

# Retrieval Platform

> Retrieval Platform 是企业级 RAG 系统中的检索中心（Retrieval Infrastructure），负责理解用户问题、召回知识、排序结果、构建上下文，并生成最终发送给大语言模型（LLM）的 Prompt。

Retrieval Platform 不负责：

- 文档管理
- 文档解析
- Embedding
- 向量索引构建
- LLM 调用

它只负责：

> **根据用户问题，从海量知识中找到最相关的信息，并构建高质量上下文。**

---

# 整体架构

```text
                      User Question
                           │
                           ▼
                 Query Understanding
                           │
                           ▼
                  Hybrid Retrieval
                           │
                           ▼
                       Reranker
                           │
                           ▼
                  Context Builder
                           │
                           ▼
                   Prompt Builder
                           │
                           ▼
                      AI Platform
```

---

# 核心职责

Retrieval Platform 包括五个核心模块：

| 模块 | 主要职责 |
|------|----------|
| Query Understanding | 理解用户问题，优化查询语句 |
| Hybrid Retrieval | 通过多种检索方式召回候选知识 |
| Reranker | 对召回结果重新排序，提高准确率 |
| Context Builder | 组织和压缩上下文，生成最终知识片段 |
| Prompt Builder | 构建发送给 LLM 的 Prompt |

---

# 1. Query Understanding（查询理解）

## 职责

Query Understanding 是整个检索流程的第一步。

它负责：

> **理解用户真正想问什么。**

很多时候：

用户输入：

```
退款
```

真正想问的是：

```
退款流程是什么？
```

或者：

```
退款需要哪些材料？
```

因此：

不能直接拿用户输入去搜索。

---

## 主要功能

### Query Rewrite（查询改写）

例如：

```
退款

↓

退款申请流程
```

提高召回率。

---

### Query Expansion（查询扩展）

例如：

```
GPU

↓

GPU

CUDA

NVIDIA

显卡
```

增加搜索覆盖范围。

---

### Query Classification（问题分类）

例如：

判断：

```
FAQ

文档查询

SQL 查询

代码搜索
```

不同类型：

走不同检索器。

---

### Query Routing（查询路由）

例如：

```
代码问题

↓

GitHub 检索

文档问题

↓

Confluence

数据库问题

↓

SQL Retriever
```

实现智能路由。

---

## 技术难点

包括：

- 同义词识别
- 多语言支持
- 拼写纠错
- 缩写扩展
- 意图识别
- 查询路由

---

# 2. Hybrid Retrieval（混合检索）

## 职责

Hybrid Retrieval 用于：

> **从多个知识源召回候选文档。**

企业 RAG：

通常不会：

只使用向量检索。

而是：

```
Vector Search

+

BM25

+

Knowledge Graph

+

SQL

+

API

+

Graph Database
```

共同召回。

---

## 为什么需要？

例如：

问题：

```
HTTP 500
```

关键词：

BM25：

效果最好。

问题：

```
如何提高 RAG 检索效果？
```

语义：

向量检索：

效果更好。

因此：

企业：

通常：

Hybrid。

---

## 支持检索器

例如：

- Vector Retriever
- BM25 Retriever
- Parent Document Retriever
- Multi Vector Retriever
- Knowledge Graph Retriever
- SQL Retriever
- API Retriever

---

## Merge

多个 Retriever：

返回：

```
Top20

Top20

Top20
```

Merge：

```
Top50
```

交给：

Reranker。

---

## 技术难点

包括：

- 多路召回融合
- 去重
- Score 标准化
- 多知识库检索
- 多租户过滤

---

# 3. Reranker（重排序）

## 职责

Reranker 用于：

> **重新计算候选文档与问题之间的相关性，并重新排序。**

这是提升 RAG 准确率最有效的方法之一。

---

## 为什么需要？

例如：

Hybrid Retrieval：

返回：

```
Top50
```

其中：

真正有用：

只有：

```
Top5
```

Reranker：

负责：

挑出来。

---

## 工作流程

```
Question

+

Chunk

↓

Cross Encoder

↓

Score

↓

Top5
```

最终：

只保留：

最相关：

几个 Chunk。

---

## 常见模型

例如：

- BGE Reranker
- Jina Reranker
- Cohere Rerank
- Cross Encoder

---

## 技术难点

包括：

- 推理速度
- Batch 排序
- 长文本排序
- 多语言支持
- GPU 利用率

---

# 4. Context Builder（上下文构建）

## 职责

Context Builder 用于：

> **将多个检索结果组织为适合 LLM 理解的上下文。**

它不仅负责拼接文本。

更重要的是：

构建：

高质量 Context。

---

## 为什么需要？

直接：

```
Chunk1

Chunk2

Chunk3

Chunk4
```

发送给 LLM：

效果不好。

需要：

合理组织。

---

## 主要功能

### 去重

例如：

多个 Chunk：

重复内容。

↓

删除。

---

### Context Compression（上下文压缩）

例如：

```
10 Chunk

↓

保留：

核心内容。
```

减少：

Token。

---

### Chunk Merge

例如：

```
Chapter1

Section2

↓

Merge
```

形成完整语义。

---

### Parent Document

例如：

找到：

Chunk。

同时：

带上：

标题。

提高：

LLM 理解能力。

---

### Metadata 注入

例如：

```
来源

作者

时间

页码
```

一起发送。

---

## 输出

例如：

```
Title

Chunk

Reference

Source

Page
```

---

## 技术难点

包括：

- Context 长度控制
- Token Budget
- Chunk Merge
- 重复检测
- 信息压缩

---

# 5. Prompt Builder（Prompt 构建）

## 职责

Prompt Builder 用于：

> **根据上下文、用户问题和 Prompt 模板，构建最终发送给 LLM 的 Prompt。**

它是 Retrieval Platform 与 AI Platform 的桥梁。

---

## Prompt 组成

通常包括：

```
System Prompt

+

User Question

+

Retrieved Context

+

Conversation History

+

Output Format
```

---

## Prompt 模板

例如：

```
你是一名企业知识助手。

请仅依据提供的知识回答问题。

如果知识中没有答案，请明确说明不知道。

知识如下：

{{context}}

问题：

{{question}}
```

---

## Prompt 管理

支持：

- Prompt Version
- Prompt Template
- Prompt Variables
- Prompt A/B Test

---

## 输出

最终：

发送：

```
Prompt

↓

LLM Gateway
```

---

## 技术难点

包括：

- Token 控制
- Prompt 版本管理
- 多模型 Prompt 兼容
- Few-shot 示例管理
- 输出格式约束

---

# 五个模块之间的关系

```text
User Question
      │
      ▼
Query Understanding
      │
      ▼
Hybrid Retrieval
      │
      ▼
    Reranker
      │
      ▼
Context Builder
      │
      ▼
 Prompt Builder
      │
      ▼
 AI Platform
```

---

# 总结

Retrieval Platform 的目标不是调用 LLM，而是**最大限度提高知识召回质量和上下文质量**，为大模型生成可靠答案提供基础。

各模块职责如下：

| 模块 | 输入 | 输出 | 主要职责 |
|------|------|------|----------|
| Query Understanding | User Question | Optimized Query | 理解用户意图，完成改写、扩展、分类和路由 |
| Hybrid Retrieval | Query | Candidate Documents | 通过向量、BM25、知识图谱、SQL 等多种方式召回候选知识 |
| Reranker | Question + Candidates | Ranked Documents | 对召回结果重新排序，提高相关性 |
| Context Builder | Ranked Documents | Context | 去重、压缩、合并知识片段，构建适合 LLM 的上下文 |
| Prompt Builder | Context + Question | Prompt | 根据模板生成最终 Prompt，发送给 AI Platform |

---

# Retrieval Platform 在整个企业级 RAG 中的位置

```text
Knowledge Platform
        │
        ▼
Index Platform
        │
        ▼
=============================
   Retrieval Platform
=============================

Query Understanding
Hybrid Retrieval
Reranker
Context Builder
Prompt Builder

        │
        ▼
AI Platform
```

Retrieval Platform 位于 Index Platform 与 AI Platform 之间，是企业级 RAG 的核心智能层。它负责把用户问题转换为高质量查询，从多个索引中召回知识，通过重排序和上下文构建生成最优 Prompt，为 AI Platform 提供可靠输入。Retrieval Platform 的设计质量，直接决定了 RAG 系统的召回率、准确率和最终回答质量。