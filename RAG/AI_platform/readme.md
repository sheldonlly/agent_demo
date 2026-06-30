```
AI Platform（AI平台）
   ├── LLM Gateway
   ├── Evaluation
   ├── Monitoring
   ├── Tracing
   ├── Feedback Loop
   └── Cost Control
```

# AI Platform

> AI Platform 是企业级 RAG 系统中的 AI 能力中心（AI Infrastructure），负责统一管理大语言模型（LLM）、评测（Evaluation）、可观测性（Observability）、用户反馈（Feedback）以及成本治理（Cost Governance）。

AI Platform 并不负责知识管理（Knowledge Platform），也不负责检索（Retrieval Platform）。

它主要负责：

> **让整个 AI 系统稳定、可监控、可评估、可优化、可持续运行。**

---

# 整体架构

```text
                        User Request
                             │
                             ▼
                      Retrieval Platform
                             │
                             ▼
                        Prompt Builder
                             │
                             ▼
                       LLM Gateway
                             │
               ┌─────────────┼─────────────┐
               │             │             │
               ▼             ▼             ▼
          Monitoring      Tracing      Cost Control
               │             │             │
               └─────────────┼─────────────┘
                             ▼
                         LLM Response
                             │
               ┌─────────────┼─────────────┐
               │                           │
               ▼                           ▼
         Evaluation                  Feedback Loop
```

---

# 核心职责

AI Platform 包括六个核心模块：

| 模块 | 主要职责 |
|------|----------|
| LLM Gateway | 管理所有大模型调用，提供统一接口 |
| Evaluation | 自动评估模型回答质量 |
| Monitoring | 实时监控系统运行状态 |
| Tracing | 记录完整调用链，方便排查问题 |
| Feedback Loop | 收集用户反馈，驱动持续优化 |
| Cost Control | 控制 Token 消耗和模型调用成本 |

---

# 1. LLM Gateway（模型网关）

## 职责

LLM Gateway 是 AI Platform 的入口。

它负责：

> **统一管理所有 LLM 的调用。**

业务系统永远不会直接调用：

```
OpenAI

Claude

Qwen

DeepSeek

Gemini
```

而是：

```
Business

↓

LLM Gateway

↓

Model Provider
```

这样后续切换模型无需修改业务代码。

---

## 为什么需要？

企业通常会使用多个模型：

例如：

```
OpenAI GPT

Claude

DeepSeek

Qwen

Llama

Gemini
```

不同模型：

- API 不同
- Token 不同
- 参数不同
- 限流不同

Gateway 的作用就是：

统一接口。

---

## 主要功能

### 多模型管理

例如：

```
Question

↓

Router

↓

GPT-4

↓

Claude

↓

DeepSeek
```

根据策略选择最适合的模型。

---

### Provider 管理

支持：

```
OpenAI

Azure OpenAI

Anthropic

Google

Ollama

vLLM

SGLang

LM Studio
```

---

### Failover

例如：

```
GPT 超时

↓

Claude

↓

继续执行
```

保证高可用。

---

### Load Balance

例如：

```
DeepSeek-1

↓

DeepSeek-2

↓

DeepSeek-3
```

均衡流量。

---

### Prompt 管理

统一：

- System Prompt
- Prompt Template
- Prompt Version

---

### 输出统一

所有模型：

最终输出统一格式：

```json
{
    "content":"",
    "usage":{},
    "finish_reason":""
}
```

方便上层处理。

---

# 2. Evaluation（评测平台）

## 职责

Evaluation 用于：

> **自动评估 AI 输出质量。**

没有 Evaluation：

企业不知道：

- 模型是否变差？
- 检索是否失败？
- Prompt 是否有效？

---

## 评测维度

### Retrieval

例如：

```
Recall

MRR

NDCG

Hit Rate
```

判断：

是否检索到了正确文档。

---

### Generation

例如：

```
Faithfulness

Answer Correctness

Groundedness

Completeness

Relevance
```

判断：

回答是否可信。

---

### Agent

例如：

```
Tool Success

Task Success

Planning Accuracy
```

---

## 自动评测流程

```
Question

↓

Retriever

↓

LLM

↓

Evaluation

↓

Score
```

最终：

生成日报。

---

## 为什么重要？

没有评测：

优化毫无依据。

企业：

每次升级：

都会：

A/B Test。

---

# 3. Monitoring（监控）

## 职责

Monitoring 用于：

> **实时监控整个 AI 系统运行状态。**

它关注的是：

系统是否健康。

---

## 监控指标

例如：

### LLM

```
请求数

成功率

失败率

Latency

Timeout
```

---

### Retrieval

```
Recall

Search Time

Rerank Time
```

---

### Embedding

```
Embedding Time

Queue Size
```

---

### API

```
QPS

Error Rate

P95

P99
```

---

## 常见工具

例如：

```
Prometheus

Grafana

OpenTelemetry
```

---

## 为什么需要？

例如：

```
今天：

LLM 延迟

2 秒

↓

突然

20 秒
```

Monitoring：

立即报警。

---

# 4. Tracing（链路追踪）

## 职责

Tracing 用于：

> **完整记录一次 AI 请求的执行过程。**

它不是监控。

而是：

Debug。

---

## 一次 Trace

例如：

```
Question

↓

Query Rewrite

↓

Retriever

↓

Chunk

↓

Prompt

↓

LLM

↓

Output Parser

↓

Response
```

全部记录。

---

## Tracing 内容

包括：

```
Question

History

Retrieved Docs

Prompt

Token

Latency

Response

Error
```

---

## 为什么需要？

例如：

用户：

```
为什么回答错了？
```

工程师：

打开 Trace：

```
发现：

Retriever

没有召回文档。
```

不是：

LLM 的问题。

---

## 推荐工具

例如：

```
LangFuse

LangSmith

Phoenix

OpenTelemetry
```

---

# 5. Feedback Loop（反馈闭环）

## 职责

Feedback Loop 用于：

> **收集用户反馈，并持续优化整个 RAG 系统。**

企业 AI：

不是一次开发完成。

而是：

持续优化。

---

## 用户反馈

例如：

```
👍

👎
```

或者：

```
回答正确

回答错误

没有帮助
```

---

## 收集内容

包括：

```
Question

Prompt

Retrieved Docs

Answer

User Feedback
```

---

## 如何利用？

例如：

错误回答：

```
↓

分析：

Retriever？

Prompt？

LLM？

```

然后：

持续优化。

---

## 可以优化哪些？

例如：

- Chunk Strategy
- Query Rewrite
- Prompt
- Reranker
- LLM Router
- Metadata

---

## 长期价值

Feedback 是：

企业不断提升 AI 质量的重要数据来源。

---

# 6. Cost Control（成本控制）

## 职责

Cost Control 用于：

> **控制 AI 系统运行成本。**

LLM：

最大的成本：

通常来自：

```
Token
```

---

## 统计内容

例如：

```
Prompt Token

Completion Token

Embedding Token

Cache Hit

Cost
```

---

## 控制策略

### Token Budget

例如：

```
最多：

4000 Token
```

超过：

自动压缩 Context。

---

### Model Routing

例如：

简单问题：

```
Qwen
```

复杂问题：

```
GPT-4
```

降低成本。

---

### Prompt Cache

例如：

相同 Prompt：

```
Cache

↓

不用重新调用。
```

---

### Embedding Cache

重复 Chunk：

```
直接使用已有Embedding。
```

---

### 限流

例如：

```
每分钟：

100 次调用。
```

避免成本失控。

---

## 成本报表

每天生成：

```
模型使用量

Token 数

费用

平均 Cost

Top User

Top Project
```

方便企业统计预算。

---

# 六个模块之间的关系

```text
                     User Request
                          │
                          ▼
                     LLM Gateway
                          │
      ┌───────────────────┼───────────────────┐
      │                   │                   │
      ▼                   ▼                   ▼
 Monitoring          Tracing          Cost Control
      │                   │                   │
      └───────────────────┼───────────────────┘
                          ▼
                     LLM Response
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
        Evaluation             Feedback Loop
              │                       │
              └───────────┬───────────┘
                          ▼
                   Continuous Improvement
```

---

# 总结

AI Platform 的目标不是生成回答，而是**保障整个 AI 系统稳定、高效、可观测、可持续优化**。

各模块职责如下：

| 模块 | 输入 | 输出 | 主要职责 |
|------|------|------|----------|
| LLM Gateway | Prompt | LLM Response | 统一管理多模型调用、路由、降级、负载均衡 |
| Evaluation | Question + Answer | Evaluation Score | 自动评估检索和生成质量，为模型优化提供依据 |
| Monitoring | 系统运行数据 | Metrics & Alert | 实时监控性能、延迟、错误率和系统健康状态 |
| Tracing | 请求链路 | Trace Record | 记录完整执行过程，支持问题定位和调试 |
| Feedback Loop | 用户反馈 | Optimization Data | 收集反馈并持续优化 Prompt、Retriever、模型等能力 |
| Cost Control | Token Usage | Cost Report | 控制 Token 消耗、模型调用成本和资源预算 |

---

# AI Platform 在整个企业级 RAG 中的位置

```text
Knowledge Platform
        │
        ▼
Index Platform
        │
        ▼
Retrieval Platform
        │
        ▼
=============================
        AI Platform
=============================

LLM Gateway
Evaluation
Monitoring
Tracing
Feedback Loop
Cost Control
```

AI Platform 位于 RAG 系统的最上层，为整个 AI 应用提供统一的大模型服务、质量保障、运行监控和持续优化能力，是企业级 AI 基础设施不可或缺的一部分。