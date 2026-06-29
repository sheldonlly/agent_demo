# Demo — LangChain + LangGraph Agent Playground

该项目是一个 **AI Agent 原型实验场**，基于 LangChain 与 LangGraph 构建，集成了多种智能体设计模式、MCP 工具协议、向量记忆/RAG 以及认知记忆体系，用于探索和对比不同 Agent 架构的可行性与效果。

---

## 项目结构

```text
.
├── agent/              # Agent 框架 —— 多模式智能体基类
│   └── agent.py        # BaseAgent / ReActAgent / ReflectionAgent / PlanAndSolveAgent
├── model/              # LLM 模型封装层
│   ├── chat_llm.py     # Chat LLM 封装（DeepSeek-V4-Flash，SiliconFlow API）
│   └── embedding_llm.py# Embedding LLM 封装（BAAI/bge-m3）
├── tools/              # 工具系统
│   ├── tools.py        # LangChain @tool 原生工具（天气、计算器）
│   ├── MCPServer.py    # MCP 协议服务端（FastMCP，stdio 传输）
│   └── MCPClient.py    # MCP 多服务器客户端（langchain-mcp-adapters）
├── memory/             # 认知记忆系统
│   ├── memory.py       # 四类记忆：工作记忆/语义记忆/情景记忆/感知记忆
│   └── memoryManager.py# 记忆管理器
├── RAG/                # 检索增强生成
│   └── rag.py          # RAG 管理器
├── qdrant/             # 向量数据库
│   └── qdrantClient.py # Qdrant 客户端封装（集合管理/向量增删查）
├── log/                # 日志系统
│   ├── logconfig.py    # 日志自动初始化（YAML 驱动）
│   └── log_config.yaml # 日志格式配置（控制台 + 文件）
├── prompt.py           # 全部提示词模板（ReAct / Draft / Reflect / Revise / Plan / Solve / Refine）
├── context.py          # 上下文工程（占位）
├── pyproject.toml      # 项目依赖与元数据
├── .env                # 环境变量（API Key、模型、地址）
└── README.md
```

---

## 核心架构

### 1. Agent 设计模式

项目内置三种经典 Agent 架构的抽象基类（`agent/agent.py`），每种模式对应独立的提示词模板（`prompt.py`）：

| 模式 | 说明 | 核心流程 |
|------|------|----------|
| **ReAct** | 推理+行动循环 | 问题 → 思考 → 行动 → 观察 → ... → 最终回答 |
| **Reflection** | 生成→反思→修正 | Draft → Reflect(评审) → Revise(修订) |
| **Plan-and-Solve** | 先规划后执行 | Plan(拆解步骤) → Solve(逐步执行) |

### 2. 工具系统（双重实现）

- **LangChain 原生工具**（`tools/tools.py`）：`get_weather`、`caculate`
- **MCP 协议工具**（`tools/MCPServer.py` + `tools/MCPClient.py`）：
  - 服务端：使用 `FastMCP` 暴露相同工具，通过 stdio 传输
  - 客户端：使用 `langchain-mcp-adapters` 连接多台 MCP 服务器（支持本地 MCP + 远程 Gaode 地图 MCP）

### 3. 记忆与 RAG

- **认知记忆**（`memory/`）：模拟人类记忆体系，分为工作记忆、语义记忆、情景记忆、感知记忆四类
- **向量数据库**（`qdrant/qdrantClient.py`）：基于 Qdrant，使用 COSINE 距离，默认 1024 维向量
- **Embedding**（`model/embedding_llm.py`）：BGE-M3 多语言嵌入模型，支持单条/批量向量化
- **RAG**（`RAG/rag.py`）：检索增强生成管理器（占位，待实现完整流程）

### 4. 数据流

```text
用户输入 → Agent(选择模式) → Prompt 模板引导 → LLM(DeepSeek-V4-Flash)
                                        ↕
                              Tool System(原生工具 / MCP)
                                        ↕
                              Memory/RAG/Vector DB(Qdrant)
```

---

## 快速开始

### 前置条件

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) 包管理器
- Qdrant 向量数据库（本地运行或远程实例）

### 安装与配置

```bash
# 克隆项目
git clone <repo-url>
cd demo

# 安装依赖（通过 uv）
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key 等信息
```

### 环境变量

| 变量 | 说明 | 示例值 |
|------|------|--------|
| `LLM_API_KEY` | Chat LLM API Key | `sk-xxx` |
| `LLM_MODEL_NAME` | 对话模型名 | `deepseek-ai/DeepSeek-V4-Flash` |
| `LLM_BASE_URL` | API 地址 | `https://api.siliconflow.cn/v1` |
| `LLM_PROVIDER` | 模型供应商 | `openai` |
| `EMBEDDING_LLM_API_KEY` | Embedding LLM API Key | `sk-xxx` |
| `EMBEDDING_LLM_MODEL_NAME` | 嵌入模型名 | `BAAI/bge-m3` |
| `EMBEDDING_LLM_BASE_URL` | Embedding API 地址 | `https://api.siliconflow.cn/v1` |
| `TAVILY_API_KEY` | Tavily 搜索 API Key | `tvly-xxx` |

### 运行示例

```bash
# 测试 Chat LLM
uv run python model/chat_llm.py

# 测试 Embedding LLM
uv run python model/embedding_llm.py

# 测试 Qdrant 向量入库与检索
uv run python qdrant/qdrantClient.py

# 测试 MCP 客户端发现工具
uv run python tools/MCPClient.py
```

---

## 依赖说明

| 依赖 | 用途 |
|------|------|
| `langchain` | 核心 LLM / Tool / Agent 框架 |
| `langchain-openai` | OpenAI 兼容接口适配 |
| `langgraph` | Agent 图编排 |
| `langchain-mcp-adapters` | LangChain ↔ MCP 协议桥接 |
| `fastmcp` | MCP 服务端框架 |
| `qdrant-client` | Qdrant 向量数据库客户端 |
| `numexpr` | 数学表达式求值 |
| `pydantic` | 数据校验与配置管理 |
| `dotenv` | 环境变量加载 |

---

## 开发规划

- [x] LLM 与 Embedding 模型封装
- [x] LangChain 原生工具定义
- [x] MCP 服务端与客户端
- [x] Qdrant 向量数据库集成
- [x] 集中式日志系统
- [x] 多模式 Agent 提示词模板
- [ ] Agent 完整运行链路（ReAct / Reflection / Plan-and-Solve）
- [ ] 记忆管理器与持久化
- [ ] RAG 完整流程（文档入库 → 检索 → 增强生成）
- [ ] 上下文工程
- [ ] Graph 可视化与 LangGraph 编排
