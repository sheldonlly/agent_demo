# Architecture Document

> Version: 0.2.0 | Last Updated: 2026-06-29

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI Agent Platform                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  ReAct   │  │Reflection│  │ Plan &   │  │   Middleware      │   │
│  │  Agent   │  │  Agent   │  │  Solve   │  │  ┌─────────────┐ │   │
│  │          │  │          │  │  Agent   │  │  │ Pre-hooks   │ │   │
│  │ Think →  │  │ Draft →  │  │ Plan →   │  │  │ Risk Detect │ │   │
│  │ Act →    │  │ Reflect →│  │ Solve →  │  │  │ Post-hooks  │ │   │
│  │ Observe  │  │ Revise   │  │ Refine   │  │  └─────────────┘ │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │             │             │                   │             │
│  ┌────▼─────────────▼─────────────▼───────────────────▼──────────┐ │
│  │                    ContextManager                              │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐   │ │
│  │  │ History  │ │ Session  │ │ Prompt   │ │ Human-in-the-  │   │ │
│  │  │          │ │ Manager  │ │ Builder  │ │ Loop           │   │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────┘   │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                             │                                     │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │                    Memory System                              │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │ │
│  │  │  Work    │ │ Semantic │ │Episodic  │ │ Perceptual   │    │ │
│  │  │  Memory  │ │  Memory  │ │  Memory  │ │   Memory     │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │ │
│  └──────────────────────────┬──────────────────────────────────┘ │
│                             │                                     │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │                    RAG System                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │  Embedding   │→│   Vector DB   │→│   Retriever      │  │ │
│  │  │  (BGE-M3)    │  │   (Qdrant)   │  │   + Formatter    │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                     │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │                    Tool System                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │  LangChain   │  │  MCP Server  │  │  MCP Client      │  │ │
│  │  │  Tools       │  │  (FastMCP)   │  │  (Multi-Server)  │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                     │
│  ┌──────────────────────────▼──────────────────────────────────┐ │
│  │                    LLM Layer                                 │ │
│  │  ┌──────────────────────┐  ┌────────────────────────────┐   │ │
│  │  │  Chat LLM            │  │  Embedding LLM             │   │ │
│  │  │  (DeepSeek-V4-Flash) │  │  (BAAI/bge-m3)             │   │ │
│  │  └──────────────────────┘  └────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Architecture

### 2.1 Agent Layer (`agent/agent.py`)

```
BaseAgent (ABC)
├── __init__(name, model, tools, max_iterations, memory, rag, context, middleware)
├── build_graph() -> CompiledStateGraph       ← abstract
├── run(query) -> str                         ← public entry
├── _init_state(query) -> dict                ← hook
├── _extract_answer(state) -> str             ← hook
│
├── ReActAgent
│   ├── State: ReActState (messages, iteration)
│   ├── Graph: START → agent → (tools → agent | END)
│   ├── _call_agent()   — LLM + bind_tools
│   ├── _execute_tools() — tool dispatch
│   └── _route()         — tool_calls based routing
│
├── ReflectionAgent
│   ├── State: ReflectionState (messages, iteration, draft)
│   ├── Graph: START → draft → reflect → (revise → reflect | END)
│   ├── _generate_draft()
│   ├── _reflect()       — review with REFLECT_PROMPT
│   ├── _revise()        — revision with REVISION_PROMPT
│   └── _route()         — FAIL/PASS based routing
│
└── PlanAndSolveAgent
    ├── State: PlanAndSolveState (messages, iteration, plan[], plan_text, current_step, step_results[])
    ├── Graph: START → plan → solve → (solve | refine) → END
    ├── _do_plan()       — PLAN_PROMPT → step list
    ├── _do_solve()      — step-by-step execution
    ├── _do_refine()     — final polish with REFINE_PROMPT
    └── _after_solve()   — remaining steps check
```

**Key Design Decisions:**
- Each agent has its own `TypedDict` state to avoid coupling
- Error recovery at every node — LLM/tool failures produce error messages, not crashes
- `run()` wraps graph execution with try/except, middleware pre/post processing, and interaction recording
- Tools use `bind_tools()` for native function calling support

### 2.2 Context Layer (`context.py`)

```
ContextManager
├── Session Management
│   ├── _generate_session_id() — timestamp + UUID
│   ├── switch_session(id)     — hot-swap persistence
│   ├── set_session_priority() — LOW/NORMAL/HIGH/CRITICAL
│   ├── add_session_tag()      — arbitrary tags
│   ├── get_session_info()     — full metadata
│   └── list_sessions()        — all saved sessions
│
├── Conversation History
│   ├── add_turn(role, content)
│   ├── get_history(limit)
│   ├── clear_history()
│   ├── _summarize_and_prune() — auto-prune at max turns
│   └── export_history(format) — JSON or Markdown
│
├── Prompt Assembly
│   ├── build_prompt(query, system_instruction, use_rag)
│   │   └── Assembled: [System] + [Recent Memory] + [Knowledge] + [Conversation] + [Query]
│
├── Human-in-the-Loop
│   ├── request_approval(action) -> bool
│   ├── approve_last_action() -> bool
│   ├── reject_last_action() -> bool
│   └── get_pending_actions() -> list
│
└── Interaction Recording
    ├── record_interaction(query, response)
    └── → add_turn(user) + add_turn(assistant) + memory.record_interaction()
```

**Session Storage Layout:**
```
memory/data/sessions/
├── {session_id}_history.json    — conversation turns
└── {session_id}_meta.json       — session metadata
```

### 2.3 Middleware Layer (`middleware/middleware.py`)

```
Middleware
├── Hook Registry
│   ├── register_pre_hook(hook)    — PreHook = (query, tool_calls?) -> str|None
│   └── register_post_hook(hook)   — PostHook = (query, response) -> str|None
│
├── Processing Pipeline
│   ├── pre_process(query, tool_calls?)
│   │   ├── 1. High-risk check → block if dangerous
│   │   └── 2. User hooks → first block wins
│   └── post_process(query, response)
│       └── 1. User hooks → chain modification
│
├── High-Risk Detection
│   ├── Keywords: delete, drop, truncate, rm -rf, shutdown, reboot,
│   │             format, dd if=, chmod 777, exec(, eval(, subprocess,
│   │             os.system, pickle.load, shelve.open
│   └── On match → request_approval() → blocked if rejected
│
└── Built-in Hooks
    ├── sanitize_output_hook()    — filter error messages
    └── log_interaction_hook()    — log all interactions
```

### 2.4 Memory System (`memory/`)

```
MemoryManager
├── WorkMemory      — short-term working context (recent interactions)
├── SemanticMemory  — long-term factual knowledge
├── EpisodicMemory  — conversation episode storage
└── PerceptualMemory — environment observations/sensory data

Each memory type extends BaseMemory:
├── add(content, metadata)
├── search(query, limit)
├── get_all()
├── clear()
└── count()
    └── Persistence: JSON files in memory/data/
```

### 2.5 RAG System (`RAG/rag.py`)

```
RAG_Manager
├── Document Ingestion
│   ├── add_documents(texts, metadata)
│   │   └── 1. Embed documents (BGE-M3)
│   │   └── 2. Store vectors (Qdrant)
│   │   └── 3. Save to JSON backup
│
├── Retrieval
│   ├── query(question, top_k)
│   │   └── 1. Embed question
│   │   └── 2. Vector search (COSINE)
│   │   └── 3. Return payloads + scores
│   └── format_context(results, max_chars)
│
└── Lifecycle
    ├── clear() — delete collection + backup
    ├── embedding (lazy init)
    └── vector_db (lazy init, supports memory_mode)
```

### 2.6 Tool System (`tools/`)

```
Tool Registry
├── LangChain Native Tools
│   ├── get_weather(city)      — simulate weather lookup
│   ├── caculate(expression)   — arithmetic via numexpr
│   ├── calculator(expression) — alias for caculate
│   └── search_knowledge(query) — RAG-based local search
│
├── MCP Server (FastMCP)
│   ├── get_weather(city)
│   └── caculate(expression)
│       └── Transport: stdio
│
└── MCP Client (MultiServerMCPClient)
    ├── sheldonMCPServer (local, stdio)
    └── gaodeMCPServer (remote, HTTP)
```

---

## 3. Data Flow

### 3.1 ReAct Agent Flow

```
User Query
    │
    ▼
Middleware.pre_process()
    │
    ▼
ContextManager.build_prompt(query)
    ├── [System] instruction
    ├── [Recent Memory] from WorkMemory
    ├── [Knowledge] from RAG (optional)
    ├── [Conversation] history
    └── [Query]
    │
    ▼
StateGraph.invoke()
    │
    ├── agent node: LLM(bound_tools) → AIMessage
    │   ├── Middleware.pre_process(tool_calls)  → block if high-risk
    │   └── tool_calls present? → continue route
    │
    ├── tools node: for each tool_call
    │   ├── find tool by name
    │   ├── tool.invoke(args) → ToolMessage
    │   └── error → ToolMessage("Error: ...")
    │
    └── (loop until no tool_calls or max_iterations)
    │
    ▼
Middleware.post_process()
    │
    ▼
ContextManager.record_interaction(query, answer)
    │
    ▼
Final Answer
```

### 3.2 Reflection Agent Flow

```
User Query
    │
    ▼
Draft Node: generate initial answer (DRAFT_PROMPT)
    │
    ▼
Reflect Node: review answer (REFLECT_PROMPT)
    ├── PASS → END
    └── FAIL → Revise Node (max 3 iterations)
            │
            └──→ Reflect Node (re-check)
```

### 3.3 Plan-and-Solve Agent Flow

```
User Query
    │
    ▼
Plan Node: decompose into steps (PLAN_PROMPT)
    │
    ▼
Solve Node: execute step i (SOLVE_PROMPT)
    ├── more steps? → Solve Node (i+1)
    └── all done → Refine Node
            │
            ▼
        Refine Node: polish answer (REFINE_PROMPT)
```

---

## 4. Exception Handling Strategy

| Layer | Exception Type | Handling Strategy |
|-------|---------------|-------------------|
| **LLM** | API timeout, auth failure, rate limit | Caught per-node, returns `AIMessage("[LLM error: ...]")`, graph continues |
| **Embedding** | API failure, invalid response | Caught in RAG query, returns empty results, prompt built without knowledge |
| **Tool** | Runtime error, unexpected input | Caught per tool call, returns `ToolMessage("Error: ...")`, other tools unaffected |
| **Vector DB** | Connection failure, collection missing | Lazy init with retry, in-memory fallback mode |
| **Middleware** | Hook exception | Caught per-hook, failing hook skipped, pipeline continues |
| **Context** | File I/O, JSON parse | Caught on load/save, falls back to empty/default state |
| **Agent Run** | Any unhandled | `run()` wraps with try/except, returns "Execution error.", logs full trace |
| **State** | Missing keys | All access via `state.get("key", default)`, no KeyError |
| **Graph** | Invalid routing | `Literal` types enforce valid route names at type-check time |

**Golden Rule:** Never let an exception in a graph node crash the entire graph. Always produce a valid state dict with error information embedded in messages.

---

## 5. Persistence Layer

```
memory/data/
├── context_history.json          ← Legacy (migration only)
├── work_memory.json              ← WorkMemory
├── semantic_memory.json          ← SemanticMemory
├── episodic_memory.json          ← EpisodicMemory
├── perceptual_memory.json        ← PerceptualMemory
├── sessions/
│   ├── {session_id}_history.json ← Session conversation turns
│   └── {session_id}_meta.json    ← Session metadata
└── {collection}_docs.json        ← RAG document backup

qdrant/qdrant_storage/           ← Qdrant persistent storage (disk mode)
```

---

## 6. Module Dependency Graph

```
                     ┌──────────┐
                     │  main.py │
                     └────┬─────┘
                          │
              ┌───────────┴───────────┐
              │                       │
         ┌────▼────┐           ┌──────▼──────┐
         │ agent/  │           │  context.py │
         │agent.py │           └──────┬──────┘
         └────┬────┘                  │
              │              ┌────────┴────────┐
         ┌────▼────┐         │                 │
         │prompt.py│    ┌────▼─────┐     ┌─────▼─────┐
         └─────────┘    │ memory/  │     │  RAG/     │
                        │manager.py│     │  rag.py   │
                        └────┬─────┘     └─────┬─────┘
                             │                 │
                        ┌────▼─────┐     ┌─────▼─────┐
                        │ memory/  │     │ qdrant/   │
                        │memory.py │     │client.py  │
                        └──────────┘     └─────┬─────┘
                                               │
                          ┌──────────────┐ ┌───┴────────┐
                          │ model/       │ │ model/     │
                          │ chat_llm.py  │ │embedding.py│
                          └──────────────┘ └────────────┘
```

---

## 7. Configuration & Environment

### Environment Variables (`.env`)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `LLM_API_KEY` | ✅ | — | Chat LLM API key |
| `LLM_MODEL_NAME` | ❌ | `deepseek-ai/DeepSeek-V4-Flash` | Chat model name |
| `LLM_BASE_URL` | ❌ | `https://api.siliconflow.cn/v1` | API endpoint |
| `LLM_PROVIDER` | ❌ | `openai` | Provider type |
| `EMBEDDING_LLM_API_KEY` | ✅ | — | Embedding API key |
| `EMBEDDING_LLM_MODEL_NAME` | ❌ | `BAAI/bge-m3` | Embedding model |
| `EMBEDDING_LLM_BASE_URL` | ❌ | Same as LLM | Embedding endpoint |
| `TAVILY_API_KEY` | ❌ | — | Web search API key |

### Logging Configuration (`log/log_config.yaml`)

- Dual output: console (stdout) + rotating file
- Format: `timestamp - file:line - function - LEVEL - message`
- Log level: INFO (configurable via YAML)

---

## 8. Key Features

### 8.1 Multi-Mode Agent Architecture
- **ReAct**: Classic reasoning + acting loop with tool calling
- **Reflection**: Self-review and refinement pipeline
- **Plan-and-Solve**: Decomposition-based complex task execution

### 8.2 Context Management
- Session isolation with persistent storage
- Priority-based session management
- Tagging and metadata tracking
- Automatic history pruning and summarization

### 8.3 Human-in-the-Loop
- High-risk action detection via keyword matching
- Middleware approval pipeline
- Pending action management (approve/reject)
- Extensible hook system

### 8.4 Cognitive Memory System
- 4 memory types mimicking human cognition
- Persistent JSON storage
- Cross-memory search
- Automatic interaction recording

### 8.5 RAG (Retrieval-Augmented Generation)
- Document ingestion with automatic embedding
- Vector similarity search (Qdrant, COSINE distance)
- Context-aware prompt augmentation
- In-memory mode for testing

### 8.6 Dual Tool System
- LangChain native `@tool` decorator tools
- MCP protocol (FastMCP server + MultiServerMCPClient)
- Support for local and remote MCP servers

### 8.7 Middleware Pipeline
- Pre/post processing hooks
- High-risk action detection with configurable keywords
- Approval-based action gating
- Extensible hook registration

---

## 9. Development Guide

### Adding a New Agent Mode

```python
from agent.agent import BaseAgent
from typing_extensions import TypedDict

class CustomState(TypedDict):
    messages: list[BaseMessage]
    iteration: int
    custom_field: str

class CustomAgent(BaseAgent):
    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(CustomState)
        # ... define nodes and edges
        return builder.compile()
```

### Adding a New Tool

```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Description of what this tool does"""
    return f"Result: {param}"
```

### Adding a New Middleware Hook

```python
middleware.register_pre_hook(lambda query, tool_calls: "blocked" if "secret" in query else None)
middleware.register_post_hook(lambda query, response: response.replace("password", "***"))
```

---

## 10. Testing Architecture

```
test/
├── __init__.py
├── run_all_tests.py         ← Test runner/discovery
├── testcase.md              ← Test case documentation
├── test_memory.py           ← Unit tests: memory module
├── test_memoryManager.py    ← Unit tests: memory manager
├── test_context.py          ← Unit tests + session tests
├── test_qdrant.py           ← Unit tests: vector DB
├── test_rag.py              ← Unit tests: RAG manager
├── test_agent.py            ← Unit tests: all 3 agent types
├── test_middleware.py       ← Unit tests: middleware
├── test_tools.py            ← Unit tests: tools
└── test_integration.py      ← Cross-module integration tests
```

**Test Strategy:**
- **Unit tests**: Each module tested in isolation with mocked dependencies
- **Integration tests**: Cross-module workflows (Context + Memory + Middleware + RAG)
- **Edge cases**: Empty inputs, LLM errors, tool failures, type mismatches
- **Mocking strategy**: LLM calls mocked with `MagicMock`, RAG embeddings patched, Qdrant uses `memory_mode=True`
