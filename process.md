# agent.py 重构记录

## 背景

`agent/agent.py` 是项目的 Agent 框架核心文件。初始版本仅包含骨架代码，存在以下问题：

1. **语法错误**：`super.__init__()` 缺少括号，应为 `super().__init__()`
2. **未使用的导入**：`from langchain.agents import create_agent`
3. **类型标注不当**：`model` 标注为 `str | BaseChatModel`（不应接收字符串），`tools` 标注为 `list[str | BaseTool]`
4. **空实现**：三个 Agent 子类只有 `__init__`，没有实际逻辑
5. **缺少运行机制**：无图构建、无执行入口、无状态管理

## 设计目标

- 利用 LangGraph 的 `StateGraph` 构建可运行的 Agent
- 三种 Agent 模式各自形成独立、完整的图结构
- 与项目已有的 `prompt.py` 提示词模板对接
- 支持工具注入与自动路由
- 类型安全、可测试

---

## 修改内容

### 1. 基础架构 — `BaseAgent`

```
BaseAgent (ABC)
├── __init__(name, model, tools, max_iterations)
├── build_graph() -> CompiledStateGraph      ← 子类实现
├── run(query) -> str                        ← 公共执行入口
├── _init_state(query) -> dict               ← 可被子类覆写
└── _extract_answer(state) -> str            ← 可被子类覆写
```

关键设计：

- `_graph` 字段缓存编译好的图，避免重复构建
- `run()` 方法封装了 `graph.invoke()` + 异常处理
- `_init_state()` 和 `_extract_answer()` 作为钩子方法，子类可覆写以扩展状态

### 2. ReAct Agent — 推理-行动循环

```
State: ReActState (messages, iteration)
Graph:
  START → agent → (continue → tools → agent | end → END)

agent 节点:
  1. 从 messages[0] 提取用户 query
  2. 格式化 REACT_PROMPT（注入 tools 描述 + tool_names）
  3. 调用 model.invoke([SystemPrompt] + messages)
  4. 返回追加后的 messages + iteration+1

tools 节点:
  1. 遍历 last_message.tool_calls
  2. 查找匹配工具并执行 tool.invoke(args)
  3. 追加 ToolMessage（含结果/错误）

路由逻辑:
  - 如果最后一条消息包含 tool_calls → continue（进入 tools）
  - 否则 → end
  - 超过 max_iterations 强制终止
```

### 3. Reflection Agent — 生成-反思-修订

```
State: ReflectionState (messages, iteration, draft)
Graph:
  START → draft → reflect → (revise → reflect | end → END)

draft 节点:
  - 使用 DRAFT_PROMPT 生成初始回答
  - 同时更新 draft 字段

reflect 节点:
  - 使用 REFLECT_PROMPT 对当前 draft 进行评审
  - 输出评分和 PASS/FAIL 结论

revise 节点:
  - 使用 REVISION_PROMPT（原答案 + 评审意见）生成修订版
  - 更新 draft 字段为最新版本

路由逻辑:
  - 评审结论含 "FAIL" 且 iteration < max_iterations → revise
  - 否则 → end
  - 最终答案从 draft 字段提取（覆写 _extract_answer）
```

### 4. Plan-and-Solve Agent — 规划-执行-润色

```
State: PlanAndSolveState (messages, iteration, plan[], plan_text, current_step, step_results[])
Graph:
  START → plan → solve → (solve | refine) → END

plan 节点:
  - 使用 PLAN_PROMPT 生成步骤计划
  - _parse_steps() 通过正则提取编号列表项

solve 节点:
  - 每次执行一步，携带前序步骤的上下文
  - 使用 SOLVE_PROMPT 按计划执行
  - 步骤结果累积到 step_results[]

refine 节点:
  - 所有步骤完成后，用 REFINE_PROMPT 汇总润色

路由逻辑:
  - current_step < len(plan) → 继续 solve
  - 全部完成 → refine
```

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 状态管理 | 每个 Agent 独立的 TypedDict | 不同类型需要不同字段（draft、plan 等），共享状态耦合度高 |
| 消息累积 | 手动拼接 `state["messages"] + [new]` | 避免对 `add_messages` reducer 的依赖，行为明确 |
| 工具执行 | 自定义 tools 节点而非 ToolNode | ToolNode 返回格式固定，手动拼接更方便融入完整消息历史 |
| Prompt 对接 | 直接 f-string 格式化 | 项目 prompt.py 中模板均为原始字符串，无需 LangChain PromptTemplate |
| 异常处理 | run() 层统一 try/except | 避免图节点内部异常导致 StateGraph 崩溃 |

---

## 与已有模块的对接

```text
agent.py → prompt.py        (使用 REACT_PROMPT, DRAFT_PROMPT 等模板)
agent.py → model/chat_llm   (通过 __main__ 测试入口对接 LLM 类)
agent.py → tools/tools.py   (通过 __main__ 测试入口注入工具)
agent.py → logging          (通过 log/logconfig 自动配置)
```

---

## 后续可扩展方向

- [ ] 持久化图结构（`graph.save()` / `graph.load()`）
- [ ] 支持流式输出（stream mode）
- [ ] ToolNode 替换自定义工具节点（简化代码）
- [ ] 增加 agent registry，支持根据配置字符串动态选择 Agent 类型
- [ ] 增加 Graphviz / Mermaid 可视化
- [ ] 增加更多 Agent 模式（如 Reflexion、Tree-of-Thought、Multi-Agent）

---

## 第二轮修改：日志系统 + 失败处理 + 工具调用修复

### 触发原因

第一次提交的代码在测试中暴露了三个问题：

1. **ReAct 工具调用不生效**：`_call_agent` 使用 `REACT_PROMPT`（文本格式，引导模型输出"行动：get_weather"），但 `_route` 检查的是 `AIMessage.tool_calls` 属性——两者完全不匹配，工具永远不会被调用
2. **中间过程不可见**：除 `run()` 出了异常会记录外，LLM 说了什么、调了什么工具、路由去了哪里，完全看不见
3. **失败场景裸奔**：LLM API 抛异常直接让 `StateGraph.invoke()` 崩溃；`messages[0].content` 在消息列表为空时直接 AttributeError；Reflection 评审结果不含 PASS/FAIL 时路由无法决策

### 修改内容

#### 1. ReAct 工具调用机制重写

```text
旧方案：REACT_PROMPT(文本格式) + model.invoke()            → ❌ 无 tool_calls
新方案：model.bind_tools(tools) + 简洁 SystemPrompt         → ✅ 正确输出 tool_calls
```

关键改动：

- 构造函数中绑定工具：`self._bound = self.model.bind_tools(self.tools)`
- 系统提示词替换为简洁版本（`_REACT_SYSTEM_PROMPT`），不再要求文本格式输出
- LLM 返回的 `tool_calls` 由模型原生驱动，路由据此决策

#### 2. 全链路日志体系

每个 Agent 的每个节点都按统一规范记录日志，共分四个级别：

| 级别 | 场景 | 示例 |
|------|------|------|
| `INFO` | 生命周期事件 | `[ReActDemo] init`, `[ReActDemo] run`, `[ReActDemo] done` |
| `INFO` | 关键决策点 | `tool_calls=[get_weather]`, `score=8 | PASS=True` |
| `DEBUG` | 过程跟踪 | `route → continue (tools pending)`, `route → end (final answer)` |
| `WARNING` | 异常但不崩溃 | 空 steps、空 draft、类型意外、iter 耗尽 |
| `ERROR` | 不可恢复异常 | LLM API 调用失败（记录后继续执行） |

日志格式为 `[AgentName] event | key=value`，方便 `grep` 过滤。

#### 3. 失败场景覆盖

| 场景 | 处理方式 |
|------|----------|
| LLM API 超时/报错 | `try/except` 捕获，返回 `AIMessage(content="[LLM error: ...]")`，不走工具路径 |
| 工具执行异常 | 捕获后返回 `ToolMessage(content="Error: ...")`，不影响其他工具 |
| 模型返回空 content | `str(resp.content) if resp and resp.content else ""` |
| 消息列表为空 | `state.get("messages", [])` |
| Reflection 草稿为空 | 跳过评审，默认 PASS |
| Plan 解析出 0 个步骤 | 使用原始文本作为 1 个步骤 |
| 状态中缺少字段 | `state.get("field", default)` |

#### 4. PlanAndSolveAgent 步数控制

```python
def __init__(self, ..., max_steps: int = 5):
    self.max_steps = max_steps
```

- 自定义 `_plan_prompt` 模板，引导 LLM 生成 `3 ~ max_steps` 个步骤
- `_parse_steps` 解析后如果超过 `max_steps` 则截断
- 默认 `max_steps=5` 使总 LLM 调用数控制在 7 次以内（1 plan + 5 solve + 1 refine），约 3.5 分钟可完成
- 消除了未提供 `max_steps` 时 LLM 生成 7~8 步导致超时的问题

#### 5. __main__ 健壮化

```python
# 1. LLM 初始化失败检测
try:
    llm = LLM().llm
except Exception as e:
    raise SystemExit(1)   # 明确退出

# 2. 每个测试独立 try/except
for label, factory, query in test_cases:
    try:
        agent = factory()
        result = agent.run(query)
    except Exception as e:
        print(f"[FAIL] {label}: {e}")   # 继续执行后续测试
```

#### 6. 模块导入修复

在文件顶部添加：

```python
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
```

使 `from prompt import ...` 在任意工作目录下都能正确解析。

### 测试结果（实际运行）

```text
Test: ReActAgent        ✅ 调用 get_weather(city="北京") → "晴朗，25°C"
Test: ReflectionAgent   ✅ 草稿 → 评分 8/10 PASS → 返回最终答案
Test: PlanAndSolveAgent ✅ 5步计划 → 逐步执行 → 润色输出 1873 字符
```

三个用例均通过，总耗时约 5 分钟（受 LLM API 响应速度影响）。

### 关键设计决策（补充）

| 决策 | 选择 | 理由 |
|------|------|------|
| 工具调用方式 | `bind_tools` + 简洁 SystemPrompt | 模型原生 `tool_calls` 比文本解析更可靠，且兼容 OpenAI 函数调用标准 |
| 日志级别划分 | INFO 报进展，DEBUG 报路由 | 日常运行只看 INFO 即可了解全貌，DEBUG 供调试时开启 |
| 失败恢复策略 | 写入错误消息到 message 列表继续执行 | 保持 StateGraph 节点的纯函数特性，避免在 node 内部抛出异常 |
| Plan 步数上限 | 构造函数参数 `max_steps`，默认 5 | 平衡回答质量与执行时间，用户可调 |
| sys.path 修正 | 文件顶部一次插入 | 避免 `__main__` 中临时修改导致的导入时序问题 |
