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
