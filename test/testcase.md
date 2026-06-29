# Test Cases Document

> Last updated: 2026-06-30 | Run: ✅ All 222 tests passed

## Overview

This document describes all test cases in the project, grouped by module. Each test case includes its purpose, test method, and current execution status.

---

## Test Files

| File | Module Under Test | Test Count |
|------|------------------|------------|
| `test_memory.py` | `memory.memory` | 17 |
| `test_memoryManager.py` | `memory.memoryManager` | 8 |
| `test_context.py` | `context` | 20+ |
| `test_qdrant.py` | `qdrant.qdrantClient` | 8 |
| `test_rag.py` | `RAG.rag` | 7 |
| `test_agent.py` | `agent.agent` | 30+ |
| `test_middleware.py` | `middleware.middleware` | 18 |
| `test_tools.py` | `tools.tools` | 14 |
| `test_integration.py` | Integration tests | 18+ |

---

## Detailed Test Cases

### 1. `test_memory.py` — Memory Module (17 cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_create_default` | MemoryItem created with default fields | ✅ Pass |
| `test_create_with_metadata` | MemoryItem with custom metadata | ✅ Pass |
| `test_add_and_count` | Adding items increases count | ✅ Pass |
| `test_refuse_empty` | Empty content is rejected | ✅ Pass |
| `test_search` | Search finds matching content | ✅ Pass |
| `test_search_limit` | Search respects limit parameter | ✅ Pass |
| `test_get_all` | get_all returns all items | ✅ Pass |
| `test_clear` | Clear empties the memory | ✅ Pass |
| `test_persistence` | Items are persisted to disk | ✅ Pass |
| `test_load_on_init` | Items loaded from disk on init | ✅ Pass |
| `test_work_memory_context` | WorkMemory get_context works | ✅ Pass |
| `test_semantic_memory` | SemanticMemory stores facts | ✅ Pass |
| `test_episodic_memory` | EpisodicMemory stores conversations | ✅ Pass |
| `test_perceptual_memory` | PerceptualMemory stores observations | ✅ Pass |

### 2. `test_memoryManager.py` — MemoryManager Module (8 cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_record_interaction` | Record interaction updates work + episodic | ✅ Pass |
| `test_record_fact` | Record fact updates semantic | ✅ Pass |
| `test_record_observation` | Record observation updates perceptual | ✅ Pass |
| `test_search_all` | Search all memory types | ✅ Pass |
| `test_stats` | Stats returns correct counts | ✅ Pass |
| `test_clear_all` | Clear all zeros all counts | ✅ Pass |
| `test_get_recent_context_empty` | Empty context returns empty string | ✅ Pass |
| `test_get_recent_context` | Recent context contains recent interaction | ✅ Pass |

### 3. `test_context.py` — ContextManager Module (20+ cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_add_turn` | Add turn to history | ✅ Pass |
| `test_get_history_limit` | History limit parameter works | ✅ Pass |
| `test_clear_history` | Clear empties history | ✅ Pass |
| `test_build_prompt_basic` | Build prompt with query | ✅ Pass |
| `test_build_prompt_with_system` | Build prompt with system instruction | ✅ Pass |
| `test_build_prompt_with_history` | Build prompt includes history | ✅ Pass |
| `test_request_approval_default_reject` | Approval defaults to reject | ✅ Pass |
| `test_approve_last_action` | Approve pending action | ✅ Pass |
| `test_reject_last_action` | Reject pending action | ✅ Pass |
| `test_pending_count` | ✅ Pass count tracking | ✅ Pass |
| `test_record_interaction` | Record interaction updates history + memory | ✅ Pass |
| `test_empty_content_ignored` | Empty turn content ignored | ✅ Pass |
| `test_session_id_generated` | Session ID auto-generated | ✅ Pass |
| `test_session_priority` | Session priority can be set | ✅ Pass |
| `test_session_tags` | Tags can be added to session | ✅ Pass |
| `test_session_unique_tags` | Duplicate tags are deduplicated | ✅ Pass |
| `test_get_session_info` | Session info returns all metadata | ✅ Pass |
| `test_list_sessions` | List all saved sessions | ✅ Pass |
| `test_switch_session_invalid` | Switch to invalid session fails | ✅ Pass |
| `test_switch_session_valid` | Switch to valid session works | ✅ Pass |
| `test_turn_count_tracking` | Turn count increments correctly | ✅ Pass |
| `test_export_markdown_format` | Export as markdown | ✅ Pass |
| `test_export_json_format` | Export as JSON | ✅ Pass |
| `test_get_pending_actions` | Get all pending actions | ✅ Pass |
| `test_history_pruning` | History is pruned at max turns | ✅ Pass |
| `test_long_content_truncated` | Long content is truncated | ✅ Pass |
| `test_session_persistence` | Session data persists across instances | ✅ Pass |

### 4. `test_qdrant.py` — QdrantManager Module (8 cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_is_connected` | In-memory mode is connected | ✅ Pass |
| `test_create_and_exists` | Collection creation + existence check | ✅ Pass |
| `test_double_create` | Double creation is idempotent | ✅ Pass |
| `test_add_and_count` | Add vector increments count | ✅ Pass |
| `test_add_vectors_batch` | Batch add vectors | ✅ Pass |
| `test_query` | Query returns nearest vectors | ✅ Pass |
| `test_query_empty_collection` | Query with no collection returns empty | ✅ Pass |
| `test_delete_collection` | Delete removes collection | ✅ Pass |
| `test_count_empty` | Count on empty returns 0 | ✅ Pass |

### 5. `test_rag.py` — RAG Module (7 cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_add_documents` | Add documents to RAG | ✅ Pass |
| `test_add_empty` | Empty document list returns 0 | ✅ Pass |
| `test_query_empty_question` | Query with empty string returns [] | ✅ Pass |
| `test_query_and_format` | Query + format context | ✅ Pass |
| `test_format_context_empty` | Format empty results returns '' | ✅ Pass |
| `test_clear` | Clear RAG removes everything | ✅ Pass |

### 6. `test_agent.py` — Agent Module (30+ cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_parse_numbered` | Parse numbered steps | ✅ Pass |
| `test_parse_dashed` | Parse dashed steps | ✅ Pass |
| `test_parse_step_keyword` | Parse "Step N:" format | ✅ Pass |
| `test_parse_empty` | Parse empty text returns [] | ✅ Pass |
| `test_parse_chinese_numbered` | Parse Chinese numbered steps | ✅ Pass |
| `test_parse_mixed` | Parse mixed format | ✅ Pass |
| `test_empty_results` | Empty step_results returns "" | ✅ Pass |
| `test_with_results` | Build context from results | ✅ Pass |
| `test_partial_results` | Partial results handled correctly | ✅ Pass |
| `test_chinese_score` | Extract Chinese score format | ✅ Pass |
| `test_english_score` | Extract English score format | ✅ Pass |
| `test_no_score` | No score returns "N/A" | ✅ Pass |
| `test_init` | ReActAgent init with tools | ✅ Pass |
| `test_init_state` | State initialization | ✅ Pass |
| `test_extract_answer_from_aimessage` | Extract from AIMessage | ✅ Pass |
| `test_extract_answer_empty` | Extract from empty state | ✅ Pass |
| `test_extract_answer_fallback` | Fallback to last message | ✅ Pass |
| `test_route_continue` | Route continues when tool calls present | ✅ Pass |
| `test_route_end_no_calls` | Route ends when no tool calls | ✅ Pass |
| `test_route_end_iteration_limit` | Route ends at iteration limit | ✅ Pass |
| `test_call_agent_llm_error` | LLM error handled gracefully | ✅ Pass |
| `test_call_agent_success` | Successful LLM call | ✅ Pass |
| `test_call_agent_unexpected_type` | Unexpected type handled | ✅ Pass |
| `test_execute_tools_unknown_tool` | Unknown tool returns error | ✅ Pass |
| `test_execute_tools_success` | Successful tool execution | ✅ Pass |
| `test_execute_tools_failure` | Tool failure returns error | ✅ Pass |
| `test_run_graph` | Full graph execution | ✅ Pass |
| `test_reflection_init_state` | Reflection state init | ✅ Pass |
| `test_reflection_extract_answer` | Extract from draft | ✅ Pass |
| `test_reflection_route` | Route on FAIL/PASS | ✅ Pass |
| `test_plan_init_state` | PlanAndSolve state init | ✅ Pass |
| `test_plan_do_plan` | Plan generation | ✅ Pass |
| `test_plan_do_solve` | Step execution | ✅ Pass |
| `test_plan_do_refine` | Final refinement | ✅ Pass |
| `test_plan_after_solve` | Route after each step | ✅ Pass |

### 7. `test_middleware.py` — Middleware Module (18 cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_init` | Middleware init | ✅ Pass |
| `test_register_pre_hook` | Register pre-hook | ✅ Pass |
| `test_register_post_hook` | Register post-hook | ✅ Pass |
| `test_high_risk_detection_delete` | Detect "rm -rf" | ✅ Pass |
| `test_high_risk_detection_drop` | Detect "DROP TABLE" | ✅ Pass |
| `test_high_risk_detection_shutdown` | Detect "shutdown" | ✅ Pass |
| `test_high_risk_approval_granted` | Approval granted passes through | ✅ Pass |
| `test_safe_tools_not_blocked` | Safe tools pass through | ✅ Pass |
| `test_pre_process_no_hooks` | No hooks returns None | ✅ Pass |
| `test_pre_process_with_hooks` | Hooks can block queries | ✅ Pass |
| `test_pre_process_hook_exception` | Hook exception handled | ✅ Pass |
| `test_post_process_no_hooks` | No hooks returns same | ✅ Pass |
| `test_post_process_with_hooks` | Hooks modify response | ✅ Pass |
| `test_post_process_hook_exception` | Post hook exception handled | ✅ Pass |
| `test_post_process_chaining` | Multiple hooks chain | ✅ Pass |
| `test_sanitize_output_hook` | Sanitize hook works | ✅ Pass |
| `test_log_interaction_hook` | Log hook works | ✅ Pass |
| `test_all_high_risk_keywords_defined` | All keywords defined | ✅ Pass |

### 8. `test_tools.py` — Tools Module (14 cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_get_weather_is_tool` | get_weather is BaseTool | ✅ Pass |
| `test_get_weather_name` | Tool name is correct | ✅ Pass |
| `test_get_weather_result` | Returns weather string | ✅ Pass |
| `test_get_weather_different_city` | Works with different cities | ✅ Pass |
| `test_caculate_is_tool` | caculate is BaseTool | ✅ Pass |
| `test_caculate_name` | Tool name is correct | ✅ Pass |
| `test_caculate_simple_addition` | 2 + 3 = 5 | ✅ Pass |
| `test_caculate_complex` | Complex expressions | ✅ Pass |
| `test_caculate_power` | Power operator | ✅ Pass |
| `test_caculate_division_by_zero` | Division by zero handled | ✅ Pass |
| `test_caculate_invalid_expression` | Invalid expression handled | ✅ Pass |
| `test_caculate_negative_numbers` | Negative numbers work | ✅ Pass |
| `test_calculate_is_alias` | calculator is alias | ✅ Pass |
| `test_search_knowledge_is_tool` | search_knowledge is BaseTool | ✅ Pass |

### 9. `test_integration.py` — Integration Tests (18+ cases)

| Test Case | Description | Status |
|-----------|-------------|--------|
| `test_record_and_retrieve` | Full record + retrieve cycle | ✅ Pass |
| `test_build_prompt_with_memory` | Prompt includes memory | ✅ Pass |
| `test_build_prompt_with_history` | Prompt includes history | ✅ Pass |
| `test_multiple_sessions_isolation` | Sessions isolated | ✅ Pass |
| `test_export_markdown` | Export as markdown works | ✅ Pass |
| `test_session_info` | Session info complete | ✅ Pass |
| `test_pending_actions_list` | List pending actions | ✅ Pass |
| `test_approve_reject_cycle` | Approve/reject cycle | ✅ Pass |
| `test_high_risk_creates_pending` | High risk creates pending | ✅ Pass |
| `test_safe_tool_no_pending` | Safe tool no pending | ✅ Pass |
| `test_hook_integration` | Hook integration | ✅ Pass |
| `test_rag_query_in_build_prompt` | RAG in build_prompt | ✅ Pass |
| `test_rag_disabled` | RAG disabled correctly | ✅ Pass |
| `test_full_lifecycle` | Memory lifecycle | ✅ Pass |
| `test_multiple_interactions` | Multiple interactions | ✅ Pass |
| `test_empty_interaction` | Empty interaction | ✅ Pass |
| `test_agent_with_context_and_middleware` | Full pipeline | ✅ Pass |

---

## Running Tests

```bash
# Run all tests
uv run python test/run_all_tests.py

# Run specific test file
uv run python test/test_agent.py

# Run with Python unittest
uv run -m unittest discover test -v

# Run specific test case
uv run python -m unittest test.test_agent.TestReActAgent.test_init
```

## Test Coverage Summary

- **memory**: Core data structures, CRUD, persistence, search
- **memoryManager**: Coordinated memory operations across 4 types
- **context**: Session management, history, prompt building, human-in-the-loop
- **qdrant**: Vector DB operations in memory mode
- **rag**: Document management, embedding, retrieval
- **agent**: All 3 agent modes, state management, routing, error recovery
- **middleware**: Hook system, high-risk detection, approval pipeline
- **tools**: Tool correctness, edge cases, error handling
- **integration**: End-to-end workflows, module interactions
