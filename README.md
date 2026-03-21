# mine-ai-agent-backend

> **[Versão em Português disponível aqui](./README.pt.md)**

A FastAPI backend for a multi-agent AI system orchestrated with LangGraph. The service dynamically discovers tools exposed by external MCP servers and coordinates specialized agents to handle complex user requests.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Request Flow](#request-flow)
- [Components](#components)
  - [HTTP Layer](#http-layer)
  - [Agent Service](#agent-service)
  - [Planner](#planner)
  - [Graph Builder](#graph-builder)
  - [Executor](#executor)
  - [Specialized Agents](#specialized-agents)
  - [MCP Integration](#mcp-integration)
  - [Agent Registry](#agent-registry)
  - [LLM Providers](#llm-providers)
  - [Embedding Providers](#embedding-providers)
  - [Callback System](#callback-system)
  - [Placeholder System](#placeholder-system)
- [Stack](#stack)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Project Structure](#project-structure)

---

## Overview

This service acts as the orchestration layer of an AI agent system. It receives a user message, breaks it down into a sequence of tasks using a Planner agent, dynamically builds a LangGraph execution graph, runs each task through the appropriate specialized agent (including MCP tool agents), and synthesizes a final response.

New tools can be added at any time by registering them on the external MCP server — no code changes required in this service.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI (HTTP)                      │
│                  POST /chat  ← JWT Auth                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     Agent Service                        │
│                                                         │
│  1. Load MCP agents from external servers               │
│  2. Merge with static specialized agents                │
│  3. Run Planner  →  decompose request into steps        │
│  4. Build StateGraph from plan                          │
│  5. Execute graph step by step                          │
│  6. Format output  →  resolve placeholders              │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    MCP Tool Agents  Python Coder    Output Formatter
    (dynamic, one     (static)          (static)
    per MCP tool)
          │
          ▼
    External MCP Server(s)
```

---

## Request Flow

1. **HTTP Request** — Client sends `POST /chat` with a `message` and a JWT Bearer token.
2. **Authentication** — `get_current_user` dependency decodes the JWT and makes the token available to the pipeline.
3. **MCP Discovery** — `load_mcp_agents()` connects to all configured MCP servers and instantiates one `MCPToolAgent` per tool.
4. **Agent Registry** — All agents (static + dynamic) are registered. The registry generates LLM summaries and builds a FAISS vector index for semantic search.
5. **Planning** — `PlannerAgent` uses the LLM to decompose the request into a sequence of `{agent, task}` steps.
6. **Graph Building** — `GraphBuilder` converts the plan into a LangGraph `StateGraph`, where each node is one plan step.
7. **Execution** — `ExecutorAgent` streams the graph, capturing each node's output and accumulating results.
8. **Formatting** — `OutputFormatterAgent` synthesizes all step outputs into a final user-readable response.
9. **Placeholder Resolution** — Any `{{key}}` placeholders remaining in the response are replaced with real values from the execution context.
10. **Response** — The final text is returned to the client.

---

## Components

### HTTP Layer

**`main.py`**
- Initializes the FastAPI application.
- Configures CORS for frontend clients (Angular on `:4200`, React on `:3000`).
- Registers global exception handlers.

**`api/routers/chat.py`**
- Exposes `POST /chat`.
- Validates the request body (`ChatRequest.message`).
- Delegates execution to `run_agent()`.

**`api/dependencies/auth.py`**
- Extracts and validates the HTTP Bearer token.
- Decodes the JWT using `INTERNAL_TOKEN_SECRET`.
- Provides a `CurrentUser` object to the route handler.

---

### Agent Service

**`services/agent_service.py`** — Main orchestration function `run_agent(request, token)`.

Responsibilities:
- Instantiates the LLM via the factory.
- Loads MCP agents with lifecycle callbacks (`inject_token`, `store_result_in_context`).
- Registers static agents with their callbacks (`store_code_in_context`).
- Calls the Planner, Graph Builder, Executor, and Output Formatter in sequence.
- Resolves all `{{placeholder}}` values before returning the final response.

---

### Planner

**`agents/planner/agent.py`** — `PlannerAgent`

Given the user request and a description of all available agents, the Planner uses the LLM to produce a structured `Plan`:

```
Plan
├── reasoning: str          # Strategy explanation
└── steps: list[PlanStep]
    ├── agent: str          # Which agent to call
    └── task: str           # What to ask it
```

Before planning, the Planner can optionally query the `AgentRegistry` to semantically filter the most relevant agents, avoiding context overload with a large tool list.

---

### Graph Builder

**`agents/graph_builder/builder.py`** — `GraphBuilder`

Converts a `Plan` into a LangGraph `StateGraph`:

```
START → step_0 → step_1 → ... → step_n → END
```

Each node receives a `PipelineState` containing:
- `request` — original user request
- `context` — shared key-value store (passed through all nodes)
- `results` — accumulated list of step outputs

Nodes call `agent.invoke(task, context)` and append the result to `results`.

---

### Executor

**`agents/executor/agent.py`** — `ExecutorAgent`

Streams the compiled graph via `graph.stream()` and captures each step's output as a `StepResult`. Returns an `ExecutionResult` with all steps and a reference to the final output.

---

### Specialized Agents

**`agents/specialized/python_coder.py`** — `PythonCoderAgent`

Generates Python code from a natural language description. Returns only the code block (no explanations). Results are stored in context via the `store_code_in_context` callback.

**`agents/specialized/output_formatter.py`** — `OutputFormatterAgent`

Receives the original request and all step outputs, then synthesizes them into a coherent, user-friendly response. Rules:
- Responds in the user's original language.
- Preserves `{{placeholder}}` tokens exactly.
- Does not fabricate or infer missing data.
- Removes technical noise (stack traces, debug metadata).

---

### MCP Integration

**`agents/mcp/mcp_tool_agent.py`** — `MCPToolAgent`

One agent is created per MCP tool. Execution lifecycle:

```
PRE_LLM  →  extract args from context/LLM
PRE_MCP  →  inject token, validate args
MCP call →  invoke tool on external server
POST_MCP →  normalize result, store in context
POST_LLM →  finalize output
```

The agent uses a two-stage argument extraction strategy:
1. **Context mapping** — values already in context get `{{key}}` placeholders (never exposed to the LLM).
2. **LLM extraction** — remaining required fields are extracted by the LLM from the task description.

**`agents/mcp/loader.py`** — `load_mcp_agents()`

Connects to each URL in `MCP_URLS`, calls `list_tools()`, and instantiates one `MCPToolAgent` per tool. Connection errors are handled gracefully.

**`mcp/mcp_client.py`** — `MCPClientFactory`

Registry-based factory for MCP transport implementations. Built-in transports:
- `streamable_http` (default)
- `sse` (Server-Sent Events)

Custom transports can be registered at runtime via `MCPClientFactory.register()`.

---

### Agent Registry

**`registry/agent_registry.py`** — `AgentRegistry`

Provides semantic search over the available agents:

1. **Summary generation** — For each agent, asks the LLM to produce a structured `AgentSummary` (`name`, `summary`, `tags`) and persists it to disk.
2. **Indexing** — Embeds all summaries (`summary + tags`) and stores them in a FAISS index. Skips re-indexing if the index is already up to date.
3. **Search** — Optimizes the user query via LLM (rewrites to English search form), embeds it, and returns the top-k most relevant agent names.

**`registry/store/faiss_store.py`** — `FaissVectorStore`

L2-normalized cosine similarity search using FAISS `IndexFlatIP`. Persists the index to `registry.faiss` and IDs to `registry_ids.json`.

**`registry/embedder.py`** — `Embedder`

Selects the embedding backend based on `EMBEDDING_PROVIDER`:

| Provider | Backend | Requires |
|---|---|---|
| `lmstudio` | `OpenAIEmbeddings` (local) | LMStudio running |
| `openai` | `OpenAIEmbeddings` | `OPENAI_KEY` |
| `fastembed` | `FastEmbedEmbeddings` | Nothing (offline) |

---

### LLM Providers

**`llm/factory.py`** — `get_llm()`

Returns a `BaseChatModel` based on `MODEL_PROVIDER`:

| Value | Provider | Config |
|---|---|---|
| `lmstudio` | `LMStudioProvider` | `LMSTUDIO_URL`, `LMSTUDIO_MODEL` |
| `openai` | `OpenAIProvider` | `OPENAI_KEY`, `OPENAI_MODEL` |
| `anthropic` | `ClaudeProvider` | `ANTHROPIC_KEY`, `ANTHROPIC_MODEL` |

All providers return a LangChain `BaseChatModel` compatible with `.bind_tools()` and `.ainvoke()`.

---

### Embedding Providers

`EMBEDDING_PROVIDER` is **independent** of `MODEL_PROVIDER`. Any combination is valid:

```env
MODEL_PROVIDER=anthropic
EMBEDDING_PROVIDER=fastembed   # offline embeddings with Claude as LLM
```

---

### Callback System

Agents support lifecycle hooks via the `AgentEvent` enum:

| Event | When | Use cases |
|---|---|---|
| `PRE_LLM` | Before LLM call | Enrich or rewrite the prompt |
| `POST_LLM` | After LLM response | Log output, extract code |
| `PRE_MCP` | Before MCP tool call | Inject token, validate args |
| `POST_MCP` | After MCP tool returns | Normalize result, store in context |

Callbacks are async callables: `(event, state, context) → (state, context) | None`.

Built-in callbacks (`agents/mcp/callbacks.py`):
- **`inject_token`** — replaces `{{token}}` in MCP args with the real JWT.
- **`store_result_in_context`** — saves MCP output to `context["{tool}.result"]` and replaces the output with a placeholder.
- **`store_code_in_context`** — extracts generated code from markdown blocks and saves to context.

---

### Placeholder System

Context values are never sent raw to the LLM. Instead, they are stored as `{{key}}` placeholders:

```
context["bucket_list.result"] = {"content": "[...]", "type": "json"}
LLM sees: "{{bucket_list.result}}"
```

After execution, `resolve_placeholders(text, context)` substitutes all `{{key}}` references with their real values before returning the final response.

---

## Stack

| Layer | Technology |
|---|---|
| HTTP | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| LLM | LangChain (OpenAI / Anthropic / LMStudio) |
| MCP | `mcp[cli]` + `langchain-mcp-adapters` |
| Vector Search | FAISS (`faiss-cpu`) |
| Embeddings | `langchain-openai` / `fastembed` |
| Auth | `python-jose` (JWT HS256) |
| Config | `pydantic-settings` + `.env` |
| Runtime | Python 3.11+ / Poetry |

---

## Configuration

Adjust the `.env` file:

```env
# LLM provider: lmstudio | openai | anthropic
MODEL_PROVIDER='lmstudio'

# Embedding provider (independent of MODEL_PROVIDER)
# lmstudio | openai | fastembed
EMBEDDING_PROVIDER='lmstudio'

# LMStudio
LMSTUDIO_URL='http://localhost:1234/v1'
LMSTUDIO_MODEL='google/gemma-3n-e4b'
LMSTUDIO_EMBEDDING_MODEL='text-embedding-nomic-embed-text-v1.5'

# FastEmbed (offline)
# BAAI/bge-small-en-v1.5 | BAAI/bge-large-en-v1.5 | nomic-ai/nomic-embed-text-v1.5
FASTEMBED_EMBEDDING_MODEL='BAAI/bge-small-en-v1.5'

# OpenAI
OPENAI_KEY=''
OPENAI_MODEL='gpt-4o'

# Anthropic / Claude
# claude-opus-4-6 | claude-sonnet-4-6 | claude-haiku-4-5
ANTHROPIC_KEY=''
ANTHROPIC_MODEL='claude-opus-4-6'

# MCP server URLs (JSON array)
MCP_URLS='["http://localhost:8000/mcp"]'

# Local storage for summaries and FAISS index
DATA_DIR='data'

# JWT signing secret
INTERNAL_TOKEN_SECRET='super-secret-change-this'
```

---

## Running Locally

```bash
# Install dependencies
poetry install

# Start the server
poetry run uvicorn mine_ai_agent_service.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

## Project Structure

```
mine_ai_agent_service/
├── main.py                        # FastAPI app, CORS, exception handlers
├── config.py                      # Centralized settings (pydantic-settings)
│
├── api/
│   ├── router.py                  # Aggregates all routers
│   ├── exception_handlers.py      # Global error responses
│   ├── routers/
│   │   └── chat.py                # POST /chat endpoint
│   └── dependencies/
│       └── auth.py                # JWT authentication dependency
│
├── services/
│   └── agent_service.py           # Main orchestration: run_agent()
│
├── agents/
│   ├── base.py                    # BaseAgent ABC + callback dispatcher
│   ├── events.py                  # AgentEvent enum, AgentCallback type
│   ├── planner/
│   │   └── agent.py               # PlannerAgent: decomposes requests
│   ├── executor/
│   │   └── agent.py               # ExecutorAgent: runs compiled graphs
│   ├── graph_builder/
│   │   └── builder.py             # GraphBuilder: plan → StateGraph
│   ├── specialized/
│   │   ├── python_coder.py        # PythonCoderAgent
│   │   └── output_formatter.py    # OutputFormatterAgent
│   └── mcp/
│       ├── mcp_tool_agent.py      # MCPToolAgent (one per MCP tool)
│       ├── loader.py              # load_mcp_agents()
│       ├── callbacks.py           # inject_token, store_result, resolve_placeholders
│       └── events.py              # MCPEvent / MCPCallback aliases
│
├── llm/
│   ├── base.py                    # BaseLLMProvider ABC
│   ├── factory.py                 # get_llm() factory
│   ├── lmstudio_provider.py       # LMStudio (OpenAI-compatible local API)
│   ├── openai_provider.py         # OpenAI
│   └── claude_provider.py         # Anthropic Claude
│
├── mcp/
│   ├── base.py                    # MCPBaseClient ABC
│   ├── mcp_client.py              # MCPClientFactory (transport registry)
│   ├── mcp_streamable_http_client.py
│   └── mcp_sse_client.py
│
├── registry/
│   ├── embedder.py                # Embedder (lmstudio / openai / fastembed)
│   ├── agent_registry.py          # Semantic search over agents
│   └── store/
│       ├── base.py                # VectorStore ABC
│       └── faiss_store.py         # FAISS implementation
│
├── core/
│   ├── logging_config.py          # Logger setup
│   └── session.py                 # decode_internal_token()
│
└── exceptions/
    ├── base.py                    # AppException base
    └── application.py             # Domain exceptions
```
