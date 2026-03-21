# mine-ai-agent-backend

> **[English version available here](./README.md)**

Backend FastAPI para um sistema multi-agente de IA orquestrado com LangGraph. O serviço descobre dinamicamente as tools expostas por servidores MCP externos e coordena agentes especializados para responder a requisições complexas.

---

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Fluxo de uma Requisição](#fluxo-de-uma-requisição)
- [Componentes](#componentes)
  - [Camada HTTP](#camada-http)
  - [Agent Service](#agent-service)
  - [Planner](#planner)
  - [Graph Builder](#graph-builder)
  - [Executor](#executor)
  - [Agentes Especializados](#agentes-especializados)
  - [Integração MCP](#integração-mcp)
  - [Agent Registry](#agent-registry)
  - [Providers de LLM](#providers-de-llm)
  - [Providers de Embedding](#providers-de-embedding)
  - [Sistema de Callbacks](#sistema-de-callbacks)
  - [Sistema de Placeholders](#sistema-de-placeholders)
- [Stack](#stack)
- [Configuração](#configuração)
- [Rodando Localmente](#rodando-localmente)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

Este serviço é a camada de orquestração de um sistema de agentes de IA. Ele recebe uma mensagem do usuário, a decompõe em uma sequência de tarefas usando um agente Planner, constrói dinamicamente um grafo de execução com LangGraph, executa cada tarefa no agente especializado adequado (incluindo agentes de tools MCP) e sintetiza uma resposta final.

Novas tools podem ser adicionadas a qualquer momento registrando-as no servidor MCP externo — sem necessidade de alterações no código deste serviço.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (HTTP)                        │
│               POST /chat  ← JWT Auth                    │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Agent Service                         │
│                                                         │
│  1. Carregar agentes MCP dos servidores externos        │
│  2. Mesclar com agentes estáticos especializados        │
│  3. Executar Planner  →  decompor em etapas             │
│  4. Construir StateGraph a partir do plano              │
│  5. Executar o grafo etapa por etapa                    │
│  6. Formatar saída  →  resolver placeholders            │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   MCP Tool Agents   Python Coder   Output Formatter
   (dinâmico, um      (estático)       (estático)
   por tool MCP)
          │
          ▼
   Servidor(es) MCP externo(s)
```

---

## Fluxo de uma Requisição

1. **Requisição HTTP** — O cliente envia `POST /chat` com uma `message` e um token JWT Bearer.
2. **Autenticação** — A dependency `get_current_user` decodifica o JWT e disponibiliza o token para o pipeline.
3. **Descoberta MCP** — `load_mcp_agents()` conecta a todos os servidores em `MCP_URLS` e instancia um `MCPToolAgent` por tool.
4. **Agent Registry** — Todos os agentes (estáticos + dinâmicos) são registrados. O registry gera resumos via LLM e constrói um índice FAISS para busca semântica.
5. **Planejamento** — `PlannerAgent` usa o LLM para decompor a requisição em uma sequência de etapas `{agente, tarefa}`.
6. **Construção do Grafo** — `GraphBuilder` converte o plano em um `StateGraph` do LangGraph, onde cada nó é uma etapa do plano.
7. **Execução** — `ExecutorAgent` faz stream do grafo capturando a saída de cada nó e acumulando os resultados.
8. **Formatação** — `OutputFormatterAgent` sintetiza todas as saídas em uma resposta final legível pelo usuário.
9. **Resolução de Placeholders** — Quaisquer `{{chave}}` restantes na resposta são substituídos por valores reais do contexto de execução.
10. **Resposta** — O texto final é retornado ao cliente.

---

## Componentes

### Camada HTTP

**`main.py`**
- Inicializa a aplicação FastAPI.
- Configura CORS para clientes frontend (Angular em `:4200`, React em `:3000`).
- Registra handlers globais de exceção.

**`api/routers/chat.py`**
- Expõe `POST /chat`.
- Valida o corpo da requisição (`ChatRequest.message`).
- Delega a execução para `run_agent()`.

**`api/dependencies/auth.py`**
- Extrai e valida o token HTTP Bearer.
- Decodifica o JWT usando `INTERNAL_TOKEN_SECRET`.
- Disponibiliza um objeto `CurrentUser` ao handler da rota.

---

### Agent Service

**`services/agent_service.py`** — Função principal de orquestração `run_agent(request, token)`.

Responsabilidades:
- Instancia o LLM via factory.
- Carrega agentes MCP com callbacks de ciclo de vida (`inject_token`, `store_result_in_context`).
- Registra agentes estáticos com seus callbacks (`store_code_in_context`).
- Chama Planner, Graph Builder, Executor e Output Formatter em sequência.
- Resolve todos os `{{placeholder}}` antes de retornar a resposta final.

---

### Planner

**`agents/planner/agent.py`** — `PlannerAgent`

A partir da requisição do usuário e da descrição de todos os agentes disponíveis, o Planner usa o LLM para produzir um `Plan` estruturado:

```
Plan
├── reasoning: str          # Explicação da estratégia
└── steps: list[PlanStep]
    ├── agent: str          # Qual agente chamar
    └── task: str           # O que pedir a ele
```

Antes de planejar, o Planner pode consultar o `AgentRegistry` para filtrar semanticamente os agentes mais relevantes, evitando sobrecarga de contexto com uma lista grande de tools.

---

### Graph Builder

**`agents/graph_builder/builder.py`** — `GraphBuilder`

Converte um `Plan` em um `StateGraph` do LangGraph:

```
START → etapa_0 → etapa_1 → ... → etapa_n → END
```

Cada nó recebe um `PipelineState` contendo:
- `request` — requisição original do usuário
- `context` — repositório chave-valor compartilhado (passa por todos os nós)
- `results` — lista acumulada de saídas das etapas

Os nós chamam `agent.invoke(tarefa, context)` e acrescentam o resultado a `results`.

---

### Executor

**`agents/executor/agent.py`** — `ExecutorAgent`

Faz stream do grafo compilado via `graph.stream()` e captura a saída de cada etapa como um `StepResult`. Retorna um `ExecutionResult` com todas as etapas e uma referência à saída final.

---

### Agentes Especializados

**`agents/specialized/python_coder.py`** — `PythonCoderAgent`

Gera código Python a partir de uma descrição em linguagem natural. Retorna apenas o bloco de código (sem explicações). Os resultados são armazenados no contexto via callback `store_code_in_context`.

**`agents/specialized/output_formatter.py`** — `OutputFormatterAgent`

Recebe a requisição original e todas as saídas das etapas e as sintetiza em uma resposta coerente e amigável para o usuário. Regras:
- Responde no idioma original do usuário.
- Preserva tokens `{{placeholder}}` exatamente como estão.
- Não fabrica nem infere dados ausentes.
- Remove ruído técnico (stack traces, metadados de debug).

---

### Integração MCP

**`agents/mcp/mcp_tool_agent.py`** — `MCPToolAgent`

Um agente é criado para cada tool MCP. Ciclo de vida da execução:

```
PRE_LLM  →  extrair args do contexto/LLM
PRE_MCP  →  injetar token, validar args
MCP call →  invocar tool no servidor externo
POST_MCP →  normalizar resultado, salvar no contexto
POST_LLM →  finalizar saída
```

O agente usa uma estratégia de extração de argumentos em dois estágios:
1. **Mapeamento de contexto** — valores já presentes no contexto recebem placeholders `{{chave}}` (nunca expostos ao LLM).
2. **Extração via LLM** — campos obrigatórios restantes são extraídos pelo LLM a partir da descrição da tarefa.

**`agents/mcp/loader.py`** — `load_mcp_agents()`

Conecta a cada URL em `MCP_URLS`, chama `list_tools()` e instancia um `MCPToolAgent` por tool. Erros de conexão são tratados de forma resiliente.

**`mcp/mcp_client.py`** — `MCPClientFactory`

Factory baseada em registry para implementações de transporte MCP. Transportes nativos:
- `streamable_http` (padrão)
- `sse` (Server-Sent Events)

Transportes customizados podem ser registrados em runtime via `MCPClientFactory.register()`.

---

### Agent Registry

**`registry/agent_registry.py`** — `AgentRegistry`

Fornece busca semântica sobre os agentes disponíveis:

1. **Geração de resumos** — Para cada agente, solicita ao LLM um `AgentSummary` estruturado (`name`, `summary`, `tags`) e persiste em disco.
2. **Indexação** — Gera embeddings de todos os resumos (`summary + tags`) e os armazena em um índice FAISS. Pula a re-indexação se o índice já estiver atualizado.
3. **Busca** — Otimiza a query do usuário via LLM (reescreve em forma de busca em inglês), gera o embedding e retorna os top-k nomes de agentes mais relevantes.

**`registry/store/faiss_store.py`** — `FaissVectorStore`

Busca por similaridade de cosseno com L2-normalização usando FAISS `IndexFlatIP`. Persiste o índice em `registry.faiss` e os IDs em `registry_ids.json`.

**`registry/embedder.py`** — `Embedder`

Seleciona o backend de embedding com base em `EMBEDDING_PROVIDER`:

| Provider | Backend | Requisito |
|---|---|---|
| `lmstudio` | `OpenAIEmbeddings` (local) | LMStudio rodando |
| `openai` | `OpenAIEmbeddings` | `OPENAI_KEY` |
| `fastembed` | `FastEmbedEmbeddings` | Nenhum (offline) |

---

### Providers de LLM

**`llm/factory.py`** — `get_llm()`

Retorna um `BaseChatModel` baseado em `MODEL_PROVIDER`:

| Valor | Provider | Configuração |
|---|---|---|
| `lmstudio` | `LMStudioProvider` | `LMSTUDIO_URL`, `LMSTUDIO_MODEL` |
| `openai` | `OpenAIProvider` | `OPENAI_KEY`, `OPENAI_MODEL` |
| `anthropic` | `ClaudeProvider` | `ANTHROPIC_KEY`, `ANTHROPIC_MODEL` |

Todos os providers retornam um `BaseChatModel` do LangChain compatível com `.bind_tools()` e `.ainvoke()`.

---

### Providers de Embedding

`EMBEDDING_PROVIDER` é **independente** de `MODEL_PROVIDER`. Qualquer combinação é válida:

```env
MODEL_PROVIDER=anthropic
EMBEDDING_PROVIDER=fastembed   # embeddings offline com Claude como LLM
```

---

### Sistema de Callbacks

Os agentes suportam hooks de ciclo de vida via o enum `AgentEvent`:

| Evento | Quando | Casos de uso |
|---|---|---|
| `PRE_LLM` | Antes da chamada ao LLM | Enriquecer ou reescrever o prompt |
| `POST_LLM` | Após a resposta do LLM | Logar saída, extrair código |
| `PRE_MCP` | Antes da chamada à tool MCP | Injetar token, validar args |
| `POST_MCP` | Após retorno da tool MCP | Normalizar resultado, salvar no contexto |

Callbacks são callables assíncronos: `(event, state, context) → (state, context) | None`.

Callbacks nativos (`agents/mcp/callbacks.py`):
- **`inject_token`** — substitui `{{token}}` nos args do MCP pelo JWT real.
- **`store_result_in_context`** — salva a saída do MCP em `context["{tool}.result"]` e substitui a saída por um placeholder.
- **`store_code_in_context`** — extrai código gerado de blocos markdown e salva no contexto.

---

### Sistema de Placeholders

Valores de contexto nunca são enviados brutos ao LLM. Em vez disso, são armazenados como placeholders `{{chave}}`:

```
context["bucket_list.result"] = {"content": "[...]", "type": "json"}
LLM vê: "{{bucket_list.result}}"
```

Após a execução, `resolve_placeholders(text, context)` substitui todas as referências `{{chave}}` pelos valores reais antes de retornar a resposta final.

---

## Stack

| Camada | Tecnologia |
|---|---|
| HTTP | FastAPI + Uvicorn |
| Orquestração | LangGraph |
| LLM | LangChain (OpenAI / Anthropic / LMStudio) |
| MCP | `mcp[cli]` + `langchain-mcp-adapters` |
| Busca Vetorial | FAISS (`faiss-cpu`) |
| Embeddings | `langchain-openai` / `fastembed` |
| Autenticação | `python-jose` (JWT HS256) |
| Configuração | `pydantic-settings` + `.env` |
| Runtime | Python 3.11+ / Poetry |

---

## Configuração

Ajuste o arquivo `.env`:

```env
# Provider de LLM: lmstudio | openai | anthropic
MODEL_PROVIDER='lmstudio'

# Provider de Embedding (independente do MODEL_PROVIDER)
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

# URLs dos servidores MCP (array JSON)
MCP_URLS='["http://localhost:8000/mcp"]'

# Armazenamento local para resumos e índice FAISS
DATA_DIR='data'

# Segredo para assinar tokens JWT internos
INTERNAL_TOKEN_SECRET='super-secret-change-this'
```

---

## Rodando Localmente

```bash
# Instalar dependências
poetry install

# Iniciar o servidor
poetry run uvicorn mine_ai_agent_service.main:app --reload
```

A API estará disponível em `http://localhost:8000`.

---

## Estrutura do Projeto

```
mine_ai_agent_service/
├── main.py                        # App FastAPI, CORS, exception handlers
├── config.py                      # Configurações centralizadas (pydantic-settings)
│
├── api/
│   ├── router.py                  # Agrega todos os routers
│   ├── exception_handlers.py      # Respostas de erro globais
│   ├── routers/
│   │   └── chat.py                # Endpoint POST /chat
│   └── dependencies/
│       └── auth.py                # Dependency de autenticação JWT
│
├── services/
│   └── agent_service.py           # Orquestração principal: run_agent()
│
├── agents/
│   ├── base.py                    # ABC BaseAgent + dispatcher de callbacks
│   ├── events.py                  # Enum AgentEvent, tipo AgentCallback
│   ├── planner/
│   │   └── agent.py               # PlannerAgent: decompõe requisições
│   ├── executor/
│   │   └── agent.py               # ExecutorAgent: executa grafos compilados
│   ├── graph_builder/
│   │   └── builder.py             # GraphBuilder: plano → StateGraph
│   ├── specialized/
│   │   ├── python_coder.py        # PythonCoderAgent
│   │   └── output_formatter.py    # OutputFormatterAgent
│   └── mcp/
│       ├── mcp_tool_agent.py      # MCPToolAgent (um por tool MCP)
│       ├── loader.py              # load_mcp_agents()
│       ├── callbacks.py           # inject_token, store_result, resolve_placeholders
│       └── events.py              # Aliases MCPEvent / MCPCallback
│
├── llm/
│   ├── base.py                    # ABC BaseLLMProvider
│   ├── factory.py                 # Factory get_llm()
│   ├── lmstudio_provider.py       # LMStudio (API local compatível com OpenAI)
│   ├── openai_provider.py         # OpenAI
│   └── claude_provider.py         # Anthropic Claude
│
├── mcp/
│   ├── base.py                    # ABC MCPBaseClient
│   ├── mcp_client.py              # MCPClientFactory (registry de transportes)
│   ├── mcp_streamable_http_client.py
│   └── mcp_sse_client.py
│
├── registry/
│   ├── embedder.py                # Embedder (lmstudio / openai / fastembed)
│   ├── agent_registry.py          # Busca semântica sobre agentes
│   └── store/
│       ├── base.py                # ABC VectorStore
│       └── faiss_store.py         # Implementação FAISS
│
├── core/
│   ├── logging_config.py          # Configuração de logger
│   └── session.py                 # decode_internal_token()
│
└── exceptions/
    ├── base.py                    # Base AppException
    └── application.py             # Exceções de domínio
```
