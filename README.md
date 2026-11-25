# DocuChat

An Enterprise RAG API with a **Researcher → Critic** multi-agent workflow.
Upload PDFs (or drop them in `documents/`), ask a question, and get an answer
that has been checked against the retrieved source passages before it is
returned.

The API is built with **FastAPI**, **LangChain**, **ChromaDB**, pypdf, and
`langchain-openai`. Groq (free tier) and local Ollama models both work through
LangChain's OpenAI-compatible adapter — no paid provider is required.

## Technology stack

| Layer | Technology | Role |
| ----- | ---------- | ---- |
| API | **FastAPI** + Pydantic v2 | `/ask`, `/ingest`, `/health`, validation, OpenAPI docs |
| Agent orchestration | **LangChain** | Researcher/Critic prompts, `ChatOpenAI`, LCEL chains, JSON parsing |
| Vector database | **ChromaDB** | Persistent local vector store and top-k retrieval |
| Containerization | **Docker** / Docker Compose | Reproducible image, healthcheck, persistent Chroma volume |
| PDF ingestion | pypdf + `RecursiveCharacterTextSplitter` | Page extraction and overlapping chunking |
| LLM inference | Groq, Ollama, or OpenAI | OpenAI-compatible chat completions |
| CI | GitHub Actions | Installs dependencies and runs pytest on every push to `main` |

## Multi-agent workflow

```text
documents/*.pdf  or  POST /ingest
        │
        ▼
pypdf → RecursiveCharacterTextSplitter → embeddings → Chroma
                                                        │
POST /ask ──────────────────────────────────────────────┘
        │
        ▼
   Researcher Agent
   drafts an answer using only the top-k passages
        │
        ▼
   Critic Agent
   claim-by-claim check against the same passages
        │
        ├── approved ──▶ verified answer returned
        │
        └── flagged ───▶ Researcher revises (bounded by CRITIC_MAX_REVISIONS)
                              │
                              ▼
                         Critic re-reviews
                              │
                              ├── approved ──▶ verified answer
                              └── still unsupported ──▶ safe refusal,
                                                        verified=false
```

The API response exposes the complete `workflow` trace (`researcher` draft,
`critic` review, optional revision, fallback) so the multi-agent hand-off is
observable rather than hidden.

## Quick start with Groq

1. Put a PDF in `documents/`:

   ```bash
   mkdir -p documents
   cp /path/to/document.pdf documents/document.pdf
   ```

2. Create a free Groq key at <https://console.groq.com/keys> and add it to
   `.env`:

   ```bash
   cp .env.example .env
   # edit .env:
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_...
   # optional: LLM_MODEL=llama-3.1-8b-instant
   ```

3. Start the app:

   ```bash
   docker compose up --build -d
   curl http://localhost:8000/health
   ```

4. Ask:

   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"What is the refund policy?"}'
   ```

The PDF in `documents/` is indexed automatically during startup. You can also
index additional PDFs without restarting:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/handbook.pdf"
```

## Ollama instead of Groq

```bash
ollama pull llama3.2
```

For a local Uvicorn run:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
OLLAMA_API_KEY=ollama
```

For Docker on Linux, Ollama runs on the host, so use:

```dotenv
LLM_BASE_URL=http://host.docker.internal:11434/v1
```

The compose file already adds `host.docker.internal` via `extra_hosts`. No
paid API key is needed for Ollama; the dummy key exists because the
OpenAI-compatible client requires one.

Without a key for a hosted provider, the app still starts and `/ask` falls back
to **retrieval-only mode**, clearly marked `"model": "extractive"` and
`"verified": false`. If a provider *is* configured but unreachable, `/ask`
returns `502` rather than silently degrading.

## Example response

```json
{
  "answer": "Customers may request a refund within 30 days of purchase [1].",
  "verified": true,
  "confidence": 0.95,
  "attempts": 1,
  "critic_feedback": null,
  "sources": [
    {
      "content": "DocuChat refund policy: customers may request a refund within 30 days of purchase.",
      "source": "document.pdf",
      "page": 1,
      "score": 0.3906
    }
  ],
  "model": "groq:llama-3.1-8b-instant",
  "workflow": [
    {
      "agent": "researcher",
      "action": "draft",
      "status": "completed",
      "detail": "Drafted an answer from 1 retrieved passage(s).",
      "duration_ms": 412
    },
    {
      "agent": "critic",
      "action": "review",
      "status": "approved",
      "detail": "All claims were supported by the retrieved passages.",
      "duration_ms": 388
    }
  ]
}
```

If the Critic flags unsupported claims and a revision still does not satisfy it:

```json
{
  "answer": "I couldn't verify an answer from the indexed documents. The Critic Agent flagged: ...",
  "verified": false,
  "confidence": 0.2,
  "attempts": 2,
  "critic_feedback": "The source says four hours, not 24 hours.",
  "model": "groq:llama-3.1-8b-instant",
  "workflow": [
    {"agent": "researcher", "action": "draft", "status": "completed", "detail": "...", "duration_ms": 400},
    {"agent": "critic", "action": "review", "status": "flagged", "detail": "...", "duration_ms": 390},
    {"agent": "researcher", "action": "revision", "status": "revised", "detail": "...", "duration_ms": 380},
    {"agent": "critic", "action": "review", "status": "flagged", "detail": "...", "duration_ms": 370},
    {"agent": "critic", "action": "fallback", "status": "fallback", "detail": "...", "duration_ms": 0}
  ]
}
```

The unverified draft is withheld rather than returned as if it were trusted.

## API

| Method | Path      | Body                   | Returns |
| ------ | --------- | ---------------------- | ------- |
| `GET`  | `/health` | –                      | `status`, `version`, `environment`, `indexed_chunks` |
| `POST` | `/ingest` | multipart PDF (`file`) | `document_id`, `filename`, `pages`, `chunks_added` |
| `POST` | `/ask`    | `{"question": "..."}`  | `answer`, `verified`, `confidence`, `attempts`, `critic_feedback`, `sources[]`, `model`, `workflow[]` |

Validation rules:

- `question` is required, trimmed, 1–2000 characters, and rejects unknown fields.
- Uploads must be `.pdf`, start with `%PDF-`, and stay under `MAX_UPLOAD_MB`.
- Encrypted and image-only/scanned PDFs are rejected with a clear message; OCR
  is intentionally out of scope.

## Configuration

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `DOCUCHAT_ENV` | `development` | Label returned by `/health` |
| `CHROMA_DIR` | `data/chroma` | Persistent Chroma directory |
| `DOCUMENTS_DIR` | `documents` | Folder scanned at startup for `*.pdf` |
| `AUTO_INGEST` | `true` | Index `DOCUMENTS_DIR` during startup |
| `COLLECTION_NAME` | `docuchat` | Chroma collection name |
| `EMBEDDING_DIMENSIONS` | `384` | Local hashed embedding size |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Text splitter settings |
| `RETRIEVAL_K` | `4` | Passages retrieved per question |
| `MAX_UPLOAD_MB` | `20` | PDF upload limit |
| `LLM_PROVIDER` | `groq` | `groq`, `ollama`, or `openai` |
| `GROQ_API_KEY` | unset | Groq free-tier key |
| `OPENAI_API_KEY` | unset | OpenAI key |
| `OLLAMA_API_KEY` | `ollama` | Dummy key for OpenAI-compatible Ollama |
| `LLM_API_KEY` | unset | Generic override for the provider key |
| `LLM_BASE_URL` | provider default | OpenAI-compatible endpoint |
| `LLM_MODEL` | provider default | Chat model name |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-call upstream timeout |
| `CRITIC_MAX_REVISIONS` | `1` | Researcher revisions allowed after a flag (0–3) |

Provider defaults:

| Provider | Base URL | Model |
| -------- | -------- | ----- |
| `groq` | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| `ollama` | `http://localhost:11434/v1` | `llama3.2` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |

## Project layout

```text
docuchat/
├── main.py                     # FastAPI factory, routes, lifespan, auto-ingest
├── app/
│   ├── agents.py               # Researcher/Critic LangChain orchestration
│   ├── config.py               # validated env-driven settings
│   ├── errors.py               # expected application exceptions
│   ├── rag.py                  # PDF ingestion, Chroma, embeddings, /ask hand-off
│   └── schemas.py              # Pydantic request/response + agent trace models
├── documents/                  # drop document.pdf (and other PDFs) here
├── tests/test_api.py           # API, ingestion, and scripted-agent tests
├── .github/workflows/ci.yml    # GitHub Actions: install deps + run pytest
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Design decisions and trade-offs

- **Bounded multi-agent loop, not unbounded autonomy.** The Researcher drafts,
  the Critic returns a structured verdict, and at most `CRITIC_MAX_REVISIONS`
  revisions occur. This caps latency and cost and makes failures predictable.
- **Fail closed on verification.** A flagged answer is not returned as trusted.
  After the revision budget is exhausted, the API returns a refusal plus the
  Critic's feedback and the cited sources. If the Critic's JSON is unparseable,
  the answer is treated as unverified rather than silently approved.
- **One LangChain provider adapter.** Groq and Ollama both expose
  OpenAI-compatible chat-completions APIs, so `ChatOpenAI` is the only
  integration needed; switching providers is configuration, not code.
- **Idempotent ingestion.** Chunks get deterministic IDs from
  `sha256(pdf) + page + index`, and previous chunks for that document are
  replaced. Restarting the app with the same `document.pdf` does not duplicate
  vectors.
- **Hashed embeddings by default.** They are deterministic, offline, and free,
  which keeps tests and the demo reproducible. They are lexical, not semantic.
  Swap `HashingEmbeddings` for a real model behind the same LangChain
  `Embeddings` interface; keep the dimension fixed per collection or re-index.
- **Native Chroma client.** `langchain-community`'s Chroma wrapper is deprecated
  in favour of `langchain-chroma`; using Chroma's own client keeps persistence
  explicit. LangChain is used for splitting, embeddings, prompts, chat models,
  output parsing, and the agent workflow.
- **One Uvicorn worker.** Chroma runs embedded with a local SQLite store, and
  agent calls are synchronous. For horizontal scale, use a Chroma server or a
  managed vector database, then run multiple stateless API replicas.
- **Python 3.10 pins.** `fastapi==0.161.1`, `langchain==0.3.30`, and
  `langchain-openai==0.3.35` are the newest releases in their lines that support
  3.10; newer FastAPI and LangChain 1.x require 3.11+. Python 3.10 reached
  upstream end-of-life in October 2026, so plan the 3.11+ migration.
- **Not included:** auth, rate limiting, tenant isolation, OCR, and
  prompt-injection filtering. Put the API behind an API gateway (OAuth2/JWT),
  partition collections per tenant, and treat retrieved text as untrusted input
  before using this in a real enterprise deployment.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`main` and on pull requests targeting `main`. It checks out the repository, sets
up Python 3.10 with pip caching, installs `requirements.txt`, and runs
`python -m pytest -q`. No API key or external network service is required by the
test suite.

## Tests

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The suite requires no network or API key. It includes a focused `/ask` contract
test (`test_ask_returns_200_with_answer_key`) that asserts HTTP 200 and the
presence of an `answer` field. It also uses LangChain's scripted
`FakeListChatModel` to exercise the real prompts, JSON parser, approval path,
revision path, fallback path, and malformed-Critic fail-closed behavior, plus
startup PDF ingestion, idempotent re-indexing, API validation, and error mapping.

With Docker:

```bash
docker run --rm \
  -v "$PWD/tests:/app/tests:ro" \
  docuchat:local python -m pytest -q /app/tests
```
