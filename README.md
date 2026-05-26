# Sankofa Legal AI

**Version 2.0.0** — Production-grade AI assistant for Zimbabwean legal research, with streaming answers, multi-turn chat memory, document review, and full user authentication.

| Layer | Technology |
|-------|------------|
| API | FastAPI 0.115 · Uvicorn (uvloop + httptools) |
| Database | PostgreSQL 16 · SQLAlchemy async · asyncpg |
| Cache | Redis 7 (reserved for sessions / rate limiting) |
| LLM & embeddings | Mistral AI (`mistral-large-latest`, `mistral-embed`) |
| Vector store | Pinecone (Zimbabwe statutes & case law index) |
| Frontend | Single-page app in `app/templates/index.html` (no build step) |
| Deployment | Docker Compose (app + db + redis) |

For exhaustive API schemas, prompt design, and performance notes, see **[documentation.md](./documentation.md)**.

---

## Table of Contents

1. [What This Application Does](#what-this-application-does)
2. [How It Works (End-to-End)](#how-it-works-end-to-end)
3. [Repository Structure](#repository-structure)
4. [Quick Start (Docker)](#quick-start-docker)
5. [Local Development](#local-development)
6. [Environment Variables](#environment-variables)
7. [Using the Application](#using-the-application)
8. [API Overview](#api-overview)
9. [Data Model](#data-model)
10. [Security](#security)
11. [Operations & Troubleshooting](#operations--troubleshooting)
12. [Legacy Files](#legacy-files)
13. [Disclaimer](#disclaimer)

---

## What This Application Does

Sankofa combines **Retrieval-Augmented Generation (RAG)** with a governed legal knowledge base:

- **Legal Research Chat** — Users ask questions about Zimbabwe law; the system retrieves relevant chunks from Pinecone, grounds the Mistral LLM on those sources, and streams cited answers via Server-Sent Events (SSE).
- **Document Review** — Users upload PDF or Word files; the system extracts text, finds legal citations with regex, verifies them against Pinecone, and returns a structured review (summary, verified/unverified citations, improvements, assessment).
- **Multi-user access** — JWT authentication, role-based access (`admin` / `user`), admin user management, password reset via email (or server logs if SMTP is unavailable).
- **Persistent history** — Chat sessions and messages stored in PostgreSQL.

---

## How It Works (End-to-End)

### System architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser — app/templates/index.html (vanilla JS, marked.js, SSE)        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTP / SSE (POST + Bearer JWT)
┌───────────────────────────────▼─────────────────────────────────────────┐
│  FastAPI — app/main.py                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐                 │
│  │ /api/v1/auth│  │ /api/v1/chat│  │/api/v1/documents │                 │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘                 │
│         └────────────────┼──────────────────┘                           │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────────┐               │
│  │ Services: rag_engine · document_review · email_service│              │
│  └──────────┬─────────────────────┬─────────────────────┘               │
│             │                     │                                     │
│  ┌──────────▼──────────┐   ┌──────▼──────┐   ┌─────────────┐            │
│  │ PostgreSQL (async)  │   │ Mistral AI  │   │  Pinecone   │            │
│  │ users, sessions,    │   │ chat + embed│   │ vector index│            │
│  │ messages, tokens    │   └─────────────┘   └─────────────┘            │
│  └─────────────────────┘                                                │
│  ┌──────────┐  Redis (configured; future rate-limit / session cache)    │
│  └──────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Legal research chat flow

When a user submits a question on the **Research** tab:

1. **Frontend** — `POST /api/v1/chat/ask` with `{ question, session_id? }` and `Authorization: Bearer <access_token>`. Opens an SSE stream via `fetch` + `ReadableStream` (not `EventSource`, so POST bodies work).

2. **API (`app/api/chat.py`)** — Validates JWT, returns `StreamingResponse` immediately (`text/event-stream`). All heavy work runs **inside** the async generator so the browser gets a connection and status events within milliseconds.

3. **Guardrails (`app/services/rag_engine.py`)** — Pure-Python checks before any external call:
   - Off-limits topics (e.g. “how to commit…”) → canned refusal
   - Jailbreak / role-play patterns → identity-lock refusal
   - Clearly non-legal topics (recipes, sports, etc.) → generic answer without retrieval

4. **Session & history** — Creates or loads `ChatSession`, persists the user message, loads last **12 messages** (6 turns) for context.

5. **Parallel RAG** — `asyncio.gather`:
   - **Intent router** — Small Mistral call → `SEARCH` | `GREETING` | `NO_SEARCH`
   - **Retrieval** — Embed query with `mistral-embed` (1024-dim) → Pinecone `query` → top-K documents
   - Optional **Veritas note** for constitutional queries

6. **Streaming generation** — SSE events:
   - `status`: `thinking` → `retrieving` → `verifying` → `finalising`
   - `citations`: deduplicated sources with `akn_url`, title, file
   - `token`: answer chunks
   - `done`: `session_id`, `elapsed_ms`

7. **Persistence** — Assistant message + citations saved to PostgreSQL; UI renders markdown once on `done`.

### Document review flow

1. **Frontend** — `POST /api/v1/documents/review` as `multipart/form-data` (`file` field).

2. **API (`app/api/documents.py`)** — Validates extension (`.pdf`, `.docx`, `.doc`) and size (`MAX_UPLOAD_SIZE_MB`, default 20).

3. **Pipeline (`app/services/document_review.py`)**:
   - Extract text (`pypdf` / `python-docx`)
   - Regex extraction of Zimbabwe citation formats
   - Per-citation Pinecone lookup (verified if score > 0.75)
   - Mistral structured review → `DocumentReviewResult` JSON

### Authentication flow

1. `POST /api/v1/auth/login` → access token (60 min) + refresh token (7 days).
2. Protected routes use `CurrentUser` / `AdminUser` dependencies (`app/core/deps.py`).
3. On `401`, the SPA retries via `POST /api/v1/auth/refresh`.
4. First startup: if no admin exists, `app/main.py` creates one from `ADMIN_EMAIL` / `ADMIN_PASSWORD`.

### Startup lifecycle

```
docker compose up
    → db & redis healthchecks pass
    → uvicorn app.main:app (2 workers)
    → lifespan: init_db() → create_all tables
    → lifespan: _ensure_default_admin()
    → serve SPA at /, /login, /reset-password
```

---

## Repository Structure

```
legal_rag_app/
│
├── app/                              # ★ Production application (use this)
│   ├── main.py                       # FastAPI factory, lifespan, routers, SPA routes, /health
│   │
│   ├── api/                          # HTTP route handlers (thin layer)
│   │   ├── auth.py                   # Login, refresh, users (admin), password reset
│   │   ├── chat.py                   # SSE streaming chat, session CRUD, message history
│   │   └── documents.py              # Multipart upload → document review
│   │
│   ├── core/                         # Cross-cutting infrastructure
│   │   ├── config.py                 # Pydantic Settings from .env (singleton via lru_cache)
│   │   ├── database.py               # Async engine, Base, get_db(), init_db()
│   │   ├── security.py               # bcrypt hash/verify, JWT create/decode
│   │   └── deps.py                   # CurrentUser, AdminUser, DBSession dependencies
│   │
│   ├── models/                       # SQLAlchemy ORM
│   │   ├── user.py                   # User, PasswordResetToken, UserRole enum
│   │   └── session.py                # ChatSession, ChatMessage (citations as JSONB)
│   │
│   ├── services/                     # Business logic & external integrations
│   │   ├── rag_engine.py             # Pinecone retrieval, intent routing, SSE generators, guardrails
│   │   ├── document_review.py        # PDF/DOCX parse, citation regex, verification, LLM review
│   │   ├── email_service.py          # SMTP password-reset emails
│   │   └── schemas.py                # Pydantic request/response models (shared by API + services)
│   │
│   ├── templates/
│   │   └── index.html                # Full SPA (~1700 lines): auth, chat, doc review, admin, theming
│   │
│   └── utils/                        # Placeholder package for shared helpers (currently empty)
│
├── migrations/                       # Reserved for Alembic migrations (.gitkeep)
├── tests/                            # Reserved for test suite (.gitkeep)
│
├── docker-compose.yml                # Orchestrates app + PostgreSQL + Redis + volumes
├── Dockerfile                        # Multi-stage build (builder → runtime, non-root user)
├── requirements.txt                  # Pinned Python dependencies
├── documentation.md                  # Deep technical reference (API, RAG, prompts, Docker)
├── README.md                         # This file
│
├── main.py                           # ⚠ Legacy: standalone RAG API prototype (do not use in production)
├── app.py                            # ⚠ Legacy: Flask static server for old templates/
├── start_all.sh                      # ⚠ Legacy: runs main.py + Flask together
├── templates/index.html              # ⚠ Legacy duplicate of SPA (superseded by app/templates/)
└── test.py                           # ⚠ Legacy Pinecone/Mistral smoke script
```

### Module responsibilities

| Path | Responsibility |
|------|----------------|
| `app/main.py` | App entry (`app.main:app`), CORS, GZip, global 500 handler, mounts `/static` if present, Jinja2 SPA |
| `app/api/auth.py` | Credentials, JWT pair, admin CRUD, forgot/reset password |
| `app/api/chat.py` | SSE generator wiring to `rag_engine`, session/message persistence |
| `app/api/documents.py` | File validation, delegates to `review_document()` |
| `app/services/rag_engine.py` | Singleton LLM/embed/Pinecone clients, retrieval, streaming, safety |
| `app/services/document_review.py` | Extraction, citation patterns, DB verification, review prompt |
| `app/services/schemas.py` | All API contracts (`ChatQueryRequest`, `Citation`, `DocumentReviewResult`, etc.) |
| `app/templates/index.html` | Entire UI: localStorage tokens, SSE consumer, admin panel, dark mode |

**Entry point for production:** `uvicorn app.main:app` (as used in Docker).

---

## Quick Start (Docker)

### Prerequisites

- Docker & Docker Compose
- [Mistral AI](https://mistral.ai/) API key
- [Pinecone](https://www.pinecone.io/) index populated with Zimbabwe legal documents (1024-dim embeddings, `mistral-embed`)

### 1. Create `.env`

Create a `.env` file in the project root (see [Environment Variables](#environment-variables)). Minimum required:

```bash
SECRET_KEY=<64+ char random string>
MISTRAL_API_KEY=<your-key>
PINECONE_API_KEY=<your-key>
PINECONE_INDEX_NAME=<your-index>
ADMIN_EMAIL=admin@sankofa.ai
ADMIN_PASSWORD=<strong-password>
```

Generate a secret key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

### 2. Build and start

```bash
docker compose up --build -d
```

This starts:

| Service | Container | Port | Role |
|---------|-----------|------|------|
| `app` | `sankofa_app` | 8000 | FastAPI + SPA |
| `db` | `sankofa_db` | 5432 | PostgreSQL 16 |
| `redis` | `sankofa_redis` | 6379 | Cache (future use) |

Docker Compose overrides `DATABASE_URL` and `REDIS_URL` to use service hostnames `db` and `redis`.

### 3. Open the app

```
http://localhost:8000
```

### 4. First login

| Field | Default (if not overridden in `.env`) |
|-------|--------------------------------------|
| Email | `admin@sankofa.ai` |
| Password | `Admin@123456` |

**Change the admin password immediately** via **My Account → Change Password**.

---

## Local Development

### Prerequisites

- Python 3.12+
- Local PostgreSQL 16 and Redis 7 (or run only `db` + `redis` via Docker)

### Setup

```bash
# Virtual environment
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# .env — point to localhost
# DATABASE_URL=postgresql+asyncpg://sankofa:sankofa_pass@localhost:5432/sankofa_db
# REDIS_URL=redis://localhost:6379/0

# Run API with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. API docs (DEBUG only): `http://localhost:8000/api/docs`.

### Optional: DB + Redis via Docker only

```bash
docker compose up -d db redis
# Keep DATABASE_URL / REDIS_URL on localhost ports 5432 / 6379
```

---

## Environment Variables

All settings are loaded by `app/core/config.py` via `pydantic-settings` from `.env` (git-ignored).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | **Yes** | auto-generated if missing | JWT signing key (use a long random value in production) |
| `MISTRAL_API_KEY` | **Yes** | — | Mistral API access |
| `PINECONE_API_KEY` | **Yes** | — | Pinecone API access |
| `PINECONE_INDEX_NAME` | **Yes** | `knowledge-base` | Vector index name |
| `ADMIN_EMAIL` | No | `admin@sankofa.ai` | Bootstrap admin email |
| `ADMIN_PASSWORD` | No | `Admin@123456` | Bootstrap admin password |
| `DATABASE_URL` | No | `localhost` URL | Async PostgreSQL URL (`postgresql+asyncpg://...`) |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection |
| `MISTRAL_CHAT_MODEL` | No | `mistral-large-latest` | Answer generation model |
| `MISTRAL_EMBED_MODEL` | No | `mistral-embed` | Query embedding model |
| `PINECONE_NAMESPACE` | No | `default` | Pinecone namespace |
| `RETRIEVER_TOP_K` | No | `5` | Chunks retrieved per query |
| `EMBED_DIM` | No | `1024` | Embedding dimensions |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `60` | Access JWT TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | No | `7` | Refresh JWT TTL |
| `MAX_UPLOAD_SIZE_MB` | No | `20` | Document upload limit |
| `ALLOWED_ORIGINS` | No | localhost list | CORS origins (JSON array in `.env`) |
| `SMTP_HOST` | No | `relay.liquidtelecom.net` | Password-reset mail relay |
| `SMTP_PORT` | No | `25` | SMTP port |
| `EMAIL_FROM` | No | `noreply@sankofa-legal.ai` | Sender address |
| `DEBUG` | No | `false` | Enables `/api/docs` and SQL echo |

Example `.env` skeleton:

```ini
APP_NAME=Sankofa Legal AI
DEBUG=false
ENVIRONMENT=development

SECRET_KEY=change-me-to-a-long-random-string
ADMIN_EMAIL=admin@sankofa.ai
ADMIN_PASSWORD=ChangeMeNow123!

DATABASE_URL=postgresql+asyncpg://sankofa:sankofa_pass@localhost:5432/sankofa_db
REDIS_URL=redis://localhost:6379/0

MISTRAL_API_KEY=
MISTRAL_CHAT_MODEL=mistral-large-latest
MISTRAL_EMBED_MODEL=mistral-embed

PINECONE_API_KEY=
PINECONE_INDEX_NAME=knowledge-base
PINECONE_NAMESPACE=default
RETRIEVER_TOP_K=5

SMTP_HOST=relay.liquidtelecom.net
SMTP_PORT=25
EMAIL_FROM=noreply@sankofa-legal.ai

ALLOWED_ORIGINS=["http://localhost:8000","http://127.0.0.1:8000"]
```

---

## Using the Application

### Legal Research

1. Log in → **Research** tab
2. Type a Zimbabwe law question → Enter
3. Watch status phases (thinking → retrieving → verifying → finalising)
4. Answer streams with markdown; sources appear when citations are emitted
5. Sessions auto-save; reopen from the sidebar

### Document Review

1. **Doc Review** tab
2. Upload PDF or Word (max 20 MB by default)
3. **Analyse Document** → summary, cited cases, unverified citations, improvements, assessment

### Admin: manage users

1. Log in as `admin`
2. **Manage Users** → create users, enable/disable, delete
3. `POST /api/v1/auth/users` (admin only) — passwords need 8+ chars, 1 uppercase, 1 digit

### Password reset

1. **Forgot your password?** on login
2. If SMTP works → email with reset link
3. If SMTP fails → link logged at `INFO` in app logs (`docker compose logs -f app`)

---

## API Overview

Base path: `/api/v1`. Authenticated routes: `Authorization: Bearer <access_token>`.

### Authentication — `/api/v1/auth`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/login` | — | Login → tokens + user |
| `POST` | `/auth/refresh` | — | New token pair |
| `GET` | `/auth/me` | User | Current profile |
| `POST` | `/auth/change-password` | User | Change own password |
| `POST` | `/auth/forgot-password` | — | Request reset |
| `POST` | `/auth/reset-password` | — | Complete reset |
| `POST` | `/auth/users` | Admin | Create user |
| `GET` | `/auth/users` | Admin | List users |
| `PATCH` | `/auth/users/{id}` | Admin | Update user |
| `DELETE` | `/auth/users/{id}` | Admin | Delete user |

### Chat — `/api/v1/chat`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/chat/ask` | User | **SSE** stream legal answer |
| `POST` | `/chat/sessions` | User | Create session |
| `GET` | `/chat/sessions` | User | List active sessions |
| `GET` | `/chat/sessions/{id}/messages` | User | Message history |
| `DELETE` | `/chat/sessions/{id}` | User | Soft-delete session |

**SSE event types:** `status`, `token`, `citations`, `done` — see [documentation.md § Streaming](./documentation.md#11-streaming-architecture).

### Documents — `/api/v1/documents`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/documents/review` | User | Upload file → structured review |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | `{"status":"ok","version":"2.0.0"}` |
| `GET` | `/` | SPA (same as `/login`) |

---

## Data Model

Tables are created on startup via `Base.metadata.create_all()` (no Alembic migrations shipped yet).

```
users
  id (UUID PK), email (unique), full_name, hashed_password,
  role (admin|user), is_active, is_verified, created_at, updated_at

password_reset_tokens
  id, user_id (FK), token (unique), expires_at, used

chat_sessions
  id, user_id (FK), title, session_type (legal_research|document_review),
  is_active, timestamps

chat_messages
  id, session_id (FK), role (user|assistant), content,
  citations (JSONB, nullable), timestamps
```

Relationships: user → many sessions (cascade delete); session → many messages (cascade delete).

---

## Security

| Concern | Implementation |
|---------|----------------|
| Passwords | bcrypt, 12 rounds (`BCRYPT_ROUNDS`) |
| API auth | HS256 JWT access (60 min) + refresh (7 days) |
| Password reset | 30 min single-use JWT + DB `used` flag |
| Authorization | `AdminUser` dependency blocks non-admins |
| Uploads | Extension whitelist + max size |
| Prompt safety | Off-limits + jailbreak filters; identity lock in system prompt |
| CORS | `ALLOWED_ORIGINS` only |
| Container | Non-root `sankofa` user in Docker image |
| Errors | Global handler hides stack traces in production |

---

## Operations & Troubleshooting

### Useful commands

```bash
# Follow app logs
docker compose logs -f app

# All services status
docker compose ps

# Stop stack
docker compose down

# Stop and wipe volumes (destructive — loses DB data)
docker compose down -v

# Rebuild app only
docker compose up --build -d app

# Shell into app container
docker exec -it sankofa_app /bin/sh

# PostgreSQL CLI
docker exec -it sankofa_db psql -U sankofa -d sankofa_db
```

### Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| App exits on start | Missing `MISTRAL_API_KEY` or `PINECONE_API_KEY` | Set in `.env` |
| Empty / weak answers | Pinecone index empty or wrong name | Verify index & namespace |
| 401 on chat | Expired token | Re-login; check refresh flow |
| No reset email | SMTP relay blocked | Check logs for reset URL |
| CORS errors | Origin not in `ALLOWED_ORIGINS` | Add your frontend URL |

### Volumes (Docker)

| Volume | Purpose |
|--------|---------|
| `postgres_data` | Database files |
| `redis_data` | Redis persistence |
| `uploads` | `/tmp/sankofa_uploads` in container |

---

## Legacy Files

The repository root still contains an **older prototype** from before the `app/` package consolidation:

| File | Notes |
|------|-------|
| `main.py` | Monolithic FastAPI RAG API — use `app.main` instead |
| `app.py` + `templates/` | Flask UI on port 5000 — superseded by integrated SPA |
| `start_all.sh` | Starts legacy `main.py` + Flask — not needed with Docker or `uvicorn app.main:app` |
| `test.py` | Manual Pinecone query script |

**Do not run `start_all.sh` or `uvicorn main:app` for production.** Use `app.main:app` only.

---

## Disclaimer

Sankofa is a legal **research** tool. Responses do not constitute legal advice. Always consult a qualified legal practitioner admitted to practice in Zimbabwe before acting on any information from this system.

---

## Further Reading

- **[documentation.md](./documentation.md)** — Full API payloads, RAG prompts, citation rules, Docker stages, performance tuning, guardrail lists
- **Interactive API docs** — Set `DEBUG=true`, run locally, visit `/api/docs`
