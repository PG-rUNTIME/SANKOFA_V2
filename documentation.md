# Sankofa Legal AI — Technical Documentation

**Version:** 2.0.0
**Stack:** FastAPI · PostgreSQL · Pinecone · Mistral AI · Redis · Docker
**Purpose:** Production-grade AI-powered Zimbabwean legal research assistant

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Technology Stack](#4-technology-stack)
5. [Environment Configuration](#5-environment-configuration)
6. [Database Models](#6-database-models)
7. [API Reference](#7-api-reference)
8. [RAG Engine](#8-rag-engine)
9. [Document Review Pipeline](#9-document-review-pipeline)
10. [Authentication & Security](#10-authentication--security)
11. [Streaming Architecture](#11-streaming-architecture)
12. [Frontend (Single-Page Application)](#12-frontend-single-page-application)
13. [Email Service](#13-email-service)
14. [Docker & Deployment](#14-docker--deployment)
15. [Performance Optimisations](#15-performance-optimisations)
16. [Guardrails & Safety](#16-guardrails--safety)
17. [Error Handling](#17-error-handling)
18. [Default Credentials & First Login](#18-default-credentials--first-login)

---

## 1. System Overview

Sankofa is a specialised AI research assistant for Zimbabwean law. It combines:

- **Retrieval-Augmented Generation (RAG):** Queries a Pinecone vector database of Zimbabwean statutes and case law, retrieves the most relevant documents, then passes them as grounded context to the Mistral LLM.
- **Streaming responses:** Answers are streamed word-by-word to the browser using Server-Sent Events (SSE) so users see output immediately as it is generated.
- **Document Review:** Users can upload PDF or Word documents; the system extracts citations, verifies them against the legal database, and produces a structured legal review.
- **Multi-user access control:** Role-based authentication (admin/user) with JWT tokens and a full admin panel for user management.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (SPA)                         │
│  index.html — vanilla JS, marked.js, SSE consumer           │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI Application                         │
│  app/main.py — lifespan, middleware, router mount            │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  auth router │  │  chat router │  │ documents router  │  │
│  │  /api/v1/auth│  │ /api/v1/chat │  │ /api/v1/documents │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
│         │                 │                    │             │
│  ┌──────▼─────────────────▼────────────────────▼──────────┐ │
│  │              Services Layer                              │ │
│  │  rag_engine.py · document_review.py · email_service.py  │ │
│  └──────┬──────────────────┬───────────────────────────────┘ │
│         │                  │                                  │
│  ┌──────▼──────┐   ┌───────▼────────┐                        │
│  │  PostgreSQL │   │   Mistral AI   │                        │
│  │  (asyncpg)  │   │  (LLM + embed) │                        │
│  └─────────────┘   └───────┬────────┘                        │
│                            │                                  │
│                    ┌───────▼────────┐                        │
│                    │    Pinecone    │                        │
│                    │  (vector DB)   │                        │
│                    └────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
         ┌──────────┐
         │  Redis   │  (configured, reserved for rate-limiting)
         └──────────┘
```

### Request Flow (Chat)

```
User types question
       │
       ▼
[1] FastAPI receives POST /api/v1/chat/ask
       │
       ▼
[2] JWT validated → user resolved from DB
       │
       ▼
[3] StreamingResponse returned immediately (HTTP connection opens)
       │
       ▼
[4] Inside generator:
    ├─ Fast guardrail checks (off-limits, jailbreak) — pure Python, <1ms
    ├─ DB: get/create ChatSession, load history, persist user message
    ├─ SSE: emit {"type":"status","content":"thinking"}
    ├─ Parallel async: classify_intent() + retrieve_documents()
    │    ├─ Mistral router LLM → SEARCH | GREETING | NO_SEARCH
    │    └─ Pinecone vector search (sync in threadpool executor)
    ├─ SSE: emit {"type":"status","content":"retrieving"}
    ├─ SSE: emit {"type":"citations","data":[...]}  (if docs found)
    ├─ SSE: emit {"type":"status","content":"verifying"}
    ├─ Stream Mistral LLM tokens → SSE: {"type":"token","content":"..."}
    ├─ At token #60: emit {"type":"status","content":"finalising"}
    └─ SSE: emit {"type":"done","session_id":"...","elapsed_ms":4210}
       │
       ▼
[5] DB: persist assistant message + commit
[6] Browser: swaps raw text → rendered markdown, appends ⏱ time badge
```

---

## 3. Project Structure

```
legal_rag_app/
├── app/
│   ├── main.py                  # FastAPI factory, lifespan, routes, SPA serving
│   ├── api/
│   │   ├── auth.py              # Login, register, user CRUD, password reset
│   │   ├── chat.py              # Streaming chat, session management
│   │   └── documents.py         # Document upload and review
│   ├── core/
│   │   ├── config.py            # Pydantic-settings (reads .env)
│   │   ├── database.py          # Async SQLAlchemy engine, session factory, Base
│   │   ├── deps.py              # FastAPI dependencies: CurrentUser, AdminUser, DBSession
│   │   └── security.py          # bcrypt hashing, JWT create/decode
│   ├── models/
│   │   ├── user.py              # User, PasswordResetToken ORM models
│   │   └── session.py           # ChatSession, ChatMessage ORM models
│   ├── services/
│   │   ├── rag_engine.py        # Retrieval, intent routing, streaming generators
│   │   ├── document_review.py   # Text extraction, citation regex, LLM review
│   │   ├── email_service.py     # SMTP password-reset emails
│   │   └── schemas.py           # All Pydantic request/response schemas
│   ├── static/                  # Static assets (CSS, images)
│   └── templates/
│       └── index.html           # Single-page application (all JS inline)
├── Dockerfile                   # Multi-stage build (builder + runtime)
├── docker-compose.yml           # App + PostgreSQL + Redis
├── requirements.txt             # Python dependencies
├── .env                         # Secrets (git-ignored)
└── .gitignore
```

---

## 4. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web framework | FastAPI | 0.115.5 |
| ASGI server | Uvicorn + uvloop + httptools | 0.32.1 |
| ORM | SQLAlchemy async | 2.0.36 |
| Database driver | asyncpg | 0.30.0 |
| Database | PostgreSQL | 16 |
| Cache | Redis | 7 |
| LLM | Mistral AI (`mistral-large-latest`) | via langchain-mistralai 0.2.4 |
| Embeddings | Mistral (`mistral-embed`, 1024-dim) | via MistralAIEmbeddings |
| Vector store | Pinecone | 5.4.2 |
| LLM framework | LangChain | 0.3.13 |
| Auth | python-jose (JWT) + bcrypt | 3.3.0 / 4.2.1 |
| Settings | pydantic-settings | 2.7.0 |
| PDF parsing | pypdf | 5.1.0 |
| Word parsing | python-docx | 1.1.2 |
| Templates | Jinja2 | 3.1.4 |
| Container | Docker + Docker Compose | — |
| Frontend markdown | marked.js | 12 (CDN) |

---

## 5. Environment Configuration

All configuration is loaded from `.env` via `app/core/config.py` using `pydantic-settings`. The file is git-ignored; never commit it.

### Full `.env` Reference

```ini
# ── Application ───────────────────────────────────────────────────────────────
APP_NAME=Sankofa Legal AI
DEBUG=false
ENVIRONMENT=production

# ── Security ──────────────────────────────────────────────────────────────────
# Must be a long random string (≥64 chars). Generate with:
#   python -c "import secrets; print(secrets.token_urlsafe(64))"
SECRET_KEY=<your-secret-key>
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7
PASSWORD_RESET_TOKEN_EXPIRE_MINUTES=30

# ── Default Admin (created on first startup if no admin exists) ───────────────
ADMIN_EMAIL=admin@sankofa.ai
ADMIN_PASSWORD=Admin@123456        # Change immediately after first login

# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_DB=sankofa_db
POSTGRES_USER=sankofa
POSTGRES_PASSWORD=sankofa_pass
DATABASE_URL=postgresql+asyncpg://sankofa:sankofa_pass@db:5432/sankofa_db

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL=redis://redis:6379/0

# ── Mistral AI ────────────────────────────────────────────────────────────────
MISTRAL_API_KEY=<your-mistral-api-key>
MISTRAL_CHAT_MODEL=mistral-large-latest
MISTRAL_EMBED_MODEL=mistral-embed
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_INDEX_NAME=knowledge-base
PINECONE_NAMESPACE=default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBED_DIM=1024
RETRIEVER_TOP_K=5

# ── Email (Liquid Telecom unauthenticated relay) ──────────────────────────────
SMTP_HOST=relay.liquidtelecom.net
SMTP_PORT=25
# SMTP_USER and SMTP_PASSWORD are optional — leave blank for unauthenticated relay
EMAIL_FROM=noreply@sankofa-legal.ai
EMAIL_FROM_NAME=Sankofa Legal AI

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS=["http://localhost","http://localhost:8000","http://127.0.0.1:8000"]

# ── File Uploads ──────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB=20
```

### Settings Class (app/core/config.py)

The `Settings` class uses `pydantic-settings` with `lru_cache` so the singleton is only instantiated once per process. The `DATABASE_URL` and `REDIS_URL` are overridden by docker-compose `environment` block to use Docker service hostnames (`db`, `redis`) instead of `localhost`.

---

## 6. Database Models

All models inherit from `Base` (defined in `app/core/database.py`) which automatically adds `created_at` and `updated_at` timestamp columns.

### User

```
Table: users
─────────────────────────────────────────────────────
id            UUID (PK, default uuid4)
email         VARCHAR(255), UNIQUE, NOT NULL, indexed
full_name     VARCHAR(255), NOT NULL
hashed_password TEXT, NOT NULL
role          ENUM('admin','user'), NOT NULL, default='user'
is_active     BOOLEAN, NOT NULL, default=true
is_verified   BOOLEAN, NOT NULL, default=false
created_at    TIMESTAMPTZ, server default now()
updated_at    TIMESTAMPTZ, server default now()

Relationships:
  chat_sessions     → ChatSession[] (cascade delete)
  password_reset_tokens → PasswordResetToken[] (cascade delete)
```

### PasswordResetToken

```
Table: password_reset_tokens
─────────────────────────────────────────────────────
id          UUID (PK)
user_id     UUID (FK → users.id, ON DELETE CASCADE), indexed
token       VARCHAR(512), UNIQUE, NOT NULL, indexed
expires_at  TIMESTAMPTZ, NOT NULL
used        BOOLEAN, NOT NULL, default=false
created_at  TIMESTAMPTZ
updated_at  TIMESTAMPTZ
```

### ChatSession

```
Table: chat_sessions
─────────────────────────────────────────────────────
id           UUID (PK)
user_id      UUID (FK → users.id, ON DELETE CASCADE), indexed
title        VARCHAR(512), nullable
session_type VARCHAR(50), NOT NULL, default='legal_research'
             Values: 'legal_research' | 'document_review'
is_active    BOOLEAN, NOT NULL, default=true
created_at   TIMESTAMPTZ
updated_at   TIMESTAMPTZ

Relationships:
  messages → ChatMessage[] (cascade delete, ordered by created_at)
```

### ChatMessage

```
Table: chat_messages
─────────────────────────────────────────────────────
id          UUID (PK)
session_id  UUID (FK → chat_sessions.id, ON DELETE CASCADE), indexed
role        VARCHAR(20), NOT NULL  — 'user' | 'assistant'
content     TEXT, NOT NULL
citations   JSONB, nullable        — serialised Citation[] list
created_at  TIMESTAMPTZ
updated_at  TIMESTAMPTZ
```

Database tables are created automatically on startup via `Base.metadata.create_all()` in `app/core/database.py:init_db()`.

---

## 7. API Reference

All routes are prefixed with `/api/v1`. The API is JSON-based. Authentication uses `Authorization: Bearer <access_token>` headers.

### Authentication — `/api/v1/auth`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/login` | None | Authenticate and receive access + refresh tokens |
| `POST` | `/auth/refresh` | None | Exchange refresh token for new token pair |
| `GET`  | `/auth/me` | User | Get current user profile |
| `POST` | `/auth/change-password` | User | Change own password |
| `POST` | `/auth/forgot-password` | None | Request password reset email |
| `POST` | `/auth/reset-password` | None | Confirm password reset with token |
| `POST` | `/auth/users` | Admin | Create a new user account |
| `GET`  | `/auth/users` | Admin | List all users |
| `PATCH`| `/auth/users/{id}` | Admin | Update user (name, active status) |
| `DELETE`| `/auth/users/{id}` | Admin | Permanently delete a user |

#### POST /auth/login
```json
// Request
{ "email": "user@example.com", "password": "Password1" }

// Response 200
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "full_name": "Jane Doe",
    "role": "user",
    "is_active": true,
    "is_verified": true,
    "created_at": "2026-03-04T10:00:00Z"
  }
}
```

#### POST /auth/users (Admin)
```json
// Request
{
  "email": "newuser@example.com",
  "full_name": "New User",
  "password": "Password1",
  "role": "user"     // "user" | "admin"
}
```

Password validation rules: minimum 8 characters, at least 1 uppercase letter, at least 1 digit.

---

### Chat — `/api/v1/chat`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/chat/ask` | User | Stream a legal research answer (SSE) |
| `POST` | `/chat/sessions` | User | Create a named chat session |
| `GET`  | `/chat/sessions` | User | List all active sessions |
| `DELETE`| `/chat/sessions/{id}` | User | Soft-delete a session (is_active=false) |
| `GET`  | `/chat/sessions/{id}/messages` | User | Get all messages in a session |
| `POST` | `/chat/sessions/{id}/expire` | User | Mark session inactive (reserved) |

#### POST /chat/ask — SSE Streaming

```json
// Request
{
  "question": "What are the grounds for divorce under the Matrimonial Causes Act?",
  "session_id": "uuid-optional"
}
```

**Response:** `Content-Type: text/event-stream`

The response is a stream of Server-Sent Events. Each event is a line prefixed `data: ` followed by a JSON object, separated by `\n\n`:

```
data: {"type":"status","content":"thinking"}

data: {"type":"status","content":"retrieving"}

data: {"type":"citations","data":[{"url":"https://...","document_title":"...","source_file":"..."}]}

data: {"type":"status","content":"verifying"}

data: {"type":"token","content":"The Matrimonial"}

data: {"type":"token","content":" Causes Act"}

data: {"type":"status","content":"finalising"}

data: {"type":"token","content":"..."}

data: {"type":"done","session_id":"uuid","elapsed_ms":4820}
```

**SSE Event Types:**

| Type | Content | Description |
|------|---------|-------------|
| `status` | `thinking` \| `retrieving` \| `verifying` \| `finalising` | Processing phase indicator |
| `token` | string | A chunk of the response text (stream these to screen) |
| `citations` | array | Citation objects to display in sources panel |
| `done` | — | Stream complete; includes `session_id` and `elapsed_ms` |

---

### Documents — `/api/v1/documents`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/documents/review` | User | Upload and review a legal document |

#### POST /documents/review

- **Content-Type:** `multipart/form-data`
- **Field:** `file` — PDF, DOCX, or DOC
- **Max size:** 20 MB (configurable via `MAX_UPLOAD_SIZE_MB`)

```json
// Response 200
{
  "summary": "This document is a lease agreement...",
  "cited_cases": ["2021 (1) ZLR 123", "HH 302-2018"],
  "non_existing_citations": ["2019 ZSC 99"],
  "areas_for_improvement": [
    "The termination clause lacks a notice period.",
    "Section 4 does not reference the Rent Regulations Act."
  ],
  "overall_assessment": "The document is generally well-drafted but requires...",
  "disclaimer": "This automated review is for research assistance only..."
}
```

---

### System

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | None | Health check — returns `{"status":"ok","version":"2.0.0"}` |

---

## 8. RAG Engine

**File:** `app/services/rag_engine.py`

The RAG engine is the core intelligence layer. It handles retrieval, intent routing, guardrails, and streaming generation.

### Singleton Instances

To avoid cold-start latency on every request, four objects are kept as module-level singletons:

| Singleton | Purpose |
|-----------|---------|
| `_llm_streaming` | `ChatMistralAI` with `streaming=True`, `max_tokens=2048` |
| `_llm_non_streaming` | `ChatMistralAI` with `streaming=False`, `max_tokens=128` (router only) |
| `_embeddings` | `MistralAIEmbeddings` using `mistral-embed` (1024-dim) |
| `_pinecone_index` | `Pinecone.Index` connected to `knowledge-base` |

### Intent Classification

```
classify_intent(question, history) → "SEARCH" | "GREETING" | "NO_SEARCH"
```

Uses a fast Mistral call (`max_tokens=128`) with `_ROUTER_SYSTEM` prompt. The router LLM classifies the question into exactly one label. Defaults to `SEARCH` on failure.

### Parallel Retrieval

```
classify_and_retrieve(question, history) → (intent, docs, veritas_note)
```

Runs intent classification and Pinecone retrieval **concurrently** using `asyncio.gather`:
- `classify_intent()` — async Mistral call
- `retrieve_documents()` — sync Pinecone call run in `loop.run_in_executor(None, ...)`
- For constitutional queries, also fetches a Veritas Zimbabwe reference note concurrently

Combined latency = `max(classify_time, retrieval_time)` instead of sequential sum.

### Retrieval

```
retrieve_documents(query, top_k=5) → List[Document]
```

1. Embeds the query using `mistral-embed` (1024-dim vector)
2. Queries Pinecone `knowledge-base` index, `default` namespace
3. Returns results sorted by relevance score (descending)
4. Each `Document` has `page_content` (chunk text) and `metadata` including `akn_url`, `document_title`, `source_file`, `chunk_index`, `_score`

### Citation Extraction

```
extract_citations(docs) → List[Citation]
```

Deduplicates documents by `akn_url`. Only documents with a valid `akn_url` are returned as citable sources. Documents without URLs are used as context only.

### Streaming Generators

**`stream_legal_answer(question, history, docs, citations, veritas_note)`**

Builds a full RAG prompt with:
- `_LEGAL_RAG_SYSTEM_TEMPLATE` — contains identity lock, response format rules, SOURCES index, LEGAL CONTEXT
- Conversation history (last 12 messages = 6 turns)
- Inline citation URLs for the LLM to embed as markdown links

Streams tokens via `chain.astream()`.

**`stream_generic_answer(question, history)`**

Uses `_GENERIC_SYSTEM` prompt for greetings and off-topic redirects. Much shorter responses, no document context.

### Prompt System

The system prompt enforces:
1. **Identity lock** — Sankofa cannot be renamed or re-assigned a different role by any user prompt
2. **Response format** — Markdown with bold headings, inline citation links, disclaimer block
3. **Citation rules** — Only cite URLs from the SOURCES list; never fabricate URLs
4. **Constitutional routing** — If the query involves the Constitution, the Veritas Zimbabwe URL is injected

---

## 9. Document Review Pipeline

**File:** `app/services/document_review.py`

### Pipeline Steps

```
review_document(filename, content_bytes)
    │
    ├─ 1. extract_text()
    │       ├─ PDF  → pypdf.PdfReader
    │       └─ DOCX → python-docx
    │
    ├─ 2. extract_raw_citations(text)
    │       └─ 9 regex patterns for Zimbabwean citations:
    │           • "2021 (1) ZLR 123"
    │           • "HH 302-2018", "SC 15-2020"
    │           • "2020 ZSC 15", "2019 ZWHHC 302"
    │           • "Party v Party"
    │           • "Section 5 of the Labour Act"
    │           • "Act No. 5 of 2007"
    │           • "Chapter 29:12"
    │
    ├─ 3. verify_citations_against_db(citations)
    │       └─ For each citation: retrieve_documents(citation, top_k=1)
    │          Verified if score > 0.75, else not_found
    │
    └─ 4. LLM review via Mistral
            └─ Structured prompt → parse SUMMARY / CITED_CASES /
               AREAS_FOR_IMPROVEMENT / OVERALL_ASSESSMENT
               → DocumentReviewResult
```

### Output Schema (DocumentReviewResult)

```python
class DocumentReviewResult(BaseModel):
    summary: str                        # 3-5 sentence overview
    cited_cases: List[str]              # All citations found in text
    non_existing_citations: List[str]   # Citations not in legal DB (score < 0.75)
    areas_for_improvement: List[str]    # Up to 5 actionable improvements
    overall_assessment: str             # Quality and legal soundness summary
    disclaimer: str                     # Standard research-only disclaimer
```

---

## 10. Authentication & Security

**Files:** `app/core/security.py`, `app/core/deps.py`, `app/api/auth.py`

### Token Architecture

Three token types, all signed with `SECRET_KEY` using `HS256`:

| Token | Claim `type` | TTL | Purpose |
|-------|-------------|-----|---------|
| Access token | `"access"` | 60 minutes | API authentication |
| Refresh token | `"refresh"` | 7 days | Renew access token |
| Password reset token | `"password_reset"` | 30 minutes | One-time password reset |

### Access Token Payload

```json
{
  "sub": "user@example.com",
  "role": "user",
  "type": "access",
  "exp": 1741234567
}
```

### Password Security

- Passwords are hashed with **bcrypt** at `rounds=12` (configurable via `BCRYPT_ROUNDS`)
- `verify_password()` uses constant-time comparison to prevent timing attacks
- Password reset tokens are stored as signed JWTs; the DB record (`PasswordResetToken`) tracks usage and expiry to prevent replay

### Dependency Injection

```python
CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser   = Annotated[User, Depends(require_admin)]     # raises 403 if not admin
DBSession   = Annotated[AsyncSession, Depends(get_db)]
```

`get_current_user` validates the Bearer JWT, checks `type == "access"`, resolves the user from DB, and verifies `is_active`.

### Password Reset Flow

```
1. POST /auth/forgot-password {email}
       ↓
2. Look up user (silently succeed if not found — prevents email enumeration)
       ↓
3. Create PasswordResetToken (JWT, 30min TTL) in DB
       ↓
4. Send email via background task (non-blocking)
       ↓
5. POST /auth/reset-password {token, new_password}
       ↓
6. Decode JWT, verify DB record exists + not used + not expired
       ↓
7. Update hashed_password, mark token used=true
```

---

## 11. Streaming Architecture

### Server (SSE)

The `/chat/ask` endpoint returns a `StreamingResponse` with `media_type="text/event-stream"`. The generator function yields events in this format:

```
data: <json>\n\n
```

Key headers:
- `Cache-Control: no-cache` — prevents proxy buffering
- `X-Accel-Buffering: no` — disables nginx proxy buffering

**Critical design:** All blocking work (DB queries, Pinecone retrieval, LLM calls) happens **inside** the generator after the HTTP response has already been initiated. This means the browser opens the connection and receives the first SSE event within milliseconds of sending the request — before any AI processing occurs.

### Client (JavaScript)

The frontend consumes the SSE stream using the native `ReadableStream` API (not `EventSource`), which allows POST requests with a body:

```javascript
const res    = await fetch('/api/v1/chat/ask', { method:'POST', ... });
const reader = res.body.getReader();
const dec    = new TextDecoder();
let buf = '';

while(true){
  const {done, value} = await reader.read();
  if(done) break;
  buf += dec.decode(value, {stream:true});

  // Split on \n\n (SSE event boundary)
  let boundary;
  while((boundary = buf.indexOf('\n\n')) !== -1){
    const raw = buf.slice(0, boundary);
    buf = buf.slice(boundary + 2);
    // parse and handle each event...
  }
}
```

**Token rendering strategy:**

During streaming, each `token` event appends to a raw `TextNode` — the cheapest possible DOM operation with zero markdown parsing overhead. When the `done` event fires, the raw text is replaced in a single pass with `marked.parse(full)` to apply full markdown formatting, then the generation time badge is appended.

**Scroll optimisation:** `requestAnimationFrame`-throttled scroll so the browser never stalls a paint frame to do layout recalculation.

---

## 12. Frontend (Single-Page Application)

**File:** `app/templates/index.html`

A self-contained single HTML file (~1700 lines) served by Jinja2. No build step, no npm, no frameworks — vanilla JavaScript with `marked.js` (CDN) for markdown rendering.

### Views

| View | Route trigger | Description |
|------|--------------|-------------|
| Auth screen | Initial load / logout | Login, forgot password, reset password forms |
| Chat view | `switchView('chat')` | Legal research chat interface (user role) |
| Document Review | `switchView('docs')` | File upload and review results (user role) |
| My Account | `switchView('account')` | Profile info, change password |
| Admin panel | `switchView('admin')` | User table: create, enable/disable, delete |

### Authentication State

```javascript
let _tok  = localStorage.getItem('sk_at')   // access token
let _rTok = localStorage.getItem('sk_rt')   // refresh token
let _user = localStorage.getItem('sk_user') // user JSON
```

Token refresh is handled automatically in `apiFetch()`: if a 401 is received, it attempts `POST /auth/refresh` with the stored refresh token, updates localStorage, and retries the original request.

### Status Cycling (Phase-Based)

While waiting for a response, a pill shows animated words cycling at ~950ms per word. The words are grouped into four phases, each triggered by the corresponding SSE status event:

| Phase | SSE trigger | Sample words |
|-------|------------|--------------|
| `thinking` | `status: thinking` | Thinking… / Processing… / Interpreting… / Decoding intent… |
| `retrieving` | `status: retrieving` | Searching database… / Fetching case law… / Sifting through cases… |
| `verifying` | `status: verifying` | Verifying sources… / Cross-referencing… / Scrutinising results… |
| `finalising` | `status: finalising` | Finalising answer… / Composing response… / Polishing answer… |

Each word transition uses a `fadeWord` CSS animation (fade in + slight upward translate).

### Admin User Management

The admin panel renders a user table from `GET /auth/users`. Each row has:
- **Disable / Enable** button → `PATCH /auth/users/{id}` with `{is_active: false/true}`
- **Delete** button → `DELETE /auth/users/{id}` with a confirmation dialog (permanent, irreversible)

### Design Tokens

The UI uses CSS custom properties for theming. Light and dark modes are supported via `[data-theme="dark"]` attribute on `<html>`, toggled with a button in the sidebar and persisted to `localStorage`.

Key tokens:
```css
--navy: #1A2C64       /* primary brand colour */
--purple: #6D28D9     /* accent / interactive */
--gold: #D97706       /* highlights */
--bg: #F1F5F9         /* page background */
--surface: #FFFFFF    /* card background */
```

---

## 13. Email Service

**File:** `app/services/email_service.py`

Sends HTML password-reset emails via SMTP.

### SMTP Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SMTP_HOST` | `relay.liquidtelecom.net` | Liquid Telecom unauthenticated relay |
| `SMTP_PORT` | `25` | Plain SMTP (no TLS) |
| `SMTP_USER` | `None` | Optional — if set, enables STARTTLS + login |
| `SMTP_PASSWORD` | `None` | Optional |
| `EMAIL_FROM` | `noreply@sankofa-legal.ai` | Sender address |

### Relay Behaviour

When `SMTP_USER` and `SMTP_PASSWORD` are `None` (the default), the service connects on port 25, sends `EHLO`, and relays the message without authentication. This is the standard behaviour for an internal unauthenticated SMTP relay.

If credentials are provided, the service will negotiate `STARTTLS` before authenticating.

### Fallback

If the SMTP send fails (network error, relay unavailable), the error is logged but **not raised** to the caller. The reset link is also logged at `INFO` level so administrators can retrieve it from server logs. This prevents SMTP failures from surfacing as errors to the end user.

---

## 14. Docker & Deployment

### Dockerfile (Multi-Stage)

**Builder stage:**
- Base: `python:3.12-slim`
- Installs `build-essential` and `libpq-dev` (required to compile `asyncpg` and `bcrypt`)
- Installs all Python packages into `/install` (separate layer)

**Runtime stage:**
- Base: `python:3.12-slim`
- Installs only `libpq5` and `curl` (runtime-only)
- Creates non-root user `sankofa` for security
- Copies installed packages from builder
- Copies `app/` directory
- Exposes port 8000
- Runs: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --loop uvloop --http httptools`

### Docker Compose Services

| Service | Image | Port | Health check |
|---------|-------|------|-------------|
| `app` (sankofa_app) | Built from Dockerfile | 8000 | `curl -f http://localhost:8000/health` |
| `db` (sankofa_db) | `postgres:16-alpine` | 5432 | `pg_isready` |
| `redis` (sankofa_redis) | `redis:7-alpine` | 6379 | `redis-cli ping` |

The `app` service depends on `db` and `redis` with `condition: service_healthy`, ensuring the database is accepting connections before the application starts.

### Volumes

| Volume | Mounted at | Purpose |
|--------|-----------|---------|
| `postgres_data` | `/var/lib/postgresql/data` | Database persistence |
| `redis_data` | `/data` | Redis persistence |
| `uploads` | `/tmp/sankofa_uploads` | Uploaded document storage |

### Deployment Commands

```bash
# First-time setup / clean rebuild
docker compose build --no-cache
docker compose up -d

# Subsequent deploys (code change)
docker compose build --no-cache
docker compose down && docker compose up -d

# View logs
docker compose logs app -f

# Check health
docker compose ps
```

### Environment Variables Override

The `docker-compose.yml` `environment` block overrides `.env` for container-specific networking:

```yaml
environment:
  DATABASE_URL: "postgresql+asyncpg://sankofa:sankofa_pass@db:5432/sankofa_db"
  REDIS_URL:    "redis://redis:6379/0"
```

This replaces `localhost` hostnames with Docker service names (`db`, `redis`).

---

## 15. Performance Optimisations

### Singleton LLM Instances

`rag_engine.py` maintains module-level singletons for all expensive objects:

```python
_llm_streaming     # ChatMistralAI(streaming=True,  max_tokens=2048)
_llm_non_streaming # ChatMistralAI(streaming=False, max_tokens=128)
_embeddings        # MistralAIEmbeddings(model="mistral-embed")
_pinecone_index    # Pinecone().Index("knowledge-base")
```

These are created once on first use and reused across all requests — eliminating the SDK initialisation cost (~200-500ms) from every request.

### Parallel Classification + Retrieval

```python
intent_task    = asyncio.create_task(classify_intent(question, history))
retrieval_task = loop.run_in_executor(None, retrieve_documents, question)
intent, docs, veritas_note = await asyncio.gather(*tasks)
```

Instead of sequential `classify → retrieve` (latency = classify + retrieve), both run concurrently (latency = max(classify, retrieve)).

### Reduced Router Context

The intent classification LLM uses `max_tokens=128` — it only needs to return one word (`SEARCH`, `GREETING`, or `NO_SEARCH`). Full `max_tokens=2048` is reserved for the answer-generation LLM.

### Conversation History Window

History is capped at the last **12 messages** (6 user + 6 assistant turns) before being sent to the LLM. This limits the context window used and reduces token cost and latency on long conversations.

### Early Guardrail Exit

Off-limits and jailbreak checks are pure Python string operations (`O(n)` where `n` = number of patterns). They run before any DB query, Pinecone call, or LLM call, returning an immediate response for blocked requests.

### Non-Legal Fast Path

`_is_clearly_non_legal()` detects obviously off-topic questions (cooking, sport, weather, etc.) and skips the entire classify+retrieve pipeline, going directly to `stream_generic_answer()`.

### Streaming Response Initiation

The HTTP response and SSE stream begin **before** any AI processing. The browser opens the connection immediately; the first status event is emitted within milliseconds. All Pinecone and Mistral work happens inside the generator after the stream is open.

### Frontend Rendering

- Token events write to a raw `TextNode` (zero parsing, zero DOM traversal)
- `scrollTop` is deferred to `requestAnimationFrame` to never block a paint frame
- Markdown is parsed exactly once — at the `done` event — not once per token

### GZip Middleware

`GZipMiddleware(minimum_size=1000)` compresses all responses ≥ 1KB (HTML, JSON). SSE streams are excluded automatically (chunked transfer encoding).

---

## 16. Guardrails & Safety

### Off-Limits Topics

Requests containing phrases related to facilitating unlawful activity are rejected immediately:

```python
OFF_LIMITS_TOPICS = {
    "how to commit", "how to murder", "how to kill", "how to assault",
    "how to defraud", "how to evade", "how to launder", "how to hack",
    "how to stalk", "how to bribe",
}
```

Returns `OFF_LIMITS_RESPONSE` — a polite refusal that invites legitimate legal questions.

### Jailbreak / Role-Play Detection

30+ patterns detect attempts to override the system's identity or instructions:

```python
JAILBREAK_PATTERNS = {
    "pretend you are", "act as a", "assume you are", "roleplay as",
    "forget your instructions", "ignore your instructions",
    "you have no restrictions", "dan mode", "developer mode",
    "as a teacher", "as a doctor", "as a hacker", ...
}
```

Returns `JAILBREAK_RESPONSE` — affirms Sankofa's fixed identity and invites legal questions.

### Identity Lock in System Prompt

The `_LEGAL_RAG_SYSTEM_TEMPLATE` contains an **IDENTITY & SCOPE** section that explicitly states the assistant's identity is fixed and cannot be changed by any user instruction, regardless of how it is framed (hypothetical, role-play, "assume", "pretend", etc.).

### Legal Domain Guardrails

- `_is_clearly_non_legal()` — fast keyword check; skips retrieval for obviously off-topic questions
- `LEGAL_DOMAIN_KEYWORDS` — 30+ legal terms; if any match, the question is treated as potentially legal even if non-legal keywords also match
- `NON_LEGAL_TOPICS` — 15 clearly off-topic domains (recipes, sport, weather, etc.)

### Citation Integrity

The LLM is instructed (and constrained by the SOURCES list in the prompt) to:
- Only use exact URLs from the retrieved documents
- Never fabricate case names, statutes, or URLs
- If `NO_SOURCES_FOUND`, answer from knowledge without inventing citations

### Legal Disclaimer

Every legal response is required (by the system prompt) to end with:

> *Disclaimer: This response is for legal research and educational purposes only. It does not constitute legal advice. You should consult a qualified legal practitioner admitted to practice in Zimbabwe before acting on any of the information provided.*

---

## 17. Error Handling

### API Layer

- All endpoints return structured JSON errors: `{"detail": "message"}`
- A global `Exception` handler in `main.py` catches unhandled exceptions and returns a 500 with a generic message (no stack trace exposure in production)
- 401 Unauthorized: returned for missing/invalid/expired JWT
- 403 Forbidden: returned for insufficient role or disabled account

### Streaming Errors

If an exception occurs inside the `generate()` function after the stream has started, the error cannot change the HTTP status code (already 200). Instead, a `token` event is emitted with a human-readable error message, and the stream ends with a `done` event.

### Database Errors

The `get_db()` dependency uses a try/except/finally pattern: commits on success, rolls back on exception, always closes the session.

### Pinecone / Mistral Errors

Both are caught at the service layer:
- Pinecone errors: logged, empty document list returned (graceful degradation)
- Mistral router errors: logged, defaults to `SEARCH` intent
- Mistral streaming errors: caught in the `generate()` try/except, error message emitted as a token

---

## 18. Default Credentials & First Login

On first startup, if no admin user exists in the database, one is created automatically:

| Field | Default Value |
|-------|--------------|
| Email | `admin@sankofa.ai` (override via `ADMIN_EMAIL` env var) |
| Password | `Admin@123456` (override via `ADMIN_PASSWORD` env var) |
| Role | `admin` |

**Change this password immediately after first login** via the My Account → Change Password form.

The admin account has access to:
- The Admin panel (User Management) — create, enable, disable, and delete user accounts
- All API endpoints requiring `AdminUser` dependency

Regular users can only access their own chat sessions and documents. They cannot view or modify other users' data.

---

*Documentation generated for Sankofa Legal AI v2.0.0*
