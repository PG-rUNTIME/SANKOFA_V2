# Sankofa Legal AI

Production-grade Zimbabwean legal research AI with streaming responses, multi-turn memory, document review, and full user authentication.

---

## Features

- **Legal Research Chat** — multi-turn conversations, streaming SSE responses, cited sources
- **Document Review** — upload PDF / DOCX for citation verification and AI analysis
- **Authentication** — JWT login, admin user management, password reset
- **PostgreSQL** — persistent users, sessions, messages
- **Docker** — single-command full-stack deployment

---

## Quick Start (Docker)

### 1. Create your environment file

```bash
cp .env.example .env
```

Edit `.env` — the only required changes are:

| Variable | What to set |
|----------|-------------|
| `SECRET_KEY` | Any long random string (min 64 chars) |
| `ADMIN_EMAIL` | Your admin email |
| `ADMIN_PASSWORD` | Strong password (change after first login) |
| `MISTRAL_API_KEY` | Your Mistral AI key |
| `PINECONE_API_KEY` | Your Pinecone key |
| `PINECONE_INDEX_NAME` | Your Pinecone index |

### 2. Build and start

```bash
docker compose up --build -d
```

Starts PostgreSQL 16, Redis 7, and the Sankofa app — all wired together automatically.

### 3. Open the app

```
http://localhost:8000
```

### 4. First login

| Field | Default |
|-------|---------|
| Email | `admin@sankofa.ai` |
| Password | `Admin@123456` |

**Change the admin password immediately after first login.**

---

## Adding Users (Admin Only)

1. Log in as admin
2. Click **Manage Users** in the sidebar
3. Fill the **Add New User** form and submit
4. Share credentials with the new user

---

## Using the App

### Legal Research

- Select the **Research** tab
- Type any question about Zimbabwe law and press Enter
- Responses stream token by token with cited sources below each answer
- Sessions are automatically saved; previous sessions appear in the sidebar

### Document Review

- Select the **Doc Review** tab
- Upload a PDF or Word document (max 20 MB)
- Click **Analyse Document**
- Receive: summary, verified citations, unverified/non-existent citations flagged in red, areas for improvement, overall assessment

### Password Reset

- Click **Forgot your password?** on the login screen
- Enter your email
- If SMTP is configured, a reset link is emailed
- If SMTP is not configured, the reset link is printed to the app container log

---

## Useful Commands

```bash
# View app logs
docker compose logs -f app

# Stop everything
docker compose down

# Stop and delete all data (destructive)
docker compose down -v

# Rebuild only the app
docker compose up --build -d app

# Shell into app container
docker exec -it sankofa_app /bin/sh

# PostgreSQL shell
docker exec -it sankofa_db psql -U sankofa -d sankofa_db
```

---

## Local Development (no Docker)

```bash
# 1. Create venv
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start local postgres + redis, then update .env:
#    DATABASE_URL=postgresql+asyncpg://sankofa:sankofa_pass@localhost:5432/sankofa_db
#    REDIS_URL=redis://localhost:6379/0

# 3. Run
uvicorn app.main:app --reload --port 8000
```

---

## Project Structure

```
app/
  main.py               FastAPI app factory + startup
  core/
    config.py           Settings (reads .env)
    database.py         Async SQLAlchemy engine
    security.py         bcrypt + JWT
    deps.py             FastAPI DI: auth guards
  models/
    user.py             User, PasswordResetToken
    session.py          ChatSession, ChatMessage
  api/
    auth.py             Login, register, password reset
    chat.py             Streaming chat + session management
    documents.py        Document upload + review
  services/
    rag_engine.py       Pinecone retrieval + LLM streaming
    document_review.py  PDF/DOCX parse + citation verification
    email_service.py    Password reset email
    schemas.py          All Pydantic schemas
  templates/
    index.html          Full SPA frontend (no build step)

Dockerfile              Multi-stage Docker build
docker-compose.yml      App + PostgreSQL + Redis
.env.example            Config template
requirements.txt        Python dependencies
```

---

## Security

- Passwords: bcrypt (12 rounds)
- Access tokens: 60-minute JWT; refresh tokens: 7-day JWT
- Password reset tokens: 30-minute single-use
- File uploads: type + size validated server-side
- Guardrails: off-limits queries rejected before reaching the LLM
- Admin routes: role-checked via FastAPI dependency injection
- CORS: restricted to `ALLOWED_ORIGINS`

---

## Disclaimer

Sankofa is a legal **research** tool. Responses do not constitute legal advice. Always consult a qualified legal practitioner before acting on any information from this system.
