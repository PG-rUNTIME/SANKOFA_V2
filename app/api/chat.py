"""
Chat API: session management, streaming legal research, message history.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator, List

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from app.core.deps import CurrentUser, DBSession
from app.models.session import ChatMessage, ChatSession
from app.services import rag_engine
from app.services.schemas import (
    ChatMessageResponse,
    ChatQueryRequest,
    ChatSessionResponse,
    NewSessionRequest,
)

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


# ── Sessions ──────────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: NewSessionRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> ChatSessionResponse:
    session = ChatSession(
        user_id=current_user.id,
        session_type=body.session_type,
        title=body.title,
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        session_type=session.session_type,
        is_active=session.is_active,
        created_at=session.created_at,
        message_count=0,
    )


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_sessions(current_user: CurrentUser, db: DBSession) -> List[ChatSessionResponse]:
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id, ChatSession.is_active.is_(True))
        .order_by(ChatSession.created_at.desc())
    )
    sessions = result.scalars().all()

    out = []
    for s in sessions:
        count_result = await db.execute(
            select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == s.id)
        )
        count = count_result.scalar_one()
        out.append(ChatSessionResponse(
            id=s.id,
            title=s.title,
            session_type=s.session_type,
            is_active=s.is_active,
            created_at=s.created_at,
            message_count=count,
        ))
    return out


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_session(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
) -> Response:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    session.is_active = False
    await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Messages ──────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_messages(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
) -> List[ChatMessageResponse]:
    sess_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
        )
    )
    if not sess_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return [ChatMessageResponse.model_validate(m) for m in msg_result.scalars().all()]


# ── Session expiry ────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/expire", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def expire_session(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
) -> Response:
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id,
            ChatSession.is_active.is_(True),
        )
    )
    session = result.scalar_one_or_none()
    if session:
        session.is_active = False
        await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Streaming Chat ────────────────────────────────────────────────────────────

@router.post("/ask")
async def ask(
    body: ChatQueryRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> StreamingResponse:
    """
    Streaming endpoint — HTTP response starts immediately, all heavy work
    (retrieval, classification) happens inside the generator so the browser
    receives the first SSE byte with zero pre-processing delay.

    SSE events:
      {"type": "status",   "content": "thinking|retrieving|verifying|finalising"}
      {"type": "token",    "content": "..."}
      {"type": "citations","data":    [...]}
      {"type": "done",     "session_id": "...", "elapsed_ms": 1234}
    """
    question = body.question.strip()

    # ── Fast guardrails (pure Python, no I/O) ─────────────────────────────────
    if rag_engine._is_off_limits(question):
        async def _off() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type':'token','content':rag_engine.OFF_LIMITS_RESPONSE})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_off(), media_type="text/event-stream",
                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    if rag_engine._is_jailbreak(question):
        async def _jb() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type':'token','content':rag_engine.JAILBREAK_RESPONSE})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_jb(), media_type="text/event-stream",
                                  headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    async def generate() -> AsyncGenerator[str, None]:
        full_answer = ""
        t_start = time.monotonic()
        session = None

        try:
            # ── Emit first byte immediately so the browser opens the stream ──
            yield f"data: {json.dumps({'type':'status','content':'thinking'})}\n\n"

            # ── DB: get/create session ────────────────────────────────────────
            if body.session_id:
                sess_result = await db.execute(
                    select(ChatSession).where(
                        ChatSession.id == body.session_id,
                        ChatSession.user_id == current_user.id,
                        ChatSession.is_active.is_(True),
                    )
                )
                session = sess_result.scalar_one_or_none()

            if not session:
                session = ChatSession(
                    user_id=current_user.id,
                    session_type="legal_research",
                    title=question[:100],
                )
                db.add(session)
                await db.flush()
                await db.refresh(session)

            # ── Load history ──────────────────────────────────────────────────
            hist_result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session.id)
                .order_by(ChatMessage.created_at.asc())
            )
            history = [
                {"role": m.role, "content": m.content}
                for m in hist_result.scalars().all()
            ]
            history.append({"role": "user", "content": question})

            # ── Persist user message ──────────────────────────────────────────
            db.add(ChatMessage(session_id=session.id, role="user", content=question))
            await db.flush()

            # ── Classify + retrieve (parallel, inside stream) ─────────────────
            if rag_engine._is_clearly_non_legal(question):
                intent = "NO_SEARCH"
                docs = []
                veritas_note = ""
            else:
                yield f"data: {json.dumps({'type':'status','content':'retrieving'})}\n\n"
                intent, docs, veritas_note = await rag_engine.classify_and_retrieve(
                    question, history
                )

            logger.info("Intent: %s | docs: %d", intent, len(docs))

            citation_objects = rag_engine.extract_citations(docs) if intent == "SEARCH" else []
            citations_payload = [c.model_dump() for c in citation_objects]

            if intent in ("GREETING", "NO_SEARCH"):
                async for chunk in rag_engine.stream_generic_answer(question, history):
                    full_answer += chunk
                    yield f"data: {json.dumps({'type':'token','content':chunk})}\n\n"
            else:
                if citations_payload:
                    yield f"data: {json.dumps({'type':'citations','data':citations_payload})}\n\n"

                yield f"data: {json.dumps({'type':'status','content':'verifying'})}\n\n"

                token_count = 0
                async for chunk in rag_engine.stream_legal_answer(
                    question, history, docs, citation_objects, veritas_note
                ):
                    full_answer += chunk
                    token_count += 1
                    yield f"data: {json.dumps({'type':'token','content':chunk})}\n\n"
                    if token_count == 60:
                        yield f"data: {json.dumps({'type':'status','content':'finalising'})}\n\n"

        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            err = "I am experiencing a technical difficulty. Please try again."
            yield f"data: {json.dumps({'type':'token','content':err})}\n\n"
            full_answer = err

        finally:
            elapsed_ms = round((time.monotonic() - t_start) * 1000)
            if session:
                try:
                    db.add(ChatMessage(
                        session_id=session.id,
                        role="assistant",
                        content=full_answer,
                        citations=([c.model_dump() for c in citation_objects]
                                   if 'citation_objects' in dir() and citation_objects else None),
                    ))
                    if not session.title or session.title == question[:100]:
                        session.title = question[:80]
                    await db.commit()
                except Exception as e:
                    logger.error("Failed to persist message: %s", e)

            sid = str(session.id) if session else ""
            yield f"data: {json.dumps({'type':'done','session_id':sid,'elapsed_ms':elapsed_ms})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
