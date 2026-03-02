"""
Chat API: session management, streaming legal research, message history.
"""
from __future__ import annotations

import json
import logging
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
    # Verify ownership
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


# ── Streaming Chat ────────────────────────────────────────────────────────────

@router.post("/ask")
async def ask(
    body: ChatQueryRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> StreamingResponse:
    """
    Streaming endpoint. Sends Server-Sent Events (SSE):
      data: {"type": "token", "content": "..."}
      data: {"type": "citations", "data": [...]}
      data: {"type": "done"}
    """
    question = body.question.strip()

    # ── Guardrails ────────────────────────────────────────────────────────────
    if rag_engine._is_off_limits(question):
        async def off_limits_stream() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type': 'token', 'content': rag_engine.OFF_LIMITS_RESPONSE})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(off_limits_stream(), media_type="text/event-stream")

    # ── Get or create session ────────────────────────────────────────────────
    session: ChatSession | None = None
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

    # ── Load conversation history ─────────────────────────────────────────────
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

    # ── Persist user message ──────────────────────────────────────────────────
    user_msg = ChatMessage(
        session_id=session.id,
        role="user",
        content=question,
    )
    db.add(user_msg)
    await db.flush()

    # ── Parallel: classify intent + retrieve documents ────────────────────────
    # Run both operations concurrently so time = max(classify, retrieve)
    # instead of classify + retrieve sequentially.
    if rag_engine._is_clearly_non_legal(question):
        intent = "NO_SEARCH"
        docs = []
        veritas_note = ""
    else:
        intent, docs, veritas_note = await rag_engine.classify_and_retrieve(question, history)

    logger.info("Intent for session %s: %s | docs: %d", session.id, intent, len(docs))

    # Extract citations from already-retrieved docs
    citation_objects = rag_engine.extract_citations(docs) if intent == "SEARCH" else []
    citations_payload = [c.model_dump() for c in citation_objects]

    # ── Build streaming response ───────────────────────────────────────────────
    async def generate() -> AsyncGenerator[str, None]:
        full_answer = ""

        try:
            # Immediately signal that processing has started
            yield f"data: {json.dumps({'type': 'status', 'content': 'thinking'})}\n\n"

            if intent in ("GREETING", "NO_SEARCH"):
                async for chunk in rag_engine.stream_generic_answer(question, history):
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            else:
                # Emit citations metadata before streaming so the UI can show
                # the source panel while text is still arriving
                if citations_payload:
                    yield f"data: {json.dumps({'type': 'citations', 'data': citations_payload})}\n\n"

                async for chunk in rag_engine.stream_legal_answer(
                    question, history, docs, citation_objects, veritas_note
                ):
                    full_answer += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        except Exception as exc:
            logger.error("Streaming error for session %s: %s", session.id, exc)
            error_msg = (
                "I am currently experiencing a technical difficulty. "
                "Please try again in a moment."
            )
            yield f"data: {json.dumps({'type': 'token', 'content': error_msg})}\n\n"
            full_answer = error_msg

        finally:
            # Persist assistant message
            try:
                ai_msg = ChatMessage(
                    session_id=session.id,
                    role="assistant",
                    content=full_answer,
                    citations=citations_payload if citations_payload else None,
                )
                db.add(ai_msg)
                # Update session title if it's the first real exchange
                if not session.title or session.title == question[:100]:
                    session.title = question[:80]
                await db.commit()
            except Exception as persist_exc:
                logger.error("Failed to persist message: %s", persist_exc)

            yield f"data: {json.dumps({'type': 'done', 'session_id': str(session.id)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
