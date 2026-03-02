"""
Core RAG engine: retrieval, intent routing, streaming generation.
- Parallel retrieval + intent classification to cut time-to-first-token
- Markdown output with inline citation hyperlinks
- Conversation memory (multi-turn)
- Guardrails
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pinecone import Pinecone

from app.core.config import settings
from app.services.schemas import Citation

logger = logging.getLogger(__name__)

# ── Guardrails ────────────────────────────────────────────────────────────────

LEGAL_DOMAIN_KEYWORDS = {
    "law", "legal", "court", "case", "act", "statute", "regulation", "constitution",
    "judgment", "penalty", "rights", "offence", "contract", "crime", "sentence",
    "tribunal", "appeal", "plaintiff", "defendant", "prosecution", "bail", "damages",
    "legislation", "section", "clause", "ordinance", "gazette", "zimbabwe",
    "high court", "supreme court", "magistrate", "counsel", "attorney", "advocate",
    "probate", "succession", "divorce", "custody", "tenancy", "lease", "mortgage",
    "defamation", "negligence", "contempt", "injunction", "interdict",
}

NON_LEGAL_TOPICS = {
    "recipe", "cooking", "sport", "football", "music", "entertainment",
    "weather", "stock market", "cryptocurrency", "dating", "travel",
    "celebrity", "fashion", "gaming", "movie", "restaurant",
}

OFF_LIMITS_TOPICS = {
    "how to commit", "how to murder", "how to kill", "how to assault",
    "how to defraud", "how to evade", "how to launder", "how to hack",
    "how to stalk", "how to bribe",
}


CONSTITUTION_KEYWORDS = {
    "constitution", "constitutional", "bill of rights", "section 69", "section 70",
    "section 48", "section 49", "section 50", "section 51", "section 52", "section 53",
    "section 54", "section 55", "section 56", "section 57", "section 58", "section 60",
    "section 61", "section 62", "section 63", "section 64", "section 65", "section 67",
    "section 68", "section 71", "section 72", "section 73", "section 74", "section 75",
    "section 76", "section 77", "section 78", "section 79", "section 80", "section 81",
    "chapter 4", "fundamental rights", "declaration of rights", "constitutional rights",
    "constitutional provision", "constitutional law", "constitutionally", "unconstitutional",
}

def _is_off_limits(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in OFF_LIMITS_TOPICS)


def _is_clearly_non_legal(question: str) -> bool:
    q = question.lower()
    if any(kw in q for kw in LEGAL_DOMAIN_KEYWORDS):
        return False
    return any(kw in q for kw in NON_LEGAL_TOPICS)


def _is_constitution_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in CONSTITUTION_KEYWORDS)


async def _fetch_veritas_note() -> str:
    """
    Return a short advisory directing the LLM to reference Veritas Zimbabwe
    for constitutional provisions. We do NOT scrape the live page (to avoid
    latency); instead we embed a reliable canonical reference.
    """
    return (
        "CONSTITUTIONAL REFERENCE (Veritas Zimbabwe): "
        "For the text of the Constitution of Zimbabwe Amendment (No. 20) Act 2013 "
        "and all constitutional provisions, the authoritative source is "
        "https://www.veritaszim.net/ (Veritas Trust Zimbabwe). "
        "When citing constitutional sections, direct the user to verify the exact "
        "text at https://www.veritaszim.net/ as the primary source."
    )


# ── LLM factory ───────────────────────────────────────────────────────────────

def _build_llm(streaming: bool = False) -> ChatMistralAI:
    return ChatMistralAI(
        model=settings.MISTRAL_CHAT_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        mistral_api_key=settings.MISTRAL_API_KEY,
        streaming=streaming,
    )


# ── Singletons ────────────────────────────────────────────────────────────────

_embeddings: Optional[MistralAIEmbeddings] = None
_pinecone_index: Optional[Any] = None


def _get_embeddings() -> MistralAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = MistralAIEmbeddings(
            model=settings.MISTRAL_EMBED_MODEL,
            mistral_api_key=settings.MISTRAL_API_KEY,
        )
    return _embeddings


def _get_pinecone_index() -> Optional[Any]:
    global _pinecone_index
    if _pinecone_index is None:
        try:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            _pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info("Pinecone index connected: %s", settings.PINECONE_INDEX_NAME)
        except Exception as exc:
            logger.error("Pinecone connection failed: %s", exc)
    return _pinecone_index


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_documents(query: str, top_k: int | None = None) -> List[Document]:
    """Embed the query, search Pinecone, return ranked Documents (sync)."""
    top_k = top_k or settings.RETRIEVER_TOP_K
    index = _get_pinecone_index()
    if index is None:
        return []
    try:
        vec = _get_embeddings().embed_query(query)
        results = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            namespace=settings.PINECONE_NAMESPACE,
        )
        matches = (
            results.get("matches", []) if isinstance(results, dict)
            else getattr(results, "matches", []) or []
        )
        docs: List[Document] = []
        for match in matches:
            score = (
                match.get("score", 0.0) if isinstance(match, dict)
                else getattr(match, "score", 0.0)
            )
            meta = (
                match.get("metadata", {}) if isinstance(match, dict)
                else getattr(match, "metadata", {})
            ) or {}
            meta = dict(meta)
            text = meta.pop("text", "") or meta.pop("content", "")
            # Keep chunk_index in metadata for traceability; just extract text
            meta["_score"] = round(float(score), 4)
            if text.strip():
                docs.append(Document(page_content=text.strip(), metadata=meta))

        docs.sort(key=lambda d: d.metadata.get("_score", 0.0), reverse=True)
        logger.info("Retrieved %d documents for query", len(docs))
        return docs
    except Exception as exc:
        logger.error("Pinecone query error: %s", exc)
        return []


def extract_citations(docs: List[Document]) -> List[Citation]:
    """Deduplicate and return Citation objects from document metadata."""
    seen: set[str] = set()
    citations: List[Citation] = []
    for doc in docs:
        meta = doc.metadata or {}
        url = meta.get("akn_url") or meta.get("url") or ""
        title = meta.get("document_title") or meta.get("title") or "Untitled Document"
        source = meta.get("source_file") or meta.get("file_name") or "Unknown Source"
        # Always include docs with a real akn_url; skip only those with no URL at all
        if not url:
            continue
        if url not in seen:
            citations.append(Citation(url=url, document_title=title, source_file=source))
            seen.add(url)
    return citations


def _build_citation_index(citations: List[Citation]) -> str:
    """
    Numbered reference list embedded in the system prompt.
    Gives the LLM exact URLs to use in inline [Title](url) markdown links.
    """
    if not citations:
        return (
            "NO_SOURCES_FOUND — No documents with a verifiable akn_url were returned "
            "for this query. Answer the question using your legal knowledge of Zimbabwean law. "
            "Do NOT mention the database, Pinecone, or any retrieval system in your response. "
            "Do NOT fabricate case URLs. You may reference well-known statutes and case names "
            "by name only (without inventing URLs). Simply answer the legal question directly."
        )
    lines = ["Available sources — use ONLY these exact URLs for inline markdown citation links:"]
    for i, c in enumerate(citations, 1):
        lines.append(
            f'[{i}] Title: "{c.document_title}"'
            f'  |  URL: {c.url}'
            f'  |  File: {c.source_file}'
        )
    return "\n".join(lines)


# ── Prompt Templates ──────────────────────────────────────────────────────────

_ROUTER_SYSTEM = (
    "You are an intent classifier for a Zimbabwean Legal AI assistant called Sankofa.\n"
    "Classify the user's latest message as exactly one of:\n"
    "  SEARCH     — a specific legal question requiring database lookup\n"
    "  GREETING   — a greeting, 'who are you', or meta-question about the system\n"
    "  NO_SEARCH  — a non-legal general question\n"
    "Return ONLY the label word, nothing else."
)

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_ROUTER_SYSTEM),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

_LEGAL_RAG_SYSTEM_TEMPLATE = """\
You are Sankofa, a specialist legal research assistant with deep expertise in Zimbabwean law —
constitutional law, criminal law, civil law, company law, labour law, family law, and all major
statutes and case law of Zimbabwe.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT — follow these rules exactly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Write in well-structured Markdown with clear paragraphs separated by blank lines.
2. Use **bold** for every case name, statute name, and section reference on first mention.
   Example: **Minister of Home Affairs v Bickle 1983 (1) ZLR 25**
3. INLINE CITATION LINKS — every time you mention a case or statute that appears in the
   SOURCES list below, embed a clickable Markdown link using the exact URL provided.
   Format:  [Document Title](exact_url_from_sources)
   Example: [Labour Act Chapter 28:01](https://akn.africa/zw/act/2001/...)
   Do this for EVERY mention of each source, not just the first.
4. For longer answers use a bold heading to introduce each section:
   **Background:** then the paragraph text.
   **Relevant Statutory Provisions:** then the paragraph.
   **Case Law:** then the paragraph.
   **Conclusion:** then the paragraph.
5. Close EVERY legal response — regardless of length — with this exact disclaimer block:

---
*Disclaimer: This response is for legal research and educational purposes only. It does not constitute legal advice. You should consult a qualified legal practitioner admitted to practice in Zimbabwe before acting on any of the information provided.*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWER QUALITY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Answer ONLY from the PROVIDED LEGAL CONTEXT below. Do NOT fabricate cases, facts, URLs, or statutes.
- CRITICAL: Only cite cases/documents that appear in the SOURCES list above with a real akn_url.
  Never invent a URL or a case name. If a case is not in SOURCES, do not cite it.
- If the SOURCES list begins with "NO_SOURCES_FOUND", answer the question directly using your
  legal knowledge of Zimbabwean law. Never mention the database, retrieval system, or Pinecone.
  Do not say "no sources were found". Just give a helpful, accurate legal answer.
- More recent case law takes precedence — note this where relevant.
- Cite the specific section number, chapter number, or full case citation for every legal point.
- Do NOT predict court outcomes or express personal opinions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{citation_index}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEGAL CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
"""

_GENERIC_SYSTEM = """\
You are Sankofa, a specialist Zimbabwean legal research AI assistant.
The user has sent a message that is not a specific legal query.

RULES:
- For greetings or 'who are you': respond warmly and professionally. Explain that you specialise
  in Zimbabwean law — statutes, case law, constitutional law, rights, and legal procedures.
  Invite the user to ask a legal question.
- For clearly off-topic questions: politely decline and redirect the user to ask a legal question.
- Keep responses short and professional.
- Do NOT cite legal sources.
- You may use **bold** for emphasis.
"""

OFF_LIMITS_RESPONSE = (
    "I am Sankofa, a legal research assistant. I am unable to assist with requests that appear "
    "to involve facilitation of unlawful activity. If you have a legitimate legal question about "
    "this topic under Zimbabwean law, please rephrase your query.\n\n"
    "*Disclaimer: This system is for legal research purposes only and does not constitute legal advice.*"
)


# ── History helpers ───────────────────────────────────────────────────────────

def _history_to_lc(history: List[dict]) -> List[Any]:
    msgs = []
    for msg in history[-16:]:  # 8 user + 8 assistant turns
        if msg["role"] == "user":
            msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            msgs.append(AIMessage(content=msg["content"]))
    return msgs


# ── Intent classification ─────────────────────────────────────────────────────

async def classify_intent(question: str, history: List[dict]) -> str:
    llm = _build_llm(streaming=False)
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    try:
        result = await chain.ainvoke({
            "question": question,
            "history": _history_to_lc(history),
        })
        intent = result.strip().upper()
        if "GREETING" in intent:
            return "GREETING"
        if "NO_SEARCH" in intent or "NO" in intent.split():
            return "NO_SEARCH"
        return "SEARCH"
    except Exception as exc:
        logger.warning("Router failed (%s); defaulting to SEARCH", exc)
        return "SEARCH"


# ── Parallel classify + retrieve ──────────────────────────────────────────────

async def classify_and_retrieve(
    question: str,
    history: List[dict],
) -> tuple[str, List[Document], str]:
    """
    Run intent classification and Pinecone retrieval concurrently.
    Combined latency = max(classify_time, retrieval_time) instead of sum.
    Retrieval is sync so it runs in a thread-pool executor.
    Returns (intent, docs, veritas_note).
    """
    loop = asyncio.get_event_loop()
    intent_task = asyncio.create_task(classify_intent(question, history))
    retrieval_task = loop.run_in_executor(None, retrieve_documents, question)

    tasks: list = [intent_task, retrieval_task]
    is_const = _is_constitution_query(question)
    if is_const:
        veritas_task = asyncio.create_task(_fetch_veritas_note())
        tasks.append(veritas_task)

    results = await asyncio.gather(*tasks)
    intent, docs = results[0], results[1]
    veritas_note = results[2] if is_const else ""
    return intent, docs, veritas_note


# ── Streaming generators ──────────────────────────────────────────────────────

async def stream_legal_answer(
    question: str,
    history: List[dict],
    docs: List[Document],
    citations: List[Citation],
    veritas_note: str = "",
) -> AsyncGenerator[str, None]:
    """Stream a markdown-formatted, fully-cited legal answer."""
    if docs:
        context_parts = []
        for d in docs:
            meta = d.metadata
            src_url = meta.get("akn_url") or meta.get("url", "N/A")
            src_title = meta.get("document_title") or meta.get("title", "Unknown")
            src_file = meta.get("source_file") or meta.get("file_name", "")
            chunk_idx = meta.get("chunk_index", "")
            score = meta.get("_score", 0)
            context_parts.append(
                f"[Source: {src_title} | URL: {src_url} | File: {src_file}"
                f"{f' | Chunk: {chunk_idx}' if chunk_idx != '' else ''}"
                f" | Relevance: {score:.3f}]\n\n"
                f"{d.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)
        if veritas_note:
            context = veritas_note + "\n\n---\n\n" + context
    else:
        context = (
            "No retrieved documents for this query. Answer using your knowledge of "
            "Zimbabwean law. Do not mention any retrieval system or database. "
            "Do not fabricate case URLs."
        )
        if veritas_note:
            context = veritas_note + "\n\n" + context

    citation_index = _build_citation_index(citations)
    lc_history = _history_to_lc(history[:-1])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_LEGAL_RAG_SYSTEM_TEMPLATE.format(
            context=context,
            citation_index=citation_index,
        )),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    llm = _build_llm(streaming=True)
    chain = prompt | llm | StrOutputParser()
    async for chunk in chain.astream({"question": question, "history": lc_history}):
        yield chunk


async def stream_generic_answer(
    question: str,
    history: List[dict],
) -> AsyncGenerator[str, None]:
    """Stream a short greeting or non-legal response."""
    lc_history = _history_to_lc(history[:-1])
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_GENERIC_SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    llm = _build_llm(streaming=True)
    chain = prompt | llm | StrOutputParser()
    async for chunk in chain.astream({"question": question, "history": lc_history}):
        yield chunk
