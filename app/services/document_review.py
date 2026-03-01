"""
Document review service: parse uploaded PDFs/DOCX, verify citations,
generate improvement suggestions via the LLM.
"""
from __future__ import annotations

import io
import logging
import re
from typing import List, Tuple

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from app.core.config import settings
from app.services.rag_engine import retrieve_documents
from app.services.schemas import DocumentReviewResult

logger = logging.getLogger(__name__)

# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(content: bytes) -> str:
    """Extract plain text from a PDF byte stream using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception as exc:
        logger.error("PDF extraction error: %s", exc)
        raise ValueError(f"Could not extract text from PDF: {exc}") from exc


def extract_text_from_docx(content: bytes) -> str:
    """Extract plain text from a DOCX byte stream using python-docx."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(content))
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as exc:
        logger.error("DOCX extraction error: %s", exc)
        raise ValueError(f"Could not extract text from DOCX: {exc}") from exc


def extract_text(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(content)
    if ext in ("docx", "doc"):
        return extract_text_from_docx(content)
    raise ValueError(f"Unsupported file type: .{ext}")


# ── Citation Extraction ───────────────────────────────────────────────────────

# Common Zimbabwean citation patterns
_CITATION_PATTERNS = [
    r"\b\d{4}\s+\(\d+\)\s+ZLR\s+\d+\b",          # 2021 (1) ZLR 123
    r"\b\d{4}\s+ZSC\s+\d+\b",                       # 2020 ZSC 15
    r"\b\d{4}\s+ZWHHC\s+\d+\b",                     # 2019 ZWHHC 302
    r"\b\d{4}\s+ZWSC\s+\d+\b",                      # 2022 ZWSC 7
    r"\bHH\s*\d+\s*[-–]\s*\d{2,4}\b",              # HH 302-2018
    r"\bSC\s*\d+\s*[-–]\s*\d{2,4}\b",              # SC 15-2020
    r"\b[A-Z][a-z]+ v [A-Z][a-z]+\b",               # Party v Party
    r"\bSection\s+\d+[A-Za-z]?\s+of\s+[A-Z][^,\.]+", # Section 5 of the Labour Act
    r"\bAct\s+No\.?\s*\d+\s+of\s+\d{4}\b",         # Act No. 5 of 2007
    r"\bChapter\s+\d+\s*:\s*\d+\b",                 # Chapter 29:12
]
_COMPILED_PATTERNS = [re.compile(p) for p in _CITATION_PATTERNS]


def extract_raw_citations(text: str) -> List[str]:
    """Return all unique raw citation strings found in the document text."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            citation = match.group().strip()
            if citation not in seen:
                found.append(citation)
                seen.add(citation)
    return found


def verify_citations_against_db(citations: List[str]) -> Tuple[List[str], List[str]]:
    """
    For each extracted citation, attempt a Pinecone lookup.
    Returns (verified_list, not_found_list).
    """
    verified: List[str] = []
    not_found: List[str] = []

    for citation in citations:
        docs = retrieve_documents(citation, top_k=1)
        if docs and docs[0].metadata.get("_score", 0) > 0.75:
            verified.append(citation)
        else:
            not_found.append(citation)

    return verified, not_found


# ── LLM-Powered Review ────────────────────────────────────────────────────────

_REVIEW_SYSTEM_PROMPT = """You are Sankofa, an expert Zimbabwean legal document reviewer with extensive
knowledge of Zimbabwean law, drafting conventions, and legal best practices.

You are reviewing a legal document and must provide:
1. SUMMARY: A concise professional summary of the document (3-5 sentences).
2. CITED_CASES: A comma-separated list of all cases, statutes, and legal instruments cited.
3. AREAS_FOR_IMPROVEMENT: Up to 5 specific, actionable improvements (drafting, legal accuracy, completeness).
4. OVERALL_ASSESSMENT: A brief overall assessment of the document quality and legal soundness.

Format your response EXACTLY as follows (no markdown, plain text):
SUMMARY: <text>
CITED_CASES: <case1>, <case2>, ...
AREAS_FOR_IMPROVEMENT:
1. <improvement>
2. <improvement>
...
OVERALL_ASSESSMENT: <text>

DOCUMENT TEXT (first 6000 characters):
{document_text}

KNOWN UNVERIFIED CITATIONS (not found in legal database):
{unverified_citations}
"""


def _parse_review_response(raw: str, not_found: List[str]) -> DocumentReviewResult:
    """Parse the structured LLM response into a DocumentReviewResult."""
    summary = ""
    cited_cases: List[str] = []
    improvements: List[str] = []
    assessment = ""

    lines = raw.strip().splitlines()
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line[len("SUMMARY:"):].strip()
            current_section = "summary"
        elif line.startswith("CITED_CASES:"):
            raw_cases = line[len("CITED_CASES:"):].strip()
            cited_cases = [c.strip() for c in raw_cases.split(",") if c.strip()]
            current_section = "cited"
        elif line.startswith("AREAS_FOR_IMPROVEMENT:"):
            current_section = "improvements"
        elif line.startswith("OVERALL_ASSESSMENT:"):
            assessment = line[len("OVERALL_ASSESSMENT:"):].strip()
            current_section = "assessment"
        elif current_section == "improvements" and re.match(r"^\d+\.", line):
            improvements.append(re.sub(r"^\d+\.\s*", "", line))
        elif current_section == "assessment" and line:
            assessment = (assessment + " " + line).strip()

    return DocumentReviewResult(
        summary=summary or "Summary not available.",
        cited_cases=cited_cases,
        non_existing_citations=not_found,
        areas_for_improvement=improvements,
        overall_assessment=assessment or "Assessment not available.",
    )


async def review_document(filename: str, content: bytes) -> DocumentReviewResult:
    """
    Full document review pipeline:
    1. Extract text
    2. Extract citations
    3. Verify citations against Pinecone
    4. Call LLM for structured review
    """
    # Step 1: Extract
    text = extract_text(filename, content)
    if not text:
        raise ValueError("Document appears to be empty or unreadable.")

    # Step 2: Extract citations
    raw_citations = extract_raw_citations(text)
    logger.info("Found %d raw citations in document", len(raw_citations))

    # Step 3: Verify
    _verified, not_found = verify_citations_against_db(raw_citations)

    # Step 4: LLM review (truncate document text for context window)
    truncated_text = text[:6000]
    unverified_str = ", ".join(not_found) if not_found else "None detected"

    llm = ChatMistralAI(
        model=settings.MISTRAL_CHAT_MODEL,
        temperature=0.1,
        max_tokens=1500,
        mistral_api_key=settings.MISTRAL_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_REVIEW_SYSTEM_PROMPT.format(
            document_text=truncated_text,
            unverified_citations=unverified_str,
        ))
    ])

    chain = prompt | llm | StrOutputParser()
    raw_response = await chain.ainvoke({})

    return _parse_review_response(raw_response, not_found)
