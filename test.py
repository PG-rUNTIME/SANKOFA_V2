import os
from typing import List, Optional, Any, Union

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Pinecone native client
from pinecone import Pinecone  # native client

# LangChain / Mistral
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Removed the optional markdown -> HTML conversion block

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([MISTRAL_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise EnvironmentError("One or more required environment variables are missing. Please check the .env file.")

# --- Pydantic Models ---

class Citation(BaseModel):
    url: str = Field(description="The AKN URL link to the full document.")
    document_title: str = Field(description="The title of the legal document or case.")
    source_file: str = Field(description="The file name of the source PDF or document.")

class RagResponse(BaseModel):
    answer: str = Field(description="The LLM-generated answer (Plain Text).")
    citations: List[Citation] = Field(description="List of cited documents used for grounding the answer.")

class RagQuery(BaseModel):
    question: str = Field(description="The user's legal question.")

# --- Pinecone and Embeddings Initialization ---

index: Optional[Any] = None
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Native Pinecone client connected to index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"🛑 CRITICAL: Pinecone init/connect failed. RAG off. Error: {e}")

# Embeddings
try:
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )
except Exception as e:
    print(f"🛑 CRITICAL: MistralAIEmbeddings init failed. Error: {e}")
    class MockEmbeddings:
        def embed_query(self, text: str) -> List[float]:
            return [0.0] * 768
    embeddings = MockEmbeddings()

# --- Helpers to tolerate dict/object Pinecone responses ---

def _get_matches(results: Any) -> List[Any]:
    """Return matches from Pinecone response whether it's dict-like or object-like."""
    if results is None:
        return []
    if isinstance(results, dict):
        return results.get("matches", [])
    return getattr(results, "matches", []) or []

def _get_attr(item: Any, key: str, default: Any = None) -> Any:
    """Fetch attribute or dict key seamlessly."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

# --- Custom Retriever (Pinecone native) ---

def custom_pinecone_retriever(query: str, top_k: int = 3) -> List[Document]:
    """
    Uses native pinecone-client + Mistral embeddings.
    Returns LangChain Document objects.
    """
    if not index or not hasattr(embeddings, 'embed_query'):
        print("⚠️ Retriever: Pinecone index or embeddings unavailable.")
        return []

    try:
        query_vector = embeddings.embed_query(query)

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        docs: List[Document] = []
        for match in _get_matches(results):
            score = _get_attr(match, "score", 0.0)
            metadata = _get_attr(match, "metadata", {}) or {}

            # 🔎 Debug log
            print(f"🔎 Match score={score} meta-keys={list(metadata.keys())}")

            page_content = metadata.pop("text", "")

            # If your ingestion used another key (e.g., "content" or "chunk"),
            # fallback here:
            if not page_content:
                page_content = metadata.pop("content", "")

            # Clean metadata noise
            metadata.pop("chunk_index", None)
            metadata.pop("score", None)

            # Skip empty chunks to avoid empty context
            if page_content.strip():
                docs.append(Document(page_content=page_content, metadata=metadata))

        print(f"✅ Retrieved {len(docs)} docs")
        return docs

    except Exception as e:
        print(f"❌ Pinecone query error: {e}")
        return []

# --- LangChain Components ---

def get_retriever_function_input(input_dict: dict) -> List[Document]:
    return custom_pinecone_retriever(input_dict["question"], top_k=5)

retriever = RunnableLambda(get_retriever_function_input)

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.1,
    mistral_api_key=MISTRAL_API_KEY
)

# --- MODIFIED PROMPT FOR PLAIN TEXT ---
template = """
You are a specialized legal research assistant focused exclusively on Zimbabwean law.
Answer only using the provided legal context. If the context does not contain the answer,
state clearly that you cannot answer based on the provided documents.

Format your response in **Plain Text**. Do NOT use Markdown, headings, bolding, or bullet points.
Use simple paragraphs. Do NOT include inline citations in the answer body.



CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs_for_prompt(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def extract_citations(docs: List[Document]) -> List[Citation]:
    citations_data: List[Citation] = []
    seen_urls = set()
    for doc in docs:
        meta = doc.metadata or {}
        url = meta.get("akn_url") or meta.get("url") or ""
        title = meta.get("document_title") or meta.get("title") or "Untitled Document"
        src = meta.get("source_file") or meta.get("file_name") or "Unknown Source"
        if url and url not in seen_urls:
            citations_data.append(Citation(url=url, document_title=title, source_file=src))
            seen_urls.add(url)
    return citations_data

# Chain construction
retrieval_chain = RunnablePassthrough.assign(docs=retriever)

rag_chain = retrieval_chain.assign(
    context=lambda x: format_docs_for_prompt(x["docs"]),
    citations_data=lambda x: extract_citations(x["docs"])
)

final_rag_chain = rag_chain | {
    "answer": prompt | llm | StrOutputParser(),  # Raw text output expected here
    "citations": lambda x: x["citations_data"]
}

# --- FastAPI App & CORS ---

app = FastAPI(
    title="Zimbabwe Legal RAG Chatbot",
    description="Retrieval-Augmented Generation for Zimbabwe Case Law."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Route (Simplified for Raw Text) ---

@app.post("/ask", response_model=RagResponse)
async def ask_question(
    query: RagQuery,
    # Kept the format query for backward compatibility, but it is ignored for output formatting
    format: str = Query(
        default="markdown",
        pattern="^(markdown|html)$",
        description="Return 'markdown' (default) or 'html'. Note: Output is now plain text regardless of this setting."
    )
) -> RagResponse:
    if not index:
        return RagResponse(
            answer="The RAG system is currently unavailable due to a critical error connecting to the Pinecone index.",
            citations=[]
        )

    try:
        result = final_rag_chain.invoke({"question": query.question})
        raw_answer_text: str = result.get("answer", "")
        citations: List[Citation] = result.get("citations", [])

        # If nothing was retrieved, make that explicit
        if not raw_answer_text.strip():
            return RagResponse(
                answer="No answer could be generated from the current context. Please try a different query or ensure the corpus is ingested.",
                citations=citations
            )

        # The LLM is now instructed to return plain text, so we return it directly.
        formatted_answer = raw_answer_text

        return RagResponse(answer=formatted_answer, citations=citations)

    except Exception as e:
        print(f"❌ RAG chain error: {e}")
        return RagResponse(
            answer="I am currently experiencing a technical difficulty and cannot process your legal query. Please try again later.",
            citations=[]
        )