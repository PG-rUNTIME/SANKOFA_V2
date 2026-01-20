import os
from typing import List, Optional, Any, Union

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Pinecone native client
from pinecone import Pinecone

# LangChain / Mistral
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

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

# --- Helpers ---

def _get_matches(results: Any) -> List[Any]:
    if results is None: return []
    if isinstance(results, dict): return results.get("matches", [])
    return getattr(results, "matches", []) or []

def _get_attr(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict): return item.get(key, default)
    return getattr(item, key, default)

# --- Custom Retriever (Pinecone native) ---

def custom_pinecone_retriever(query: str, top_k: int = 3) -> List[Document]:
    if not index or not hasattr(embeddings, 'embed_query'):
        print("⚠️ Retriever: Pinecone index or embeddings unavailable.")
        return []

    try:
        query_vector = embeddings.embed_query(query)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        docs: List[Document] = []
        for match in _get_matches(results):
            score = _get_attr(match, "score", 0.0)
            metadata = _get_attr(match, "metadata", {}) or {}
            page_content = metadata.pop("text", "")
            if not page_content:
                page_content = metadata.pop("content", "")

            # Cleanup
            metadata.pop("chunk_index", None)
            metadata.pop("score", None)

            if page_content.strip():
                docs.append(Document(page_content=page_content, metadata=metadata))

        print(f"✅ Retrieved {len(docs)} docs")
        return docs

    except Exception as e:
        print(f"❌ Pinecone query error: {e}")
        return []

# --- LangChain Components ---

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.1,
    mistral_api_key=MISTRAL_API_KEY
)

# 1. ROUTER CHAIN: Decides if we need RAG or not
router_template = """
You are an intelligent classifier for a Zimbabwean Legal AI.
Analyze the user's input and determine if it requires a search in the legal database.

Return string 'SEARCH' if:
- The user is asking a specific legal question.
- The user mentions acts, cases, rights, penalties, or legal definitions.

Return string 'NO_SEARCH' if:
- The user is sending a greeting (e.g., "Hello", "Hi").
- The user is asking a general knowledge question unrelated to law (e.g., "What is 2+2?").
- The user is asking 'who are you' or other chatbot meta-questions.

Input: {question}

Output (SEARCH or NO_SEARCH):
"""
router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | llm | StrOutputParser()

# 2. GENERIC CHAIN: For greetings/non-legal queries (No RAG)
generic_template = """
You are a specialized legal research assistant focused on Zimbabwean law.
The user has sent a message that does not appear to be a specific legal query (e.g., a greeting or general question).

User Message: {question}

Instructions:
- If it is a greeting, reply politely and professionally.
- If it is a general question, answer briefly but remind the user you specialize in Zimbabwe Law.
- Do NOT fabricate legal cases or advice.
- Keep the response in Plain Text.
"""
generic_prompt = ChatPromptTemplate.from_template(generic_template)
generic_chain = generic_prompt | llm | StrOutputParser()

# 3. RAG CHAIN: For specific legal queries (Uses Pinecone)
rag_template = """
You are a specialized legal research assistant focused exclusively on Zimbabwean law.
Answer only using the provided legal context. If the context does not contain the answer,
state clearly that you cannot answer based on the provided documents.

Format your response in **Plain Text**. Do NOT use Markdown, headings, bolding, or bullet points.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

def get_retriever_function_input(input_dict: dict) -> List[Document]:
    return custom_pinecone_retriever(input_dict["question"], top_k=5)

retriever = RunnableLambda(get_retriever_function_input)

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

retrieval_chain = RunnablePassthrough.assign(docs=retriever)

rag_chain_with_context = retrieval_chain.assign(
    context=lambda x: format_docs_for_prompt(x["docs"]),
    citations_data=lambda x: extract_citations(x["docs"])
)

final_rag_chain = rag_chain_with_context | {
    "answer": rag_prompt | llm | StrOutputParser(),
    "citations": lambda x: x["citations_data"]
}

# --- FastAPI App & CORS ---

app = FastAPI(
    title="Zimbabwe Legal RAG Chatbot",
    description="Retrieval-Augmented Generation for Zimbabwe Case Law."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Route ---

@app.post("/ask", response_model=RagResponse)
async def ask_question(
    query: RagQuery,
    format: str = Query(default="markdown") # format arg ignored per previous request
) -> RagResponse:
    
    # 1. Router Step: Decide Intent
    try:
        # We strip whitespace and normalize to ensure clean matching
        intent = router_chain.invoke({"question": query.question}).strip().upper()
        print(f"🧠 Router Decision: {intent}")
    except Exception as e:
        print(f"⚠️ Router failed, defaulting to SEARCH. Error: {e}")
        intent = "SEARCH"

    # 2. Logic Branching
    
    # BRANCH A: Generic / Greeting (No RAG, No Citations)
    if "NO_SEARCH" in intent:
        try:
            answer = generic_chain.invoke({"question": query.question})
            return RagResponse(answer=answer, citations=[])
        except Exception as e:
            print(f"❌ Generic chain error: {e}")
            return RagResponse(answer="I'm having trouble processing your request.", citations=[])

    # BRANCH B: Legal Query (RAG + Citations)
    else:
        if not index:
            return RagResponse(
                answer="The legal database is currently unavailable.",
                citations=[]
            )
        try:
            result = final_rag_chain.invoke({"question": query.question})
            raw_answer_text: str = result.get("answer", "")
            citations: List[Citation] = result.get("citations", [])

            if not raw_answer_text.strip():
                return RagResponse(
                    answer="No answer could be generated from the current context.",
                    citations=citations
                )

            return RagResponse(answer=raw_answer_text, citations=citations)

        except Exception as e:
            print(f"❌ RAG chain error: {e}")
            return RagResponse(
                answer="I am currently experiencing a technical difficulty and cannot process your legal query.",
                citations=[]
            )