Zimbabwe Legal RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot designed to answer Zimbabwean law-related questions with accurate citations from legal documents. This system combines FastAPI, LangChain, Mistral AI, and Pinecone to deliver grounded, context-aware responses.

Features

Accepts user legal questions via a REST API.
Retrieves relevant legal documents from Pinecone vector database.
Generates concise answers using Mistral LLM.
Provides citations (URLs, document titles, source files) for transparency.
Built for Zimbabwean law context.


Tech Stack

FastAPI – API framework for backend.
LangChain – Orchestration of RAG pipeline.
langchain-mistralai – Integration with Mistral AI models.
Pydantic – Data validation for request/response models.
Pinecone – Vector database for document retrieval.
uvicorn – ASGI server for FastAPI.
python-dotenv – Environment variable management.
Flask – (Optional) for auxiliary endpoints or admin tools.


Architecture Overview

User Query → /ask endpoint.
Embeddings generated using MistralAIEmbeddings.
Retriever queries Pinecone index for top-k relevant documents.
LangChain RAG Pipeline:

Formats retrieved docs into context.
Passes context + question to Mistral LLM.
Extracts citations from metadata.


Response → JSON with answer + citations.