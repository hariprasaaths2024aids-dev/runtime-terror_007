📦 requirements.md

This file documents all the key Python packages required to run the LLM Document Query System using FastAPI, LangChain, Groq GPT-4.0, FAISS, and PDF document processing.

✅ Core Dependencies

Package

Version

Purpose

fastapi

0.116.1

Web API framework

uvicorn

0.35.0

ASGI server to run FastAPI apps

python-dotenv

1.1.1

Load environment variables from .env

📄 Document Processing

Package

Version

Purpose

PyMuPDF

1.26.3

Extract text from PDFs

docx2txt

0.9

Read .docx files (if needed)

python-docx

1.2.0

Process Word .docx documents

🌐 Web Utilities

Package

Version

Purpose

requests

2.32.4

Download documents from URLs

python-multipart

0.0.20

Handle file uploads (optional)

🧠 Embedding & Retrieval

Package

Version

Purpose

faiss-cpu

1.11.0.post1

Vector store for similarity search

sentence-transformers

5.0.0

Generate embeddings for documents/questions

🧼🔗 LangChain Ecosystem

Package

Version

Purpose

langchain

0.3.27

Framework for LLM-based apps

langchain-community

0.3.27

Community integrations (like FAISS)

langchain-core

0.3.72

Core utilities used by LangChain

langchain-openai

0.3.28

OpenAI/Groq wrapper for LangChain

langchain-groq

0.3.7

Groq LLM integration for LangChain

langchain-huggingface

0.3.1

(Optional) HuggingFace model support

langchain-text-splitters

0.3.9

Text chunking for document processing

🤖 LLM APIs

Package

Version

Purpose

openai

1.98.0

Use GPT models (works with Groq endpoint)

groq

0.31.0

Direct Groq client for OpenAI-compatible API

google-generativeai

0.8.5

(Optional) Use Gemini/Google PaLM models

🛠️ Support Libraries

Package

Version

Purpose

pydantic

2.11.7

Data validation and parsing

pydantic-settings

2.10.1

Manage environment config settings

tiktoken

0.9.0

Tokenizer for OpenAI/Groq models

📌 Installation

Install all packages with: