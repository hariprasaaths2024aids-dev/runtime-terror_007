from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.loader import load_file
import os

# Use a public model compatible with Groq or basic transformer embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Load and chunk any document
def load_document(file_path):
    raw_chunks = load_file(file_path)

    if isinstance(raw_chunks, list):
        full_text = "\n".join(raw_chunks)
    else:
        full_text = raw_chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.create_documents(
        texts=[full_text],
        metadatas=[{"source": os.path.basename(file_path)}]
    )

    return docs

# Create a FAISS vectorstore from the document chunks
def create_vectorstore(docs):
    embeddings = get_embedding_model()
    return FAISS.from_documents(docs, embeddings)

# Search similar chunks using FAISS
def search_vectorstore(vectorstore, query: str, k=5):
    return vectorstore.similarity_search(query, k=k)
