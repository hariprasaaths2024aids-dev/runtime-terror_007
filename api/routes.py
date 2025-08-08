import traceback
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List
import tempfile
import requests
import os

from app.embedding import load_document, create_vectorstore
from app.decision import evaluate_with_llm

router = APIRouter()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@router.post("/hackrx/run", response_model=QueryResponse)
def run_query(payload: QueryRequest):
    tmp_path = None
    try:
        print("📄 Downloading document from:", payload.documents)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            response = requests.get(payload.documents, timeout=20)
            response.raise_for_status()
            tmp.write(response.content)
            tmp_path = tmp.name

        print("📥 Document saved to:", tmp_path)
        docs = load_document(tmp_path)
        print("✅ Document loaded.")

        vectorstore = create_vectorstore(docs)
        print("✅ Vectorstore created.")

        results = []
        for q in payload.questions:
            print("🔍 Processing question:", q)
            raw_answer = evaluate_with_llm(q, vectorstore)
            print("🤖 Raw answer:", raw_answer)
            flat = raw_answer.get("justification", "No justification provided.")
            results.append(flat)

        return {"answers": results}

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("❌ ERROR in /hackrx/run:", traceback_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
