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
    answers: List[str]  # ✅ Flat strings only

@router.post("/api/v1/hackrx/run", response_model=QueryResponse)
def run_query(payload: QueryRequest):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            response = requests.get(payload.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        docs = load_document(tmp_path)
        vectorstore = create_vectorstore(docs)

        results = []
        for q in payload.questions:
            try:
                raw_answer = evaluate_with_llm(q, vectorstore)
                # ✅ Only return the justification as a flat answer
                flat = raw_answer.get("justification", "No justification provided.")
                results.append(flat)
            except Exception as e:
                results.append(f"Error: {str(e)}")

        return {"answers": results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
