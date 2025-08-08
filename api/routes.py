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
    answers: List[str]  # Flat strings only

@router.post("/hackrx/run", response_model=QueryResponse)
def run_query(payload: QueryRequest):
    tmp_path = None
    try:
        # Download PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            response = requests.get(payload.documents)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download document")
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load + embed
        docs = load_document(tmp_path)
        vectorstore = create_vectorstore(docs)

        # Answer questions
        results = []
        for q in payload.questions:
            try:
                raw_answer = evaluate_with_llm(q, vectorstore)
                flat = raw_answer.get("justification", "No justification provided.")
                results.append(flat)
            except Exception as e:
                results.append(f"Error: {str(e)}")

        return {"answers": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
