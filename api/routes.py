from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import tempfile
import requests
import os
import traceback

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
    results = []

    try:
        # Download document
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            response = requests.get(payload.documents, timeout=20)
            response.raise_for_status()
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load into vector store
        docs = load_document(tmp_path)
        vectorstore = create_vectorstore(docs)

        # Process each question
        for q in payload.questions:
            try:
                raw_answer = evaluate_with_llm(q, vectorstore)
                results.append(raw_answer.get("justification", "No justification provided."))
            except Exception as e:
                results.append(f"Error processing question: {str(e)}")

    except Exception as e:
        traceback.print_exc()
        results = [f"Fatal error in processing: {str(e)}"]

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"answers": results}
