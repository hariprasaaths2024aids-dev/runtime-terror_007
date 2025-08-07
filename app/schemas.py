from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class Clause(BaseModel):
    text: str
    page: Optional[int]

class QueryResponse(BaseModel):
    decision: str
    amount: Optional[str]
    justification: str
    clauses_used: List[Clause]