# app/decision.py
from app.embedding import search_vectorstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from app.config import settings

# 🔄 Generalized system prompt
system_prompt = """
You are an AI assistant that helps users answer complex questions based on provided documents like policies, contracts, HR guidelines, travel rules, or emails.

Your job is to:
- Determine if the user's query can be approved, rejected, or has insufficient information.
- Justify your answer using quotes from the document.
- Only use what's present in the document – do not guess or assume anything.
- Extract the exact sentences, paragraphs, or clauses used in your reasoning.

Respond in this strict JSON format:
{{
  "decision": "approved" or "rejected" or "insufficient information",
  "amount": number or null,
  "justification": "brief explanation for your decision",
  "clauses_used": [
    {{
      "text": "exact sentence or clause from the document",
      "page": "page number or section (if available)"
    }}
  ]
}}
"""

def evaluate_with_llm(query: str, vectorstore):
    relevant_docs = search_vectorstore(vectorstore, query, k=8)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Documents:\n\n{context}\n\nQuery:\n{query}")
    ])

    model = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key
    )

    chain = prompt | model | JsonOutputParser()

    try:
        result = chain.invoke({"context": context, "query": query})
        return result
    except Exception as e:
        return {
            "decision": "error",
            "amount": None,
            "justification": str(e),
            "clauses_used": []
        }
