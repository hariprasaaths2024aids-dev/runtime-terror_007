def retrieve_clauses(vectorstore, question, k=5):
    return vectorstore.similarity_search(question, k=k)