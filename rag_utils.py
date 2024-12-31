# rag_utils.py
from qdrant_db import insert_documents_to_qdrant as insert_docs, query_qdrant as query_docs

def insert_documents_to_qdrant(documents):
    """
    Wrapper for inserting documents into Qdrant.
    """
    insert_docs(documents)

def query_qdrant(query, top_k=3):
    """
    Wrapper for querying Qdrant.
    """
    return query_docs(query, top_k)
