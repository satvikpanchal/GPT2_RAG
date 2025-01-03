from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from transformers import AutoTokenizer, AutoModel
import torch
import uuid

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)  # Replace with your Qdrant host and port
collection_name = "real_time_rag"

# Create a collection with vectors_config
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance="Cosine")
)

# Load tokenizer and model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings
def embed_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Insert embeddings into Qdrant
def insert_documents_to_qdrant(documents):
    embeddings = embed_texts(documents)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": documents[i]}
        )
        for i in range(len(documents))
    ]
    client.upsert(collection_name=collection_name, points=points)

# Query the Qdrant collection
def query_qdrant(query_text, top_k=3):
    query_embedding = embed_texts([query_text])[0]
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    return [result.payload["text"] for result in results] if results else ["No results found."]
