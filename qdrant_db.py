from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from transformers import AutoTokenizer, AutoModel
import torch

# Connect to Qdrant
client = QdrantClient(host="qdrant", port=6333)
collection_name = "real_time_rag"

# Create a collection with vectors_config
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance="Cosine")  # Define embedding size and similarity metric
)

# Load tokenizer and model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example dataset
documents = [
    "Artificial intelligence is the simulation of human intelligence.",
    "Machine learning is a subset of AI focused on training models.",
    "Neural networks are a foundational technology in deep learning."
]

# Generate embeddings
def embed_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Pooling method: Take the mean of the token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Insert embeddings into Qdrant
import uuid
from qdrant_client.models import PointStruct

# Insert embeddings into Qdrant
def insert_documents_to_qdrant(documents):
    embeddings = embed_texts(documents)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),  # Generate a unique UUID for each document
            vector=embeddings[i],
            payload={"text": documents[i]}
        )
        for i in range(len(documents))
    ]
    client.upsert(collection_name=collection_name, points=points)

# Insert documents into Qdrant
insert_documents_to_qdrant(documents)

# Query the Qdrant collection
def query_qdrant(query_text, top_k=3):
    # Generate embedding for the query
    query_embedding = embed_texts([query_text])[0]
    
    # Search in Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Retrieve and return document texts
    return [result.payload["text"] for result in results]

# Example Query
query = "What are neural networks?"
retrieved_docs = query_qdrant(query, top_k=1)
print("Retrieved Documents:", retrieved_docs)
