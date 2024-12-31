import streamlit as st
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client.models import VectorParams
import numpy as np

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Load Hugging Face model and tokenizer
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Open-source model for embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def compute_embeddings(texts):
    """Compute embeddings for a list of texts using Hugging Face model."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    # Average the token embeddings to get sentence embeddings
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

st.title("Qdrant Local Dashboard with Hugging Face Embeddings")

# Sidebar: File Upload
st.sidebar.title("Upload Text Files")
uploaded_files = st.sidebar.file_uploader("Drag and drop text files to upload", accept_multiple_files=True, type=["txt"])

if uploaded_files:
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s).")

# Step 1: List collections
collections = client.get_collections().collections
collection_names = [col.name for col in collections]
st.sidebar.title("Collections")
selected_collection = st.sidebar.selectbox("Select a collection to add vectors", collection_names + ["Create New Collection"])

if selected_collection:
    # Step 2: If creating a new collection
    if selected_collection == "Create New Collection":
        new_collection_name = st.text_input("Enter new collection name")
        if st.button("Create Collection") and new_collection_name:
            vectors_config = VectorParams(size=384, distance="Cosine")
            client.create_collection(new_collection_name, vectors_config=vectors_config)
            st.success(f"Collection '{new_collection_name}' created!")
            selected_collection = new_collection_name  # Switch to the new collection

    # Step 3: Process uploaded files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            text = uploaded_file.read().decode("iso-8859-1")
            documents.extend(text.split("\n"))  # Split the file into lines or segments

        # Step 4: Compute embeddings using Hugging Face
        st.write(f"Processing {len(documents)} segments...")
        embeddings = compute_embeddings(documents)

        # Step 5: Insert into Qdrant
        points = [
            {"id": i, "vector": embedding.tolist(), "payload": {"text": documents[i]}}
            for i, embedding in enumerate(embeddings)
        ]
        client.upsert(collection_name=selected_collection, points=points)
        st.success(f"Inserted {len(points)} embeddings into '{selected_collection}'!")

# Step 6: Search in selected collection
if selected_collection and selected_collection != "Create New Collection":
    st.subheader(f"Search in Collection: {selected_collection}")
    query_text = st.text_area("Query Text")
    if st.button("Search"):
        # Generate embedding for the query
        query_embedding = compute_embeddings([query_text])[0]

        # Search in Qdrant
        results = client.search(collection_name=selected_collection, query_vector=query_embedding, limit=5)
        st.write("Search Results:")
        for result in results:
            st.json(result.dict())
