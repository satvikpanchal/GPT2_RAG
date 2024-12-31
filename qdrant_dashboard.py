import streamlit as st
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
from qdrant_client.models import VectorParams
import numpy as np
import plotly.express as px
import os

# Fix protobuf issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Initialize Qdrant client
def initialize_qdrant_client():
    try:
        client = QdrantClient(host="localhost", port=6333)
        return client
    except Exception as e:
        st.error(f"Unable to connect to Qdrant: {e}")
        st.stop()

client = initialize_qdrant_client()

# Load Hugging Face model and tokenizer
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Open-source model for embeddings
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

tokenizer, model = load_model()

def compute_embeddings(texts):
    """Compute embeddings for a list of texts using Hugging Face model."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)  # Average the token embeddings
    return embeddings.numpy()

st.title("Qdrant Dashboard with Enhanced Visualizations")

# Sidebar: File Upload
st.sidebar.title("Upload Text Files")
uploaded_files = st.sidebar.file_uploader("Drag and drop text files to upload", accept_multiple_files=True, type=["txt"])

if uploaded_files:
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s).")

# Step 1: List collections
try:
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
except Exception as e:
    st.error(f"Unable to fetch collections: {e}")
    st.stop()

st.sidebar.title("Collections")
selected_collection = st.sidebar.selectbox("Select a collection or create new", collection_names + ["Create New Collection"])

if selected_collection:
    # Step 2: Create a new collection
    if selected_collection == "Create New Collection":
        new_collection_name = st.text_input("Enter new collection name", placeholder="e.g., my_collection")
        if st.button("Create Collection") and new_collection_name:
            try:
                vectors_config = VectorParams(size=384, distance="Cosine")
                client.create_collection(new_collection_name, vectors_config=vectors_config)
                st.success(f"Collection '{new_collection_name}' created!")
                st.experimental_rerun()  # Refresh the app to include the new collection
            except Exception as e:
                st.error(f"Error creating collection: {e}")
    else:
        # Display details for the selected collection
        st.subheader(f"Selected Collection: {selected_collection}")
        try:
            collection_info = client.get_collection(selected_collection)
            st.write(f"Collection Status: **{collection_info.status}**")
            st.write(f"Total Points: **{collection_info.points_count}**")
        except Exception as e:
            st.error(f"Error fetching collection info: {e}")

    # Step 3: Process uploaded files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            try:
                text = uploaded_file.read().decode("iso-8859-1")
                documents.extend(text.strip().split("\n"))  # Split the file into lines or segments
            except Exception as e:
                st.error(f"Error reading file: {e}")

        if documents:
            # Step 4: Compute embeddings using Hugging Face
            st.write(f"Processing {len(documents)} text segments...")
            embeddings = compute_embeddings(documents)

            # Step 5: Insert data into Qdrant
            try:
                points = [
                    {"id": i, "vector": embedding.tolist(), "payload": {"text": documents[i]}}
                    for i, embedding in enumerate(embeddings)
                ]
                client.upsert(collection_name=selected_collection, points=points)
                st.success(f"Inserted {len(points)} embeddings into '{selected_collection}'!")

                # Plot embeddings
                st.subheader("Embedding Visualization")
                if embeddings.shape[1] >= 3:
                    fig = px.scatter_3d(
                        x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
                        color=np.arange(len(embeddings)),
                        title="Embeddings in 3D Space",
                        labels={"x": "Dimension 1", "y": "Dimension 2", "z": "Dimension 3"}
                    )
                else:
                    fig = px.scatter(
                        x=embeddings[:, 0], y=embeddings[:, 1],
                        color=np.arange(len(embeddings)),
                        title="Embeddings in 2D Space",
                        labels={"x": "Dimension 1", "y": "Dimension 2"}
                    )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error inserting data into Qdrant: {e}")

    # Step 6: Search in the selected collection
    st.subheader("Search in Collection")
    query_text = st.text_area("Enter query text", placeholder="Type your search query here")
    if st.button("Search"):
        try:
            # Generate embedding for the query
            query_embedding = compute_embeddings([query_text])[0]

            # Search in Qdrant
            results = client.search(collection_name=selected_collection, query_vector=query_embedding, limit=5)
            st.write("Search Results:")
            for result in results:
                st.markdown(f"""
                **ID:** {result.id}  
                **Text:** {result.payload['text']}  
                **Score:** {result.score:.2f}
                """)
        except Exception as e:
            st.error(f"Error during search: {e}")
