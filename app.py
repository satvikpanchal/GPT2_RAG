import streamlit as st
from rag_utils import insert_documents_to_qdrant, query_qdrant
from qdrant_client.models import VectorParams
from model import load_model, generate_response
from settings import MODEL_NAME
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, ApiException
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set protocol buffers implementation
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.path.append("..")  # Add the parent directory to the Python path

# Initialize Qdrant Client
client = QdrantClient(host="localhost", port=6333, timeout=30.0)

# Helper function to ensure collection exists
def ensure_collection(client, collection_name, vector_size):
    """
    Ensure the specified Qdrant collection exists. If not, create it with the provided vector size.
    """
    try:
        # Check if the collection already exists
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Collection '{collection_name}' does not exist. Creating it...")
        # Create the collection with the necessary vector configuration
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )


# Initialize Qdrant with dummy data for testing
def initialize_qdrant():
    """
    Initialize Qdrant with a test collection and dummy data.
    """
    dummy_data = [
        {
            "id": str(uuid.uuid4()),  # Use a valid UUID for the point ID
            "vector": [0.1] * 384,
            "payload": {"text": "This is a test document."}
        }
    ]
    ensure_collection(client, "real_time_rag", vector_size=384)
    client.upsert(
        collection_name="real_time_rag",
        points=dummy_data,
    )
    print("Initialized Qdrant with dummy data.")


initialize_qdrant()

# Load GPT-2 model and tokenizer
st.spinner("Loading model...")
try:
    tokenizer, model = load_model(MODEL_NAME, "cpu")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit App Title
st.title("Satvik-GPT with RAG")
st.subheader("Enhanced by GPT-2 and Qdrant")

# Sidebar: File Upload
st.sidebar.title("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Drag and drop text files to upload", accept_multiple_files=True, type=["txt"]
)

# Process uploaded files
if uploaded_files:
    documents = []
    try:
        for uploaded_file in uploaded_files:
            text = uploaded_file.read().decode("iso-8859-1", errors="ignore")
            # Improved chunking strategy: Split by paragraphs or sentences
            segments = text.split("\n\n")  # Chunk by paragraphs
            documents.extend([segment.strip() for segment in segments if segment.strip()])
        
        # Ensure collection exists and insert documents
        ensure_collection(client, "collection_name", vector_size=128)
        try:
            insert_documents_to_qdrant(documents)
            st.sidebar.success(f"Uploaded and processed {len(documents)} chunks!")
        except ResponseHandlingException as e:
            st.sidebar.error(f"Timeout error while inserting documents: {e}")
        except Exception as e:
            st.sidebar.error(f"Error inserting documents to Qdrant: {e}")
    except Exception as e:
        st.sidebar.error(f"Error processing files: {e}")
else:
    st.sidebar.warning("No documents uploaded yet. Please upload files to populate the database.")

# Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Settings
st.sidebar.title("Settings")
max_length = st.sidebar.slider("Response Max Length", 50, 300, 150, 10)

# User Input Section
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:", placeholder="Type your query here...")
    submit = st.form_submit_button("Submit")

if submit and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Retrieve relevant documents from Qdrant
    try:
        with st.spinner("Retrieving context..."):
            retrieved_docs = query_qdrant(user_input, top_k=3)
            if not retrieved_docs:
                st.warning("No relevant documents found in the database.")
                context = ""
            else:
                context = " ".join(retrieved_docs)
                st.session_state.messages.append({"role": "context", "content": context})
    except ResponseHandlingException as e:
        st.error(f"Timeout error while retrieving documents: {e}")
        context = ""
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        context = ""
    
    # Generate response
    try:
        with st.spinner("Generating response..."):
            bot_response = generate_response(model, tokenizer, user_input, context, max_length)
            st.session_state.messages.append({"role": "bot", "content": bot_response})
    except Exception as e:
        bot_response = "Sorry, I encountered an error while generating a response."
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        st.error(f"Error generating response: {e}")

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "context":
        st.markdown(f"**Retrieved Context:** {msg['content']}")
    elif msg["role"] == "bot":
        st.markdown(f"**Bot:** {msg['content']}")

# Footer
st.markdown("---")
st.markdown("Powered by [GPT-2](https://huggingface.co/transformers/) and [Qdrant](https://qdrant.tech/)")
