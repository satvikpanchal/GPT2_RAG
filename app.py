import streamlit as st
from rag_utils import insert_documents_to_qdrant, query_qdrant
from model import load_model, generate_response
from settings import MODEL_NAME
import os
import sys

# Set protocol buffers implementation
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.path.append("..")  # Add the parent directory to the Python path

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
        
        # Insert documents into Qdrant
        insert_documents_to_qdrant(documents)
        st.sidebar.success(f"Uploaded and processed {len(documents)} chunks!")
    except Exception as e:
        st.sidebar.error(f"Error processing files: {e}")

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
            context = " ".join(retrieved_docs)
            st.session_state.messages.append({"role": "context", "content": context})
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
