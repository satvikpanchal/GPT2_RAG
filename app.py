import streamlit as st
from rag_utils import insert_documents_to_qdrant, query_qdrant
from model import load_model, generate_response
from settings import MODEL_NAME

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
sys.path.append("..")  # Add the parent directory to the Python path

# Load GPT-2 model and tokenizer
tokenizer, model = load_model(MODEL_NAME, "cpu")

# Streamlit App
st.title("Satvik-GPT with RAG")
st.subheader("Powered by GPT-2 and Qdrant")

# Sidebar: File Upload
st.sidebar.title("Upload Text Files")
uploaded_files = st.sidebar.file_uploader("Drag and drop text files to upload", accept_multiple_files=True, type=["txt"])

# Process uploaded files
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        segments = text.split(",")
        documents.extend([segment.strip() for segment in segments if segment.strip()])

    # Insert documents into Qdrant
    insert_documents_to_qdrant(documents)
    st.sidebar.success(f"Uploaded and processed {len(documents)} segments!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Settings
st.sidebar.title("Settings")
max_length = st.sidebar.slider("Max Length", 50, 300, 100, 10)

# User Input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="input", placeholder="Ask me anything...")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve relevant documents
    retrieved_docs = query_qdrant(user_input, top_k=3)
    context = " ".join(retrieved_docs)
    st.session_state.messages.append({"role": "context", "content": context})

    # Generate response
    bot_response = generate_response(model, tokenizer, user_input, context, max_length)
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "context":
        st.markdown(f"**Context Retrieved:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
