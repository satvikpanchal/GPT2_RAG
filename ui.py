import os
import warnings
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging
import streamlit as st
from qdrant_db import insert_documents_to_qdrant, query_qdrant

# Suppress TensorFlow and other library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2-large"  # Change to 'gpt2-medium' or 'gpt2-large' for larger models
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit App
st.title("Satvik-GPT with RAG")
st.subheader("Powered by GPT-2 and Qdrant")

# Drag-and-Drop for Text Files
st.sidebar.title("Upload Text Files")
uploaded_files = st.sidebar.file_uploader(
    "Drag and drop text files to upload",
    accept_multiple_files=True,
    type=["txt"]
)

# Process uploaded files
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Read the content of the text file
        text = uploaded_file.read().decode("iso-8859-1")
        # Split content by commas
        segments = text.split(",")
        documents.extend([segment.strip() for segment in segments if segment.strip()])

    # Insert documents into Qdrant
    insert_documents_to_qdrant(documents)
    st.sidebar.success(f"Uploaded and processed {len(documents)} segments!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings
st.sidebar.title("Settings")
max_length = st.sidebar.slider("Max Length", 50, 300, 100, 10)

# User Input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="input", placeholder="Ask me anything...")
    submit = st.form_submit_button("Send")

if submit and user_input:
    # Append user input to messages
    st.session_state.messages.append({"role": "user", "content": user_input})

    # RAG Step: Retrieve relevant documents
    retrieved_docs = query_qdrant(user_input, top_k=3)  # Retrieve top 3 documents
    context = " ".join(retrieved_docs)

    # Append retrieved context to chat history
    st.session_state.messages.append({"role": "context", "content": context})

    # Prepare input for GPT-2
    full_context = f"{context} {user_input}"
    inputs = tokenizer(full_context, return_tensors="pt", truncation=True).to(model.device)

    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,  # Reduces randomness
        top_k=50,         # Limits to top-50 tokens
        top_p=0.9         # Nucleus sampling
    )

    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True).split(user_input)[-1].strip()

    # Append bot response to messages
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "context":
        st.markdown(f"**Context Retrieved:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
