import os
import warnings
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging

# Suppress TensorFlow and other library warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Load the GPT-2 model and tokenizer
model_name = "gpt2-large"  # Options: 'gpt2', 'gpt2-medium', 'gpt2-large'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prompt for the model
prompt = "What is AI?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7,  # Reduces randomness
    top_k=50,         # Limits to top-50 tokens
    top_p=0.9         # Nucleus sampling
)

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
