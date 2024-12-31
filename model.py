# import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_name, device):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    return tokenizer, model

def generate_response(model, tokenizer, user_input, context, max_length):
    full_context = f"{context} {user_input}"
    inputs = tokenizer(full_context, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split(user_input)[-1].strip()
