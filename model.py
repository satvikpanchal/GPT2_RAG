import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)

def load_model(model_name, device):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def generate_response(model, tokenizer, user_input, context, max_length):
    try:
        # Validate inputs
        full_context = f"{context.strip()} {user_input.strip()}"
        logging.info(f"Full Context: {full_context}")

        # Tokenize and truncate
        inputs = tokenizer(full_context, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Generated Response: {response}")
        
        return response.split(user_input)[-1].strip()
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        return "Sorry, I encountered an error while generating a response."
