FROM python:3.10

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set the default environment variable for Hugging Face cache
ENV TRANSFORMERS_CACHE=/app/gpt2

# Expose the Streamlit app port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
