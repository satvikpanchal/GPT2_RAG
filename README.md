# Satvik-GPT

Satvik-GPT is a Streamlit application that uses GPT-2 for generating text. It integrates with Qdrant for vector search and retrieval.

## Project Structure


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/satvikpanchal/GPT2_RAG
    cd GPT2_RAG
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Start the Qdrant server using Docker:
    ```sh
    docker-compose up
    ```

2. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

## Files

- [app.py]: Main Streamlit application file.
- [model.py]: Contains functions to load and use the GPT-2 model.
- [qdrant_dashboard.py]: Dashboard for managing Qdrant collections.
- [qdrant_db.py]: Functions for interacting with Qdrant.
- [rag_utils.py]: Utility functions for retrieval-augmented generation.
- [settings.py]: Configuration settings for the application.
- [requirements.txt]: List of required Python packages.

## License

This project is licensed under the MIT License.