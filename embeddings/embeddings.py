# embeddings.py
from fastembed.embedding import FlagEmbedding as Embedding
from typing import List
from decimal import Decimal
import numpy as np
from flask import Flask, request, jsonify  # Import jsonify for converting to JSON

def create_app():
    app = Flask(__name__)
    # Create a temporary directory for the TensorFlow Hub cache

    @app.route('/embedding', methods=['POST'])
    def hello_world():
        data = request.json  # Assuming the data is in JSON format
        text = data.get('text', 'Guest')  # Retrieve 'text' from the JSON data, default to 'Guest' if not present
        embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512)

        # Get embeddings
        embeddings = get_embedding([text],embedding_model)

        # Convert Decimal embeddings to string for JSON serialization
        str_embeddings = [str(embed) for embed in embeddings]

        # Return the result as JSON
        return embeddings

    return app

def get_embedding(documents: List[str],embedding_model) -> List[Decimal]:
    """
    Get embeddings for a list of documents using the fastembed library.

    Parameters:
    - documents (List[str]): A list of documents.

    Returns:
    - List[Decimal]: A list of decimal representations of embeddings corresponding to the input documents.
    """
    # Create an instance of the Embedding model

    # Get embeddings for the documents
    embeddings = list(embedding_model.embed(documents))
    print(type(embeddings))
    # Convert embeddings to decimal representation

    result = float_to_decimal(embeddings[0].tolist())
    return result

def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(v) for v in obj]
    return obj