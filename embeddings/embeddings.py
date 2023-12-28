# embeddings.py
from decimal import Decimal
from flask import Flask,request
import numpy as np
import tensorflow_hub as hub
import os
import tempfile

def create_app():
    app = Flask(__name__)
    # Create a temporary directory for the TensorFlow Hub cache
    hub_cache_path = os.path.join(tempfile.gettempdir(), "tfhub_cache")
    os.makedirs(hub_cache_path, exist_ok=True)

    # Set the TFHUB_CACHE_DIR environment variable
    os.environ["TFHUB_CACHE_DIR"] = hub_cache_path

    # Load the model
    embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


    @app.route('/embedding', methods=['POST'])
    def hello_world():
        data = request.json  # Assuming the data is in JSON format
        text = data.get('text', 'Guest')  # Retrieve 'name' from the JSON data, default to 'Guest' if not present
        return get_embeddingTensorF(text,embed_model)

    return app


def get_embeddingTensorF(text,embed_model):
    """
    Returns the embedding for a given topic using the Universal Sentence Encoder.
    """
    embedding_list = float_to_decimal(embed_model([text])[0].numpy().tolist())       
    return embedding_list


def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(v) for v in obj]
    return obj