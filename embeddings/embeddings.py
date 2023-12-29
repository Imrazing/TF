from fastembed.embedding import FlagEmbedding as Embedding
from typing import List
from decimal import Decimal
from flask import Flask, request, jsonify

def create_app():
    app = Flask(__name__)

    # Move the instantiation of embedding_model outside the route handler

    @app.route('/embedding', methods=['POST'])
    def get_embedding_route():
        try:
            data = request.json
            text = data.get('text', 'Guest')
            embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512)
            # Get embeddings
            embeddings = get_embedding([text], embedding_model)

            # Convert Decimal embeddings to string for JSON serialization
            str_embeddings = [str(embed) for embed in embeddings]

            # Return the result as JSON
            return jsonify(result=str_embeddings)

        except Exception as e:
            return jsonify(error=str(e)), 400  # Return a JSON response with error message and status code 400 for bad request

    return app

def get_embedding(documents: List[str], embedding_model) -> List[Decimal]:
    try:
        # Get embeddings for the documents
        embeddings = list(embedding_model.embed(documents))

        # Convert embeddings to decimal representation
        result = float_to_decimal(embeddings[0].tolist())
        return result

    except Exception as e:
        raise RuntimeError(f"Error in embedding process: {str(e)}")

def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(v) for v in obj]
    return obj
