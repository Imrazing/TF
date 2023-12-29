from gunicorn.workers.gthread import GThreadWorker
from fastembed.embedding import FlagEmbedding as Embedding
from flask import Flask, request, jsonify
import gevent

def on_worker_init(worker):
    # Load the model for each worker
    worker.embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512)

def create_app():
    app = Flask(__name__)

    @app.route('/embedding', methods=['POST'])
    def get_embedding_route():
        try:
            data = request.json
            text = data.get('text', 'Guest')

            # Get embeddings using the model from the worker
            embeddings = get_embedding([text], g.worker.embedding_model)

            # Convert Decimal embeddings to string for JSON serialization
            str_embeddings = [str(embed) for embed in embeddings]

            # Return the result as JSON
            return jsonify(result=str_embeddings)

        except Exception as e:
            return jsonify(error=str(e)), 400

    return app

def get_embedding(documents, embedding_model):
    try:
        # Simulate some processing time to show the benefits of async workers
        gevent.sleep(0.1)

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
