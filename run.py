# run.py
from embeddings.embeddings import create_app

app = create_app()

if __name__ == "__main__":
    # Use Gunicorn as the server
    from gunicorn.app.wsgiapp import WSGIApplication
    from gunicorn.workers.gthread import GThreadWorker

    options = {
        'bind': '0.0.0.0:5000',
        'workers': 4,  # Adjust the number of workers based on your needs
        'worker_class': GThreadWorker,
    }

    WSGIApplication(app, options).run()
