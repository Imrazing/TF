# Use an official Python runtime as a parent image
FROM python:3.10.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
RUN mkdir -p /tmp/fastembed_cache/ && chmod 777 /tmp/fastembed_cache/

# Install TensorFlow and other needed packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV TFHUB_CACHE_DIR=/app/tfhub_cache

# Run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "run:app"]

