# Use an official Python runtime as a parent image
FROM python:3.10.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install TensorFlow and other needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV TFHUB_CACHE_DIR=/app/tfhub_cache

# Run the application
CMD ["python", "run.py"]
