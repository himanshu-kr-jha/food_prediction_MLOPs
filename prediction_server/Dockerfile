# Start with a basic Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy and install the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our app's code
COPY main.py .

# This command runs our web server when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]