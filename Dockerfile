# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install Git (needed by dagshub + gitpython)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy Flask app files into the container
COPY flask_app/ /app/

# Install pip, wheel, and setuptools first
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY flask_app/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Code for local use
CMD ["python", "app.py"]

# Code for production use - Start the app using gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
