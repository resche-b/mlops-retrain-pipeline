FROM python:3.9-slim

WORKDIR /app
COPY train.py ./train_cloud.py
COPY requirements.txt ./requirements.txt

# Install Python dependencies from the requirements file
CMD ["python3", "train_cloud.py"]


# Base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy files into container
COPY train.py ./train_cloud.py
COPY requirements.txt ./requirements.txt

# Install necessary Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Entry point for your container (run the training script)
CMD ["python3", "train_cloud.py"]
