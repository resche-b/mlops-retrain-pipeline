# Use AWS Lambda base image for Python 3.9
FROM public.ecr.aws/lambda/python:3.9

# Set working directory
WORKDIR /app

# Copy your requirements.txt into the container
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Lambda handler script into the container
COPY predict.py ./predict.py

# Set the Lambda handler function
CMD ["predict.lambda_handler"]
