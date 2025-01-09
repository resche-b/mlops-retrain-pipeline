# Use AWS Lambda base image for Python 3.9
FROM public.ecr.aws/lambda/python:3.9

# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the Lambda function code to the root directory
COPY predict.py ./

# Set the Lambda handler to point to the function in predict.py
CMD ["predict.lambda_handler"]

