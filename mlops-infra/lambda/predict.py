import base64
import json
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import boto3
import os

# Initialize S3 client
s3_client = boto3.client('s3')

# Environment variable for the S3 bucket name
BUCKET_NAME = os.environ.get('MODEL_BUCKET_NAME', 'your-model-bucket-name')

# Download the latest model during cold start (outside the handler)
def get_latest_model_key():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    if 'Contents' not in response:
        raise Exception("No models found in the bucket!")

    # Sort objects by LastModified timestamp in descending order
    sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
    latest_model_key = sorted_files[0]['Key']  # Get the latest model's S3 key
    print(f"[INFO] Latest model found: {latest_model_key}")
    return latest_model_key

def download_model():
    latest_model_key = get_latest_model_key()
    local_model_path = f"/tmp/{latest_model_key.split('/')[-1]}"
    
    if not os.path.exists(local_model_path):
        print(f"[INFO] Downloading model: {latest_model_key}")
        s3_client.download_file(BUCKET_NAME, latest_model_key, local_model_path)
    else:
        print(f"[INFO] Model already exists: {local_model_path}")
    
    return local_model_path

# Download and load the latest model during cold start
MODEL_PATH = download_model()
MODEL = torch.load(MODEL_PATH, map_location="cpu")
MODEL.eval()
print(f"[INFO] Model loaded successfully from {MODEL_PATH}")

# Lambda handler function
def lambda_handler(event, context):
    try:
        # Parse request body
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        base64_image = body['image']  # Base64-encoded image string
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))

        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Return the prediction
        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": int(predicted.item())}),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",  # Enable CORS for React frontend
            }
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            }
        }
