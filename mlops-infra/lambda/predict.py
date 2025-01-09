import base64
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO
import boto3
import os

# **Model Class Definition**
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# **Initialize S3 client**
s3_client = boto3.client('s3')

# **Environment Variable for the S3 Bucket Name**
BUCKET_NAME = os.environ.get('MODEL_BUCKET_NAME', 'your-model-bucket-name')

# **Function to Get Latest Model Key from S3**
def get_latest_model_key():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="trained_models/")
    if 'Contents' not in response:
        raise Exception("No models found in the 'trained_models/' folder!")

    sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
    latest_model_key = sorted_files[0]['Key']
    print(f"[INFO] Latest model found: {latest_model_key}")
    return latest_model_key

# **Function to Download Model from S3**
def download_model():
    latest_model_key = get_latest_model_key()
    local_model_path = f"/tmp/{latest_model_key.split('/')[-1]}"

    if not os.path.exists(local_model_path):
        print(f"[INFO] Downloading model: {latest_model_key} from S3")
        s3_client.download_file(BUCKET_NAME, latest_model_key, local_model_path)
    else:
        print(f"[INFO] Model already exists locally: {local_model_path}")

    return local_model_path

# **Load the Latest Model During Cold Start**
MODEL_PATH = download_model()
print(f"[INFO] Using model: {MODEL_PATH}")

# **Load Model Weights into Model Structure**
MODEL = ImprovedCNN()  # Instantiate model structure
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))  # Load weights
MODEL.eval()  # Set the model to evaluation mode
print(f"[INFO] Model loaded successfully from {MODEL_PATH}")

# **Lambda Handler Function**
def lambda_handler(event, context):
    print("[INFO] Received event:")
    print(json.dumps(event, indent=4))
    try:
        # Check if 'body' exists in the event
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            raise ValueError("[ERROR] No 'body' key found in event!")

        # Check for 'image' key in the body
        if 'image' not in body:
            raise ValueError("[ERROR] No 'image' key found in request body!")

        base64_image = body['image']  # Base64-encoded image string
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))

        # **Preprocess the Image**
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # **Perform Inference**
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Print the prediction for logging
        print(f"[INFO] Prediction result: {int(predicted.item())}")

        # **Return the Prediction**
        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": int(predicted.item()), "model_name": MODEL_PATH}),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",  # Enable CORS for React frontend
            }
        }

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            }
        }
