import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys

# **Environment Variables**
DATA_BUCKET_NAME = os.environ.get("DATA_BUCKET_NAME")  # S3 bucket name for CIFAR-10 data
MODEL_BUCKET_NAME = os.environ.get("MODEL_BUCKET_NAME")  # S3 bucket for model output
MODEL_NAME = os.environ.get("MODEL_NAME", "default_model.pth")  # Model name
MODEL_SAVE_PATH = f"/tmp/{MODEL_NAME}"  # Save path for the trained model

# **S3 Client**
s3_client = boto3.client('s3')

# **Helper Function for Logging**
def log_message(message):
    print(f"[INFO] {message}")
    sys.stdout.flush()

# **Download CIFAR-10 Files from S3**
def download_cifar10_from_s3():
    log_message("Starting to download CIFAR-10 files from S3...")
    local_data_path = "/tmp/data/cifar-10-batches-py"
    os.makedirs(local_data_path, exist_ok=True)

    batch_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch", "batches.meta"]
    for file_name in batch_files:
        s3_key = f"cifar-10-batches-py/{file_name}"
        local_file_path = f"{local_data_path}/{file_name}"
        try:
            log_message(f"Downloading {file_name} from S3: {DATA_BUCKET_NAME}/{s3_key} to {local_file_path}...")
            s3_client.download_file(DATA_BUCKET_NAME, s3_key, local_file_path)
            log_message(f"Successfully downloaded {file_name}")
        except Exception as e:
            log_message(f"Error downloading {file_name}: {e}")

    log_message("All CIFAR-10 files downloaded successfully.")

# **Define CNN Model**
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

# **Device Configuration**
device = torch.device("cpu") 
log_message(f"Using device: {device}")

# **Testing Function to Evaluate Accuracy**
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# **Training Loop**
def train_model(epochs=10):
    log_message("Starting training process...")
    # Download dataset from S3
    download_cifar10_from_s3()

    log_message("Loading CIFAR-10 dataset...")
    # Load CIFAR-10 Dataset from Local Path
    try:
        train_dataset = datasets.CIFAR10(root='/tmp/data', train=True, download=False, transform=transforms.Compose([ 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        test_dataset = datasets.CIFAR10(root='/tmp/data', train=False, download=False, transform=transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        log_message("CIFAR-10 dataset loaded successfully.")
    except Exception as e:
        log_message(f"Error loading dataset: {e}")
        return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    log_message(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            log_message(f"Processing batch {batch_idx + 1}/{len(train_loader)} for epoch {epoch + 1}")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        log_message(f"Epoch [{epoch + 1}/{epochs}] completed, Loss: {running_loss / len(train_loader):.4f}")

        # Evaluate the model on test data after each epoch
        accuracy = test_model(model, test_loader)
        log_message(f"Epoch [{epoch + 1}/{epochs}] Test Accuracy: {accuracy:.2f}%")

    # Save and upload the trained model
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        log_message(f"Model saved locally at: {MODEL_SAVE_PATH}")
        log_message(f"Uploading model to S3: {MODEL_BUCKET_NAME}/trained_models/{MODEL_NAME}")
        s3_client.upload_file(MODEL_SAVE_PATH, MODEL_BUCKET_NAME, f"trained_models/{MODEL_NAME}")
        log_message("Model upload completed successfully.")
    except Exception as e:
        log_message(f"Error saving/uploading model: {e}")

# **Main Function**
if __name__ == "__main__":
    log_message("Starting training in cloud environment...")
    train_model(epochs=50)
