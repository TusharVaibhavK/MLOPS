# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm  # Import tqdm for progress bar
# # Import from preprocess.py
# from preprocess import train_loader, val_loader, test_loader, dataset

# # Define Logistic Regression Model


# class LogisticRegression(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(LogisticRegression, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(input_size, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.linear(x)

# # Define SVM Model (Linear SVM with hinge loss)


# class LinearSVM(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(LinearSVM, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(input_size, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.linear(x)

# # Define Simple CNN Model


# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(32 * 56 * 56, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)

# # Training function with progress bar


# def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, model_name="model"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if not torch.cuda.is_available():
#         print(f"CUDA not available, using CPU")
#     else:
#         print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss(
#     ) if model_name != "svm" else nn.MultiMarginLoss()  # Hinge loss for SVM
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         # Add tqdm progress bar for training
#         train_bar = tqdm(
#             train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}", leave=False)
#         for images, labels in train_bar:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             train_bar.set_postfix({"Train Loss": loss.item()})

#         # Validation
#         model.eval()
#         val_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             val_bar = tqdm(
#                 val_loader, desc=f"Validating {model_name}", leave=False)
#             for images, labels in val_bar:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         print(f"{model_name} Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * correct/total:.2f}%")

#     # Save model
#     torch.save(model.state_dict(), f"{model_name}.pth")
#     return model


# # Initialize and train models
# input_size = 3 * 224 * 224  # Flattened image size
# num_classes = len(dataset.classes)  # 19 classes

# # Logistic Regression
# logreg_model = LogisticRegression(input_size, num_classes)
# train_model(logreg_model, train_loader, val_loader,
#             num_epochs=10, lr=0.001, model_name="logreg")

# # SVM
# svm_model = LinearSVM(input_size, num_classes)
# train_model(svm_model, train_loader, val_loader,
#             num_epochs=10, lr=0.001, model_name="svm")

# # CNN
# cnn_model = SimpleCNN(num_classes)
# train_model(cnn_model, train_loader, val_loader,
#             num_epochs=10, lr=0.001, model_name="cnn")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import os
import itertools

# Define paths and transformations
data_dir = "./dataset/flowers/Flower_Classification_Dataset"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
num_classes = len(dataset.classes)

# Use a 10% subset of the dataset
subset_size = int(0.1 * len(dataset))  # ~1,729 images
subset_indices = torch.randperm(len(dataset))[:subset_size]
subset_dataset = Subset(dataset, subset_indices)

# Define models


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearSVM, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Training and evaluation function with MLflow logging


def train_or_load_model(model, train_loader, val_loader, num_epochs, lr, batch_size, model_name, run_name, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(f"CUDA not available, using CPU for {model_name}")
    else:
        print(
            f"Using CUDA device: {torch.cuda.get_device_name(0)} for {model_name}")

    model = model.to(device)

    # Check if model .pth file exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        # Evaluate loaded model
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss() if model_name != "svm" else nn.MultiMarginLoss()
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("dataset_size", len(train_loader.dataset))
            with torch.no_grad():
                val_bar = tqdm(
                    val_loader, desc=f"Validating {model_name}", leave=False)
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            mlflow.log_metric("val_loss", avg_val_loss, step=0)
            mlflow.log_metric("val_accuracy", val_accuracy, step=0)
            mlflow.log_artifact(model_path)
            mlflow.pytorch.log_model(model, f"{model_name}_pytorch")
            print(
                f"Loaded {model_name}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    else:
        # Train new model
        criterion = nn.CrossEntropyLoss() if model_name != "svm" else nn.MultiMarginLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("dataset_size", len(train_loader.dataset))

            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                train_bar = tqdm(
                    train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}", leave=False)
                for images, labels in train_bar:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_bar.set_postfix({"Train Loss": loss.item()})

                # Validation
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    val_bar = tqdm(
                        val_loader, desc=f"Validating {model_name}", leave=False)
                    for images, labels in val_bar:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                # Log metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * correct / total
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

                print(
                    f"{model_name} Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Save and log model
            try:
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                print(f"Saved and logged {model_name} model to {model_path}")
            except Exception as e:
                print(f"Error saving {model_name} model: {e}")

            # Log PyTorch model
            mlflow.pytorch.log_model(model, f"{model_name}_pytorch")

    return model

# Hyperparameter tuning


def run_hyperparameter_tuning():
    # Define hyperparameters
    learning_rates = [0.001]
    batch_sizes = [16]
    num_epochs = 5
    input_size = 3 * 224 * 224

    # Split subset dataset
    train_size = int(0.7 * len(subset_dataset))  # ~1,206 images
    val_size = int(0.15 * len(subset_dataset))   # ~259 images
    test_size = len(subset_dataset) - train_size - val_size  # ~259 images
    train_dataset, val_dataset, _ = random_split(
        subset_dataset, [train_size, val_size, test_size])

    # Set MLflow experiment
    mlflow.set_experiment("Flower_Classification")

    # Iterate over models and hyperparameters
    for model_type in ["logreg", "svm", "cnn"]:
        for lr, batch_size in itertools.product(learning_rates, batch_sizes):
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            if model_type == "logreg":
                model = LogisticRegression(input_size, num_classes)
            elif model_type == "svm":
                model = LinearSVM(input_size, num_classes)
            else:
                model = SimpleCNN(num_classes)

            # Define model path
            run_name = f"{model_type}_lr_{lr}_bs_{batch_size}"
            model_path = f"./models/{run_name}.pth"
            print(f"Starting run: {run_name}")
            train_or_load_model(model, train_loader, val_loader, num_epochs,
                                lr, batch_size, model_type, run_name, model_path)


if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    run_hyperparameter_tuning()
