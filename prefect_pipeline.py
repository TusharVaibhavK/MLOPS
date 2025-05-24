import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
from prefect import flow, task
import os
from preprocess import test_loader, dataset  # Import from preprocess.py

# Define models (same as in models.py)


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

# Task to evaluate a model


@task
def evaluate_model(model, test_loader, model_name, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(f"CUDA not available, using CPU for {model_name}")
    else:
        print(
            f"Using CUDA device: {torch.cuda.get_device_name(0)} for {model_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss() if model_name != "svm" else nn.MultiMarginLoss()

    # Load model weights
    if os.path.exists(model_path):
        print(f"Loading {model_name} from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")

    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"Testing {model_name}", leave=False)
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    return {"model_name": model_name, "test_loss": avg_test_loss, "test_accuracy": test_accuracy}

# Task to log results to MLflow


@task
def log_to_mlflow(result):
    with mlflow.start_run(run_name=f"eval_{result['model_name']}"):
        mlflow.log_param("model_name", result["model_name"])
        mlflow.log_metric("test_loss", result["test_loss"])
        mlflow.log_metric("test_accuracy", result["test_accuracy"])
        mlflow.log_artifact(f"./regular_models-pth/{result['model_name']}.pth")
        print(
            f"Logged {result['model_name']} to MLflow: Test Loss: {result['test_loss']:.4f}, Test Accuracy: {result['test_accuracy']:.2f}%")

# Prefect flow


@flow(name="Flower_Classification_Evaluation")
def evaluation_flow():
    mlflow.set_experiment("Flower_Classification")
    input_size = 3 * 224 * 224
    num_classes = len(dataset.classes)

    # Define models and paths
    models = [
        ("logreg", LogisticRegression(input_size, num_classes),
         "./regular_models-pth/logreg.pth"),
        ("svm", LinearSVM(input_size, num_classes), "./regular_models-pth/svm.pth"),
        ("cnn", SimpleCNN(num_classes), "./regular_models-pth/cnn.pth")
    ]

    results = []
    for model_name, model, model_path in models:
        try:
            result = evaluate_model(model, test_loader, model_name, model_path)
            log_to_mlflow(result)
            results.append(result)
        except FileNotFoundError as e:
            print(e)

    # Find best model
    if results:
        best_model = max(results, key=lambda x: x["test_accuracy"])
        print(
            f"\nBest model: {best_model['model_name']} with Test Accuracy: {best_model['test_accuracy']:.2f}%")
        return best_model
    else:
        print("No models were evaluated successfully.")
        return None


# Run the flow
if __name__ == "__main__":
    # Optional: Schedule the flow (uncomment to enable)
    # from prefect.schedules import IntervalSchedule
    # from datetime import timedelta
    # schedule = IntervalSchedule(interval=timedelta(hours=24))
    # evaluation_flow.with_options(schedule=schedule)()

    # Run manually
    evaluation_flow()
