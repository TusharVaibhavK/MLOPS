import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define paths
data_dir = "flowers"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # Normalize
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split dataset
train_size = int(0.7 * len(dataset))  # 70% train
val_size = int(0.15 * len(dataset))   # 15% validation
test_size = len(dataset) - train_size - val_size  # 15% test

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset info
print(f"Classes: {dataset.classes}")
print(
    f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

# Save class names for later use
with open("class_names.txt", "w") as f:
    f.write("\n".join(dataset.classes))
