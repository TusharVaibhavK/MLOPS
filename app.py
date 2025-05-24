import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os

# Define CNN model (same as in models.py)


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


# Define class names (from preprocess.py output)
classes = ['astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy',
           'carnation', 'common_daisy', 'coreopsis', 'daffodil', 'dandelion', 'iris',
           'lavender', 'lotus', 'magnolia', 'orchid', 'rose', 'sunflower', 'tulip', 'water_lily']

# Load model


def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(classes))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")

# Preprocess image


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict class


def predict(model, image, device):
    image = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Streamlit app


def main():
    st.title("Flower Classification App")
    st.write("Upload a flower image to classify it using a CNN model.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    # Load model
    model_path = "./regular_models-pth/cnn.pth"
    try:
        model, device = load_model(model_path, num_classes=len(classes))
        st.success("CNN model loaded successfully!")
    except FileNotFoundError as e:
        st.error(str(e))
        return

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict and display result
        try:
            predicted_idx = predict(model, image, device)
            predicted_class = classes[predicted_idx]
            st.write(f"**Prediction**: {predicted_class}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    main()
