


##########################################################


import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import glob
import time
import random
import threading
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# ============================
# Define CNN Models
# ============================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================
# Custom Dataset Loader
# ============================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [f for f in glob.glob(os.path.join(root_dir, '**', '*'), recursive=True) if
                            f.endswith(('jpg', 'png', 'jpeg', 'tif'))]
        if len(self.image_paths) == 0:
            raise ValueError("❌ No valid images found in the dataset folder.")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, random.randint(0, 9)


# ============================
# Training Function
# ============================
def train_model(model, train_loader, num_epochs, learning_rate, progress_bar):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.progress((epoch + 1) / num_epochs)
    torch.save(model.state_dict(), "model.pth")
    st.success("✅ Training Complete! Model Saved.")


# ============================
# Model Testing Function
# ============================
def load_model(model_type):
    model = SimpleCNN() if model_type == "SimpleCNN" else AdvancedCNN()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    class_names = [f"Class {i}" for i in range(10)]
    st.subheader("🔍 Model Predictions on Test Data")
    cols = st.columns(5)

    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            image = image.to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1).item()
            pred_label = class_names[pred]

            img_pil = transforms.ToPILImage()(image.squeeze(0).cpu())
            with cols[i % 5]:
                st.image(img_pil, caption=f"Prediction: {pred_label}", use_column_width=True)
            if i >= 19:
                break


# ============================
# Streamlit UI
# ============================
st.title("🔥 Streamlit ML Trainer - Custom Dataset")

st.sidebar.header("📂 Dataset Selection")
dataset_path = st.sidebar.text_input("Dataset Folder Path", "D:/Dataset/DIP3E_Original_Images_CH01")

if st.sidebar.button("📂 Load Dataset"):
    try:
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        train_dataset = CustomImageDataset(root_dir=dataset_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        if len(train_dataset) > 0:
            st.sidebar.success(f"✅ Loaded {len(train_dataset)} images.")
            st.session_state.train_loader = train_loader
        else:
            st.sidebar.error("❌ No images found in the dataset folder.")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading dataset: {e}")

st.sidebar.header("⚙️ Training Parameters")
num_epochs = st.sidebar.slider("Epochs", 1, 10, 2)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

st.sidebar.header("🛠 Model Selection")
model_option = st.sidebar.selectbox("Choose Model", ["SimpleCNN", "AdvancedCNN"])

if st.button("🚀 Start Training"):
    if "train_loader" in st.session_state:
        progress_bar = st.progress(0)
        threading.Thread(target=train_model, args=(
            SimpleCNN() if model_option == "SimpleCNN" else AdvancedCNN(),
            st.session_state.train_loader,
            num_epochs,
            learning_rate,
            progress_bar
        )).start()
    else:
        st.error("❌ Please load the dataset before starting training!")

if st.button("🔍 Test Model on New Data"):
    model = load_model(model_option)
    test_model(model, st.session_state.train_loader)
