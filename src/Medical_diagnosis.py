from PIL import ImageTk
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import easygui
import os
import requests
from io import BytesIO


class TumorSegmentationModel(nn.Module):
    def __init__(self):
        super(TumorSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.sigmoid(self.conv6(x))
        return x

    
class XRayAbnormalityModel(nn.Module):
    def __init__(self):
        super(XRayAbnormalityModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 128) 
        self.fc2 = nn.Linear(128, 2)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 64 * 64 * 64)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    
def load_mri_image(image_path):
    # Load and preprocess the MRI image (e.g., normalize, resize)
    # Return the preprocessed image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    image_np = np.array(image)
    # Convert to grayscale if necessary
    if image_np.ndim == 3:
        image_np = np.mean(image_np, axis=2)
    # Add any necessary preprocessing steps here (e.g., normalization, resizing)
    return image_np  # Return the NumPy array here


def load_xray_image(image_path):
    
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    image_np = np.array(image)

    if image_np.ndim == 3:
        image_np = np.mean(image_np, axis=2)
    
    return image_np  

def upload_image():
    global image_path
    image_path = easygui.fileopenbox(title="Select Image", default="*", filetypes=["*.jpg", "*.png", "*.jpeg"])
    if image_path:
        image = Image.open(image_path)
        # Resize the image for display
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo


def check_brain_tumor():
    if image_path:
        # Load the model (ensure the model is defined before calling this function)
        model = TumorSegmentationModel()

        # ... (Your code for loading the image and generating the tumor mask) ...
        mri_image = load_mri_image(image_path)
        input_tensor = torch.from_numpy(mri_image).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            predicted_mask = model(input_tensor)

        # ... (Your code for displaying the results) ...
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(mri_image, cmap="gray")
        plt.title("Original MRI")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask[0, 0].detach().numpy(), cmap="coolwarm", vmin=0, vmax=1)
        plt.title("Predicted Tumor Mask")
        plt.axis("off")
        plt.legend(['No Tumor', 'Tumor'], fontsize='x-large')

        plt.show()

        # ... (Your code for printing the prediction) ...
        if predicted_mask[0, 0].detach().numpy().mean() > 0.5:
            print("The MRI scan shows a tumor.")
        else:
            print("The MRI scan does not show a tumor.")


def check_xray_abnormality():
    if image_path:
        # Load the model (ensure the model is defined before calling this function)
        model = XRayAbnormalityModel()

        # ... (Your code for loading the image and generating the prediction) ...
        xray_image = load_xray_image(image_path)
        input_tensor = torch.from_numpy(xray_image).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            output = model(input_tensor)

        predicted_class = torch.argmax(output).item()

        # ... (Your code for displaying the results) ...
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(xray_image, cmap="gray")
        plt.title("Original X-ray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        if predicted_class == 0:
            plt.text(0.5, 0.5, "Normal", ha='center', va='center', fontsize=18)
        else:
            plt.text(0.5, 0.5, "Abnormal", ha='center', va='center', fontsize=18)
        plt.axis("off")
        plt.title("Prediction")

        plt.show()


# Create main window
window = tk.Tk()
window.title("Medical Image Analyzer")
window.geometry("800x600")

# Image label to display the uploaded image
image_label = tk.Label(window)
image_label.pack()

# Button to upload an image
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack()

# Button to check for brain tumor
tumor_button = tk.Button(window, text="Check for Brain Tumor", command=check_brain_tumor)
tumor_button.pack()

# Button to check for X-ray abnormalities
xray_button = tk.Button(window, text="Check X-ray for Abnormalities", command=check_xray_abnormality)
xray_button.pack()


# Start the main event loop
window.mainloop()

