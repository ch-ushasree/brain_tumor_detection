from PIL import ImageTk
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import easygui
import cv2
from tkinter import ttk
import requests
from io import BytesIO


class TumorSegmentationModel(nn.Module):
    def __init__(self):
        super(TumorSegmentationModel, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        mid = self.middle(enc2)
        dec1 = self.decoder1(mid)
        dec2 = self.decoder2(dec1)
        out = self.dropout(dec2)
        return out


def load_mri_image(image_path):
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
  image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
  if image_path:
    image = Image.open(image_path)
    image.thumbnail((400, 400))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

    
def check_brain_tumor():
    if image_path:
        model = TumorSegmentationModel()  

        mri_image = load_mri_image(image_path)
        input_tensor = torch.from_numpy(mri_image).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            predicted_mask = model(input_tensor)

        predicted_mask_np = predicted_mask[0, 0].detach().numpy()

        predicted_mask_resized = cv2.resize(predicted_mask_np, (mri_image.shape[1], mri_image.shape[0]))
        plt.figure(figsize=(6, 6))
        plt.imshow(predicted_mask_resized, cmap='gray')
        plt.title("Predicted Mask")
        plt.show()

        masked_image = mri_image * predicted_mask_resized
        if masked_image.ndim == 2:
            masked_image = masked_image[..., np.newaxis]

        color_masked_image = np.zeros((mri_image.shape[0], mri_image.shape[1], 3), dtype=np.uint8)
        color_masked_image[predicted_mask_resized > 0.5] = (255, 0, 0)  
        color_masked_image[predicted_mask_resized <= 0.5] = (0, 0, 255)  
        masked_image_resized = cv2.resize(masked_image, (color_masked_image.shape[1], color_masked_image.shape[0]))
        masked_image_resized = np.repeat(masked_image_resized[:, :, np.newaxis], 3, axis=2)
        mri_image_normalized = cv2.normalize(mri_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mri_image_3d = np.repeat(mri_image_normalized[:, :, np.newaxis], 3, axis=2)

        combined_image = np.hstack((mri_image_3d, color_masked_image))

        plt.figure(figsize=(12, 6))
        plt.imshow(combined_image)
        plt.title("Original Image (Left) and Masked Image (Right)")
        plt.show()

        if predicted_mask[0, 0].detach().numpy().mean() > 0.5:
            print("The MRI scan shows a tumor.")
        else:
            print("The MRI scan does not show a tumor.")

window = tk.Tk()
window.title("Medical Image Analyzer")
window.geometry("1280x1024")

button_frame = ttk.Frame(window)
button_frame.pack(side=tk.TOP, pady=10)

upload_button = ttk.Button(button_frame, text="Upload Image", command=upload_image)
upload_button.pack(side=tk.LEFT, padx=10)
upload_button.config(width=20)  

tumor_button = ttk.Button(button_frame, text="Check for Brain Tumor", command=check_brain_tumor)
tumor_button.pack(side=tk.LEFT, padx=10)
tumor_button.config(width=20) 


image_label = ttk.Label(window)
image_label.pack(expand=True)  

window.mainloop()
