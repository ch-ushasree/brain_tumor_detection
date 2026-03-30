from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import base64

app = Flask(__name__, template_folder='../templates')


# ─── Brain Tumor Segmentation Model (U-Net style) ─────────────────────────────

class TumorSegmentationModel(nn.Module):
    def __init__(self):
        super(TumorSegmentationModel, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        mid  = self.middle(enc2)
        dec1 = self.decoder1(mid)
        dec2 = self.decoder2(dec1)
        return self.dropout(dec2)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image(file) -> np.ndarray:
    """Load uploaded image and convert to grayscale numpy array."""
    image    = Image.open(file)
    image_np = np.array(image)
    if image_np.ndim == 3:
        image_np = np.mean(image_np, axis=2)
    return image_np.astype(np.float32)


def to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 PNG for browser display."""
    img_norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, buffer = cv2.imencode('.png', img_norm)
    return base64.b64encode(buffer).decode('utf-8')


def build_overlay(mri: np.ndarray, mask: np.ndarray) -> str:
    """Blend tumor mask onto MRI scan — red = tumor, green = healthy."""
    mri_norm   = cv2.normalize(mri, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mri_3d     = np.stack([mri_norm] * 3, axis=2)
    color_mask = np.zeros_like(mri_3d)
    color_mask[mask > 0.5]  = [220, 50,  50]   # red   → tumor
    color_mask[mask <= 0.5] = [50,  200, 100]   # green → healthy
    blended    = cv2.addWeighted(mri_3d, 0.65, color_mask, 0.35, 0)
    _, buffer  = cv2.imencode('.png', blended)
    return base64.b64encode(buffer).decode('utf-8')


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file      = request.files['image']
    mri_image = load_image(file)

    # Model — replace model = TumorSegmentationModel() with
    # model.load_state_dict(torch.load('model.pth')) after training
    model = TumorSegmentationModel()
    model.eval()

    input_tensor = torch.from_numpy(mri_image).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        predicted_mask = model(input_tensor)

    mask_np      = predicted_mask[0, 0].detach().numpy()
    mask_resized = cv2.resize(mask_np, (mri_image.shape[1], mri_image.shape[0]))
    avg          = float(mask_resized.mean())

    if avg > 0.55:
        verdict, severity = 'Tumor Detected', 'high'
    elif avg > 0.45:
        verdict, severity = 'Uncertain — Consult a Doctor', 'medium'
    else:
        verdict, severity = 'No Tumor Detected', 'low'

    return jsonify({
        'verdict':      verdict,
        'severity':     severity,
        'confidence':   round(avg * 100, 2),
        'original_img': to_base64(mri_image),
        'overlay_img':  build_overlay(mri_image, mask_resized),
        'mask_img':     to_base64(mask_resized * 255),
    })


if __name__ == '__main__':
    app.run(debug=True, port=8000)
