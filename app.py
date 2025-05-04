from flask import Flask, render_template, Response, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timm
from torch import nn
import numpy as np

app = Flask(__name__)

# 1. Updated Model Definition
class EfficientNetRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNet-Lite0 as feature extractor
        self.feature_extractor = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=0)

        # Output is already [B, 1280]
        self.layer_norm = nn.LayerNorm(1280)

        self.fnn = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU()
        )

        self.head_ss_var  = nn.Linear(128, 1)
        self.head_iso_var = nn.Linear(128, 1)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)  # [B, 1280]

        # Layer normalization
        x = self.layer_norm(x)

        # Fully connected layers
        x = self.fnn(x)

        # Separate heads for SS and ISO predictions
        ss_var = self.head_ss_var(x)
        iso_var = self.head_iso_var(x)

        return ss_var, iso_var

# 2. Preprocessing
def preprocess_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((224,224))
    gray = img.convert("L")
    img = Image.merge("RGB", (gray, gray, gray))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(img).unsqueeze(0)

# Initialize model
model = EfficientNetRegressionModel()
state = torch.load("best_model_ss.pth", map_location=torch.device("cpu"))
model.load_state_dict(state)
model.eval()

# Variables for Shutter Speed and ISO
shutter_speed = 100  # Default
iso_setting = 400  # Default

# 3. Video Generator
def generate_frames():
    cap = cv2.VideoCapture(0)  # OpenCV captures video from the default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            inp = preprocess_frame(frame)
            with torch.no_grad():
                ss_var, iso_var = model(inp)
            ss_val, iso_val = ss_var.item(), iso_var.item()

            # Overlay predictions on the frame
            display = frame.copy()
            text = f"SS Change: {ss_val:.3f}, ISO Change: {iso_val:.3f}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', display)
            frame = buffer.tobytes()

            # Yield the frame to be displayed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 4. Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global shutter_speed, iso_setting
    shutter_speed = int(request.form.get('shutter_speed', 100))
    iso_setting = int(request.form.get('iso_setting', 400))
    return "Settings updated", 200

if __name__ == '__main__':
    app.run(debug=True)
