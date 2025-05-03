import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from model import EfficientNetRegressionModel  # your class definition here

# 1. Load the trained model (model.pth must sit alongside app.py)
@st.cache_resource
def load_model():
    model = EfficientNetRegressionModel()
    state = torch.load("model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# 2. Preprocessing function matching ImageDataset
def preprocess_pil(img_pil):
    # Resize
    img = img_pil.resize((224, 224))
    # Grayscale → replicate to RGB
    gray = img.convert("L")
    img = Image.merge("RGB", (gray, gray, gray))
    # ToTensor + Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(img).unsqueeze(0)  # [1,3,224,224]

st.title("🖼️ SS_var & ISO_var Predictor")
st.write("Take a photo or upload one, then click **Predict** to see your scores.")

# 3. Camera input widget
captured = st.camera_input("📸 Capture an image")

if captured:
    # Display the captured image
    st.image(captured, caption="📷 Your Image", use_column_width=True)

    if st.button("Predict"):
        # Convert Streamlit UploadedFile to PIL.Image
        img = Image.open(captured).convert("RGB")
        # Preprocess
        inp = preprocess_pil(img)
        # Inference
        with torch.no_grad():
            ss_out, iso_out = model(inp)
        # Show results
        st.success(f"**SS_var:** {ss_out.item():.4f}")
        st.success(f"**ISO_var:** {iso_out.item():.4f}")
