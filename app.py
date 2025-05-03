import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import cv2
import numpy as np
from model import EfficientNetRegressionModel  # your model class


# Load model
@st.cache_resource
def load_model():
    model = EfficientNetRegressionModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Define preprocessing to match ImageDataset
def preprocess_image_opencv(frame_bgr):
    # Convert BGR (OpenCV) to RGB (PIL expects RGB)
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Resize to (224, 224)
    image = image.resize((224, 224))

    # Convert to grayscale and replicate channels to get RGB
    image = image.convert("L")  # grayscale
    image = Image.merge("RGB", (image, image, image))

    # Apply tensor and normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]


# Camera frame processor
class FrameProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img
        return img

# Streamlit UI
st.title("SS_var and ISO_var Predictor")
st.write("Open your webcam and capture an image to get predictions.")

ctx = webrtc_streamer(key="camera", video_processor_factory=FrameProcessor)

if st.button("Predict from current frame"):
    if ctx.video_processor and hasattr(ctx.video_processor, "last_frame"):
        frame = ctx.video_processor.last_frame

        # Preprocess image using the same steps as ImageDataset
        input_tensor = preprocess_image_opencv(frame)

        # Predict
        with torch.no_grad():
            ss_score, iso_score = model(input_tensor)

        # Display results
        st.image(frame, caption="Captured Frame", use_column_width=True)
        st.success(f"SS_var: {ss_score.item():.4f}")
        st.success(f"ISO_var: {iso_score.item():.4f}")
    else:
        st.warning("Please start the webcam and try again.")
