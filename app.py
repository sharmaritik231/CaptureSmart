import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timm
from torch import nn
from av import VideoFrame

# Define model class
class EfficientNetRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=0)
        self.layer_norm = nn.LayerNorm(1280)
        self.fnn = nn.Sequential(nn.Linear(1280, 512), nn.ReLU())
        self.head_ss_var  = nn.Linear(512, 1)
        self.head_iso_var = nn.Linear(512, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.layer_norm(x)
        x = self.fnn(x)
        return self.head_ss_var(x), self.head_iso_var(x)

# Preprocessing function
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

# Video processor with in-frame prediction overlay
class FrameProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = EfficientNetRegressionModel()
        state = torch.load("models/best_model_ss.pth", map_location=torch.device("cpu"))
        self.model.load_state_dict(state)
        self.model.eval()

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # preprocess & predict
        inp = preprocess_frame(img)
        with torch.no_grad():
            ss_var, iso_var = self.model(inp)
        ss_val = ss_var.item()
        iso_val = iso_var.item()
        # overlay text
        display = img.copy()
        text = f"SS Change: {ss_val:.3f}, ISO Change: {iso_val:.3f}"
        cv2.putText(display, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return VideoFrame.from_ndarray(display, format="bgr24")

# Streamlit UI
st.title("Problem 3: CaptureSmart AI – Blur-Aware Mobile Camera Control")
st.markdown(
    """
    **Background:**  
    Capturing crisp photos in motion-heavy or low-light environments is tough...

    The goal is to use AI to detect blur and adjust camera settings (shutter, exposure, ISO) dynamically.

    **Use the live camera feed below—the SS_var and ISO_var predictions are overlaid in real time.**
    """
)

# Start webcam with real-time overlay
ctx = webrtc_streamer(
    key="camera",
    video_processor_factory=FrameProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
