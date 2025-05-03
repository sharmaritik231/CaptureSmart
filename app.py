import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timm
from torch import nn

class EfficientNetRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNet-Lite0 as feature extractor
        self.feature_extractor = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=0)

        # Output is already [B, 1280]
        self.layer_norm = nn.LayerNorm(1280)

        self.fnn = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU()
        )

        self.head_ss_var  = nn.Linear(512, 1)
        self.head_iso_var = nn.Linear(512, 1)

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 1280]
        x = self.layer_norm(x)
        x = self.fnn(x)
        return self.head_ss_var(x), self.head_iso_var(x)

# 1. Load model once and cache it
@st.cache_resource
def load_model():
    model = EfficientNetRegressionModel()
    state = torch.load("models/best_model_ss.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# 2. Preprocessing function matching your training pipeline
def preprocess_frame(frame_bgr):
    # BGR → RGB → PIL
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # Resize 224×224
    img = img.resize((224, 224))

    # Grayscale → replicate to RGB
    gray = img.convert("L")
    img = Image.merge("RGB", (gray, gray, gray))

    # ToTensor + Normalize(mean=0.5, std=0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(img).unsqueeze(0)  # shape [1,3,224,224]

# 3. Video processor to hold last frame
class FrameProcessor(VideoTransformerBase):
    def transform(self, frame):
        # store most recent BGR frame
        self.last_frame = frame.to_ndarray(format="bgr24")
        return self.last_frame

# 4. Streamlit UI
st.title("Problem 3: CaptureSmart AI – Blur-Aware Mobile Camera Control")
st.markdown(
    """
    **Background:**  
    Capturing crisp photos in motion-heavy or low-light environments is tough. Mobile camera auto-settings often fail to prevent blur from fast-moving objects or shaky hands. Inspired by research like *“Active Exposure Control for Robust Visual Odometry in HDR Environments”*, this challenge aims to enhance mobile photography using AI.

    The goal is to use machine learning or deep learning to analyze image blur and dynamically adjust camera settings—such as shutter speed, exposure, and ISO—to reduce motion blur while preserving brightness and detail.

    **Problem Statement:**  
    Build a mobile app that uses image-based blur detection and AI models to automatically recommend or adjust camera parameters in real time.

    Below, the camera feed is live and analysis runs continuously on the latest frame. If SS_var & ISO_var ≈ 0, the capture is considered clean; otherwise, camera settings may need adjustment.
    """
)

ctx = webrtc_streamer(
    key="camera", 
    video_processor_factory=FrameProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# threshold for “close to zero”
EPS = 0.01

# 5. Continuous real-time prediction on the latest frame
if ctx.video_processor and hasattr(ctx.video_processor, "last_frame"):
    frame = ctx.video_processor.last_frame

    inp = preprocess_frame(frame)
    with torch.no_grad():
        ss_var, iso_var = model(inp)

    ss_val = ss_var.item()
    iso_val = iso_var.item()

    # display the frame and the raw scores
    st.image(frame, caption="Live Analyzed Frame", use_column_width=True)
    st.write(f"**SS_var:** {ss_val:.4f}  **ISO_var:** {iso_val:.4f}")

    # decision
    if abs(ss_val) < EPS and abs(iso_val) < EPS:
        st.success("✅ Your capture is clean.")
    else:
        st.error("❌ Camera SS is not correct. Consider adjusting shutter speed/exposure/ISO.")
else:
    st.warning("⚠️ Starting camera... please allow access and wait a moment.")
