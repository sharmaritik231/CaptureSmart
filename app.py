import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timm
from torch import nn
from av import VideoFrame

# 1. Model Definition
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

# 3. Video Processor
class FrameProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = EfficientNetRegressionModel()
        state = torch.load("models/best_model_ss.pth", map_location=torch.device("cpu"))
        self.model.load_state_dict(state)
        self.model.eval()

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        inp = preprocess_frame(img)
        with torch.no_grad():
            ss_var, iso_var = self.model(inp)
        ss_val, iso_val = ss_var.item(), iso_var.item()
        display = img.copy()
        text = f"SS Change: {ss_val:.3f}, ISO Change: {iso_val:.3f}"
        cv2.putText(display, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return VideoFrame.from_ndarray(display, format="bgr24")

# 4. Streamlit UI
st.title("Problem 3: CaptureSmart AI – Blur-Aware Mobile Camera Control")
st.markdown(
    """
    **Background:**  
    Capturing crisp photos in motion-heavy or low-light environments is tough. Auto camera settings often fail under blur-inducing scenarios.

    The goal is to detect blur using AI and dynamically adjust **shutter speed** and **ISO** settings to optimize image sharpness.

    **Use the sliders below to tweak camera parameters in real time; predictions overlay on the live feed.**
    """
)

# 5. User Controls
shutter_ms = st.slider("Shutter Speed (ms)", min_value=1, max_value=1000, value=100)
iso_setting = st.slider("ISO Sensitivity", min_value=100, max_value=3200, value=400)

# 6. Camera Constraints
video_constraints = {
    "width": 640,
    "height": 480,
    "frameRate": 30,
    "advanced": [
        {"exposureTime": float(shutter_ms)},
        {"isoSensitivity": float(iso_setting)}
    ]
}

# 7. Start Stream
ctx = webrtc_streamer(
    key=f"camera_{shutter_ms}_{iso_setting}",
    video_processor_factory=FrameProcessor,
    media_stream_constraints={"video": video_constraints, "audio": False},
)

if not ctx.state.playing:
    st.warning("⚠️ Starting camera... please wait or allow access.")
