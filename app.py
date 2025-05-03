import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timm
from torch import nn
from av import VideoFrame

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

@st.cache_resource
def load_model():
    model = EfficientNetRegressionModel()
    state = torch.load("models/best_model_ss.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# preprocessing
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

# video processor
class FrameProcessor(VideoTransformerBase):
    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.last_frame = img
        return VideoFrame.from_ndarray(img, format="bgr24")

# UI
st.title("Problem 3: CaptureSmart AI – Blur-Aware Mobile Camera Control")
st.markdown(
    """
    **Background:**  
    Capturing crisp photos in motion-heavy or low-light environments is tough...
    """
)

# sliders for camera controls
shutter_ms = st.slider("Shutter Speed (ms)", min_value=1, max_value=1000, value=100)
iso_val    = st.slider("ISO Sensitivity",    min_value=100, max_value=3200, value=400)

# build constraints
video_constraints = {
    "width": 640,
    "height": 480,
    "frameRate": 30,
    # advanced constraints (browser support varies)
    "advanced": [
        {"exposureTime": float(shutter_ms)},
        {"isoSensitivity": float(iso_val)}
    ]
}

ctx = webrtc_streamer(
    key=f"camera_{shutter_ms}_{iso_val}",
    video_processor_factory=FrameProcessor,
    media_stream_constraints={"video": video_constraints, "audio": False},
)

EPS = 0.01

if ctx.video_processor and hasattr(ctx.video_processor, "last_frame"):
    frame = ctx.video_processor.last_frame
    inp = preprocess_frame(frame)
    with torch.no_grad(): ss_var, iso_var = model(inp)
    ss_val, iso_val_pred = ss_var.item(), iso_var.item()
    st.image(frame, caption="Live Analyzed Frame", use_column_width=True)
    st.write(f"**SS_var:** {ss_val:.4f} **ISO_var:** {iso_val_pred:.4f}")
    if abs(ss_val) < EPS and abs(iso_val_pred) < EPS:
        st.success("✅ Your capture is clean.")
    else:
        st.error("❌ Camera SS is not correct. Adjust settings and try again.")
else:
    st.warning("⚠️ Starting camera... please wait.")
