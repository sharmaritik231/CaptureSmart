import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from model import EfficientNetRegressionModel  # your model definition

# 1. Load model once and cache it
@st.cache_resource
def load_model():
    model = EfficientNetRegressionModel()
    state = torch.load("model.pth", map_location=torch.device("cpu"))
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
st.title("📷 Camera SS Check")
st.write(
    "Your camera feed is live below. "
    "When you click **Check Capture**, we'll analyze the last frame. "
    "If SS_var & ISO_var ≈ 0, your capture is clean."
)

ctx = webrtc_streamer(
    key="camera", 
    video_processor_factory=FrameProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# threshold for “close to zero”
EPS = 0.01

if st.button("Check Capture"):
    proc = ctx.video_processor
    if proc and hasattr(proc, "last_frame"):
        frame = proc.last_frame

        # preprocess & predict
        inp = preprocess_frame(frame)
        with torch.no_grad():
            ss_var, iso_var = model(inp)

        ss_val = ss_var.item()
        iso_val = iso_var.item()

        # display the frame and the raw scores
        st.image(frame, caption="Analyzed Frame", use_column_width=True)
        st.write(f"**SS_var:** {ss_val:.4f}  **ISO_var:** {iso_val:.4f}")

        # decide clean vs. not correct
        if abs(ss_val) < EPS and abs(iso_val) < EPS:
            st.success("✅ Your capture is clean.")
        else:
            st.error("❌ Camera SS is not correct. Please adjust your camera or lighting and try again.")
    else:
        st.warning("⚠️ Camera feed not ready. Please allow access and wait a moment.")
