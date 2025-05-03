import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from model import EfficientNetRegressionModel  # ensure this file defines your class


# 1. Load the trained model (model.pth should be in the same directory)
@st.cache_resource
def load_model():
    model = EfficientNetRegressionModel()
    state = torch.load("model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()


# 2. Preprocessing function matching your ImageDataset
def preprocess_frame(frame_bgr):
    # BGR (OpenCV) → RGB → PIL Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # Resize
    img = img.resize((224, 224))

    # GRAYSCALE → replicate to 3-channel RGB
    gray = img.convert("L")
    img = Image.merge("RGB", (gray, gray, gray))

    # ToTensor + Normalize(mean=0.5, std=0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    tensor = transform(img).unsqueeze(0)  # shape [1,3,224,224]
    return tensor


# 3. Webcam frame processor
class FrameProcessor(VideoTransformerBase):
    def transform(self, frame):
        # store the latest BGR frame for prediction
        self.last_frame = frame.to_ndarray(format="bgr24")
        return self.last_frame


# 4. Streamlit UI
st.title("🖼️ SS_var & ISO_var Predictor")
st.write("Use your webcam to capture a frame, then click **Predict** to see the scores.")

ctx = webrtc_streamer(
    key="camera",
    video_processor_factory=FrameProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if st.button("Predict"):
    if ctx.video_processor and hasattr(ctx.video_processor, "last_frame"):
        frame = ctx.video_processor.last_frame

        # preprocess & predict
        inp = preprocess_frame(frame)
        with torch.no_grad():
            ss_out, iso_out = model(inp)

        # display
        st.image(frame, caption="Captured Frame", use_column_width=True)
        st.markdown(f"**SS_var:** {ss_out.item():.4f}  **ISO_var:** {iso_out.item():.4f}")
    else:
        st.warning("⚠️ Webcam not active yet—please allow camera access.")
