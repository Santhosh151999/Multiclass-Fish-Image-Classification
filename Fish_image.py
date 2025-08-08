import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ----- Custom Gradient + Glass Styling -----
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(120deg, #5D3C9C 0%, #37B2BF 35%, #FFCA6A 65%, #286DA8 100%);
        min-height: 100vh;
        padding-top: 0 !important;
    }
    .block-container {
        background: rgba(34,34,51, 0.82);
        border-radius: 18px;
        padding: 2.3rem 2.5rem 2rem 2.5rem;
        box-shadow: 0 8px 32px 0 rgba(60, 60, 120, 0.29);
        color: #F6F7FF;
        margin-top: 0 !important;  /* true top of page */
        margin-bottom: 32px;
        backdrop-filter: blur(12px);
    }
    /* Remove Streamlit default header/footer/background lines */
    header, footer, .css-18e3th9 { display: none; }

    /* Modern title styling */
    .title-style {
        font-family: "Montserrat", "Segoe UI", Arial, sans-serif;
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #FFA940;
        text-align: left;
        margin-bottom: 0.4em;
        margin-top: 0;
        text-shadow: 0 2px 20px rgba(60,60,120,0.15);
        letter-spacing: 0.03em;
        display: flex;
        align-items: center;
        gap: 0.7em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Model Definition -----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 11),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class_names = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped__d_mullet",
    "fish sea_food trout",
]

@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("fish_classifier_v1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image = image.convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

def predict(image, model):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        return predicted_class, confidence.item()

# ----- UI - True Top, Stylish Title with Sticker -----
st.markdown('<div class="block-container">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="title-style">
        🐠 Multiclass Fish Image Classification
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h3 style="color:#FFCA6A;">About This Site</h3>
    <p>
    Discover the fascinating world of fish species with our AI-powered classifier! Upload a fish image, and instantly get the species prediction and a confidence score.<br>
    <br>✨ Features:
    <ul>
        <li>Trained on 11 unique fish categories</li>
        <li>Fast and accurate classification</li>
        <li>Privacy-focused: uploaded images are never saved</li>
    </ul>
    </p>
    <hr/>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📂 Upload an image of a fish (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image and predicting fish species..."):
        pred_class, confidence = predict(image, model)

    st.success(f"### Predicted Fish Category: {pred_class}")
    st.info(f"Confidence Score: {confidence:.2%}")

else:
    st.info("Please upload a fish image above to get started.")

st.markdown("""
<hr>
<p style='font-size:14px;color:#F6F7FF;'>Fish classifier powered by advanced AI, with a fresh frosted-glass look!</p>
</div>
""", unsafe_allow_html=True)
