import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
from datetime import datetime

MODEL_PATHS = ["waste_classifier.keras", "best_model.keras"]

# Load your trained model (adjust the path if needed)
@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str):
    return tf.keras.models.load_model(path, compile=False)

def load_model():
    last_exc = None
    for path in MODEL_PATHS:
        try:
            return _load_model_cached(path)
        except Exception as exc:
            last_exc = exc
            _load_model_cached.clear()
    raise RuntimeError(
        "Failed to load the TensorFlow/Keras model from any known path. "
        "Ensure tensorflow==2.15.0, keras==2.15.0, and that the model was "
        "exported with Keras 2.x (or re-export as a SavedModel or H5 file)."
    ) from last_exc

try:
    model = load_model()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

# Define the target image size (same as used during training)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define your class mapping (update if necessary)
class_indices = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
idx2label = {v: k for k, v in class_indices.items()}

def load_and_preprocess_image(image_data):
    """Load an image from file bytes and preprocess it."""
    # Open the image using PIL
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    # Resize the image
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    # Convert image to numpy array
    img_array = image.img_to_array(img_resized)
    # Expand dimensions to match model input (1, IMG_HEIGHT, IMG_WIDTH, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image using MobileNetV2's preprocessing
    img_array = preprocess_input(img_array)
    return img_array, img

# -------------------------------
# UI: Styling
# -------------------------------
st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg: #0c1220;
        --bg-2: #111a2f;
        --card: #151f33;
        --card-2: #18243b;
        --text: #eef3ff;
        --muted: #9fb0c9;
        --accent: #72d97e;
        --accent-2: #5ccf9b;
        --border: rgba(255, 255, 255, 0.08);
        --shadow: 0 10px 30px rgba(0,0,0,0.35);
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--text);
    }

    .stApp {
        background: radial-gradient(1200px 600px at 20% -10%, #1b2a4b 0%, transparent 60%),
                    radial-gradient(1000px 500px at 90% -10%, #0b3b2a 0%, transparent 55%),
                    linear-gradient(180deg, var(--bg) 0%, #0a0f1a 100%);
    }

    .app-container {
        max-width: 1120px;
        margin: 0 auto;
        padding: 1.5rem 1.2rem 3rem 1.2rem;
    }

    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.2rem;
        border-radius: 14px;
        background: linear-gradient(90deg, rgba(114,217,126,0.12), rgba(92,207,155,0.06));
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        margin-bottom: 1.6rem;
    }

    .navbar .brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }

    .navbar .brand .dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--accent);
        box-shadow: 0 0 18px rgba(114,217,126,0.7);
    }

    .hero {
        display: grid;
        grid-template-columns: 1.4fr 1fr;
        gap: 1.5rem;
        margin-bottom: 1.8rem;
    }

    .hero-card {
        padding: 1.6rem 1.6rem 1.2rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(21,31,51,0.95), rgba(18,28,46,0.9));
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }

    .hero-card:before {
        content: "";
        position: absolute;
        top: -120px;
        right: -120px;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(114,217,126,0.18), transparent 60%);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1rem;
        max-width: 520px;
    }

    .hero-meta {
        display: flex;
        gap: 0.8rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .chip {
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(114,217,126,0.12);
        border: 1px solid rgba(114,217,126,0.35);
        color: var(--accent);
        font-size: 0.85rem;
        font-weight: 600;
    }

    .card {
        padding: 1.2rem 1.2rem;
        border-radius: 16px;
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }

    .upload-area {
        background: linear-gradient(135deg, rgba(24,36,59,0.9), rgba(18,26,44,0.9));
        border: 1px dashed rgba(255,255,255,0.2);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    .upload-area:hover {
        border-color: rgba(114,217,126,0.6);
        box-shadow: 0 0 0 2px rgba(114,217,126,0.12);
    }

    .upload-icon {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }

    .upload-title {
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .upload-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(114,217,126,0.15);
        color: var(--accent);
        font-weight: 700;
        font-size: 0.85rem;
    }

    .muted {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .footer {
        text-align: center;
        color: var(--muted);
        margin-top: 2rem;
        font-size: 0.85rem;
    }

    .history-card {
        background: var(--card-2);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        border: 1px solid var(--border);
        margin-bottom: 0.6rem;
    }

    .success-anim {
        animation: glow 1.2s ease-in-out;
    }

    @keyframes glow {
        0% { box-shadow: 0 0 0 rgba(114,217,126,0); }
        50% { box-shadow: 0 0 24px rgba(114,217,126,0.3); }
        100% { box-shadow: 0 0 0 rgba(114,217,126,0); }
    }

    @media (max-width: 900px) {
        .hero { grid-template-columns: 1fr; }
        .result-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.markdown("## Smart Waste Classification")
    st.markdown(
        """
        **Eco-tech assistant** that helps identify waste and suggests responsible disposal.
        """
    )
    st.markdown("---")
    st.markdown("**Supported Types**")
    st.markdown(
        """
        - Cardboard
        - Glass
        - Metal
        - Paper
        - Plastic
        - Trash
        """
    )
    st.markdown("---")
    st.markdown("**Recycling Tips**")
    st.markdown(
        """
        - Rinse containers before recycling
        - Separate by material type
        - Keep cardboard dry
        """
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        """
        This project uses a MobileNetV2-based model fine-tuned for waste categories.
        """
    )

st.markdown('<div class="app-container">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="navbar">
        <div class="brand">
            <div class="dot"></div>
            Smart Waste Classification
        </div>
        <div class="muted">Operational • v1.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-card">
            <div class="hero-title">Smart Waste Classification &amp; Recycling Suggestions ♻️</div>
            <div class="hero-subtitle">
                Upload a waste image and receive accurate classification with responsible disposal guidance.
                Built for clarity, speed, and sustainable impact.
            </div>
            <div class="hero-meta">
                <span class="chip">AI-Powered</span>
                <span class="chip">Eco-Tech</span>
                <span class="chip">MobileNetV2</span>
            </div>
        </div>
        <div class="card">
            <div style="font-weight:700; margin-bottom:0.4rem;">Quick Overview</div>
            <div class="muted">Upload an image to identify waste and view tailored recycling guidance.</div>
            <div style="margin-top:0.8rem;" class="muted">Tip: Use well-lit, centered images for best accuracy.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="upload-area">
        <div class="upload-icon">📤</div>
        <div class="upload-title">Drag &amp; drop your waste image</div>
        <div class="upload-subtitle">Supported formats: JPG, JPEG, PNG</div>
    </div>
    """,
    unsafe_allow_html=True,
)
# File uploader for image input
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Read the image file
    image_data = uploaded_file.read()
    # Preprocess the image and get original image for display
    img_array, original_img = load_and_preprocess_image(image_data)

    # Make prediction with the model
    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        predicted_label = idx2label[predicted_class]
        confidence = predictions[0][predicted_class]
    
    # Expanded recycling suggestions with more detailed information
    recycling_suggestions = {
        'cardboard': (
            "Recycle cardboard by flattening the boxes and keeping them dry. "
            "Place them in the designated cardboard recycling bin. "
            "Ensure there is no food residue attached."
        ),
        'glass': (
            "Glass should be cleaned and sorted by color (if required by your local guidelines). "
            "Place it in the glass recycling container. "
            "Avoid mixing with ceramics or mirrors."
        ),
        'metal': (
            "Metals such as aluminum and steel should be rinsed and sorted. "
            "Recycle them in the metal recycling bin. "
            "Scrap metal collectors may offer better returns for large quantities."
        ),
        'paper': (
            "Recycle paper by ensuring it is clean and dry. "
            "Flatten and bundle paper items before placing them in the paper recycling bin. "
            "Avoid mixing with contaminated or greasy paper products."
        ),
        'plastic': (
            "Plastics should be rinsed to remove food residues and then sorted by type if possible. "
            "Check for the recycling symbol on the plastic. "
            "Place in the appropriate plastic recycling bin."
        ),
        'trash': (
            "If an item cannot be recycled, dispose of it as general waste. "
            "Consider ways to reduce waste or reuse items before discarding. "
            "Consult your local waste management guidelines for hazardous or electronic waste."
        )
    }
    
    suggestion = recycling_suggestions.get(predicted_label, "No suggestion available.")

    icon_map = {
        "cardboard": "📦",
        "glass": "🧴",
        "metal": "🔩",
        "paper": "📄",
        "plastic": "🧃",
        "trash": "🗑️",
    }

    label_icon = icon_map.get(predicted_label, "♻️")

    st.markdown("<div class=\"card success-anim\">", unsafe_allow_html=True)
    st.markdown("### Results", unsafe_allow_html=True)

    st.markdown("<div class=\"result-grid\">", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.image(original_img, caption="Uploaded Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.markdown(
            f"<div class=\"badge\">{label_icon} {predicted_label.title()}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style=\"margin-top:0.6rem; font-weight:700;\">Confidence</div>")
        st.progress(float(confidence))
        st.markdown(f"<div class=\"muted\">{confidence * 100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class=\"card\">", unsafe_allow_html=True)
        st.markdown("<div style=\"font-weight:700;\">Recycling Suggestion</div>")
        st.markdown(f"<div class=\"muted\" style=\"margin-top:0.4rem;\">{suggestion}</div>")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Expandable recycling tips
    with st.expander("More recycling tips"):
        st.markdown(
            """
            - Clean containers before recycling
            - Remove food residue from cardboard and paper
            - Sort by local guidelines for best results
            """
        )

    # Prediction history
    st.session_state.history.insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "label": predicted_label.title(),
            "confidence": f"{confidence * 100:.2f}%",
        },
    )

    if len(st.session_state.history) > 8:
        st.session_state.history = st.session_state.history[:8]

    st.markdown("### Prediction History")
    for item in st.session_state.history:
        st.markdown(
            f"""
            <div class="history-card">
                <div style="font-weight:700;">{item['label']} <span class="muted">({item['confidence']})</span></div>
                <div class="muted">{item['time']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Download results
    download_text = (
        f"Prediction: {predicted_label.title()}\n"
        f"Confidence: {confidence * 100:.2f}%\n"
        f"Suggestion: {suggestion}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    st.download_button(
        label="Download Result",
        data=download_text,
        file_name="waste_classification_result.txt",
        mime="text/plain",
    )

st.markdown(
    """
    <div class="footer">
        Built for sustainable waste management • Smart Waste Classification
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
