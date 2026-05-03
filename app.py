import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

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

# Streamlit App Layout
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;700&display=swap');

    :root {
        --bg: #0c0f14;
        --bg-accent: #121723;
        --card: #151b29;
        --card-2: #1b2233;
        --text: #e8edf6;
        --muted: #b7c0d6;
        --accent: #ffb347;
        --accent-2: #ff6f91;
        --ring: rgba(255, 179, 71, 0.35);
    }

    .stApp {
        background: radial-gradient(1200px 600px at 20% -10%, #1b2336 0%, transparent 55%),
                    radial-gradient(900px 500px at 90% 10%, #3a1b2a 0%, transparent 60%),
                    linear-gradient(180deg, var(--bg) 0%, #0b0e13 100%);
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        max-width: 920px;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Fraunces', serif;
        letter-spacing: 0.2px;
    }

    h1 {
        font-size: 2.4rem;
        color: var(--text);
        text-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    p, .stMarkdown, .stText, .stWrite {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.6;
    }

    .stFileUploader {
        background: linear-gradient(135deg, var(--card), var(--card-2));
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
        animation: rise 700ms ease-out;
    }

    .stFileUploader label {
        color: var(--text);
        font-weight: 600;
    }

    .stImage img {
        border-radius: 14px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stAlert {
        border-radius: 12px !important;
    }

    .stMarkdown strong {
        color: var(--text);
    }

    .stMarkdown hr {
        border-color: rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] {
        background: #0e1420;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: #0c0f14;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 10px 30px var(--ring);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }

    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 14px 36px var(--ring);
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stMarkdown, .stImage, .stFileUploader, .stWrite {
        animation: rise 600ms ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Waste Classification & Recycling Suggestion System")
st.write("Upload an image of waste, and the model will predict its class along with detailed recycling instructions.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image_data = uploaded_file.read()
    # Preprocess the image and get original image for display
    img_array, original_img = load_and_preprocess_image(image_data)
    
    # Make prediction with the model
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    predicted_label = idx2label[predicted_class]
    confidence = predictions[0][predicted_class]
    
    # Display results in Streamlit with a reduced image width (e.g., 300 pixels)
    st.image(original_img, caption="Uploaded Image", width=300)
    st.write(f"**Prediction:** {predicted_label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    
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
    st.write(f"**Recycling Suggestion:** {suggestion}")
