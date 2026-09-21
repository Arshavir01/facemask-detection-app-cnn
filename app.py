"""Face Mask Detection – Streamlit app.

Loads the CNN trained in Colab, lets the user upload a photo,
and reports whether the person is wearing a mask.
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "face_mask_model.keras"

# Must match the notebook: image size and label encoding
IMG_SIZE = (128, 128)
WITHOUT_MASK, WITH_MASK = 0, 1


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # Imported here so the page can show a friendly error if TensorFlow/model is missing
    from tensorflow import keras

    return keras.models.load_model(MODEL_PATH, compile=False)  # no training, so no compile needed


def preprocess(image: Image.Image) -> np.ndarray:
    """Same steps as training: RGB -> resize 128x128 -> divide by 255.

    NOTE: the notebook trained on PIL images (RGB order), so we use PIL here too.
    (cv2.imread would give BGR and hurt accuracy.)
    """
    image = ImageOps.exif_transpose(image)  # fixes sideways phone photos
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)  # shape (1, 128, 128, 3)


def predict_mask(model, image: Image.Image) -> tuple[int, float]:
    """Return (label, confidence) where label is WITH_MASK or WITHOUT_MASK."""
    scores = model.predict(preprocess(image), verbose=0)[0]
    label = int(np.argmax(scores))
    # The notebook's output layer uses sigmoid, so the two scores don't sum to 1.
    # Normalise them to get an easy-to-read percentage.
    confidence = float(scores[label] / scores.sum())
    return label, confidence


def main():
    st.set_page_config(page_title="Face Mask Detection", page_icon="😷")
    st.title("😷 Face Mask Detection")
    st.write("Upload a photo of a person and the model will tell you whether they are wearing a mask.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: `{MODEL_PATH.name}`. Put it in the same folder as app.py.")
        st.stop()

    model = load_model()

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Waiting for an image. Best results: one person, face clearly visible and fairly close to the camera.")
        return

    try:
        image = Image.open(uploaded)
        image.load()
    except Exception:
        st.error("Sorry, that file could not be read as an image. Please try another one.")
        return

    st.image(image, caption="Uploaded image")

    with st.spinner("Analysing..."):
        label, confidence = predict_mask(model, image)

    if label == WITH_MASK:
        st.success(f"✅ The person is **wearing a mask**  \nConfidence: {confidence:.0%}")
    else:
        st.error(f"⚠️ The person is **not wearing a mask**  \nConfidence: {confidence:.0%}")


if __name__ == "__main__":
    main()
