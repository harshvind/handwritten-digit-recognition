"""Streamlit interface for the Handwritten Digit Recognition System."""
from pathlib import Path
import json
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from preprocess import preprocess_pil_image, preprocess_canvas_rgba
from predict import load_model, predict_digit

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 16px;
       background: linear-gradient(120deg, #101d36, #203d67); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin-bottom: .35rem;}
.small-muted {color: #64748b; font-size: .95rem;}
.result-number {font-size: 5rem; font-weight: 800; line-height: 1.1; color: #2563eb;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

def show_prediction(model, processed):
    digit, confidence, probabilities = predict_digit(model, processed)
    st.success("Prediction completed.")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f'<div class="result-number">{digit}</div>', unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.2f}%")
    with c2:
        st.subheader("Probability for each digit")
        st.bar_chart({str(i): float(probabilities[i] * 100) for i in range(10)})
    return digit

def model_required():
    try:
        return get_model()
    except Exception as exc:
        st.error(str(exc))
        st.info("First install the requirements and run `python train_model.py`.")
        return None

st.sidebar.title("✍️ Digit AI")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Draw Digit", "Upload Image", "Model Performance", "About"],
)
st.sidebar.caption("CNN • TensorFlow/Keras • MNIST")

if page == "Home":
    st.markdown("""
    <div class="hero">
      <h1>Handwritten Digit Recognition</h1>
      <p>An AI-powered application that recognizes handwritten digits using Deep Learning and CNN.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("Draw a digit or upload an image, and the trained neural network will predict the number from 0 to 9.")
    cols = st.columns(3)
    cols[0].metric("Classes", "10", "Digits 0–9")
    cols[1].metric("Dataset", "MNIST", "28 × 28 grayscale")
    cols[2].metric("Model", "CNN", "Convolutional Neural Network")
    st.subheader("How it works")
    st.write("1. The image is converted to grayscale and normalized.")
    st.write("2. Preprocessing centers the digit on a 28 × 28 canvas.")
    st.write("3. A CNN predicts probabilities for all ten digit classes.")
    st.write("4. The class with the highest probability is displayed.")
    st.info("If this is your first run, train the model with `python train_model.py` before making predictions.")

elif page == "Draw Digit":
    st.title("Draw a digit")
    st.write("Use your mouse or touchpad to draw one large digit in the canvas.")
    brush = st.slider("Brush size", 8, 30, 18)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=brush,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas",
        display_toolbar=True,
    )
    if st.button("Recognize digit", type="primary"):
        model = model_required()
        if model is not None:
            try:
                processed = preprocess_canvas_rgba(canvas_result.image_data)
                with st.expander("See processed 28 × 28 image"):
                    st.image(processed[0, :, :, 0], clamp=True, width=180)
                show_prediction(model, processed)
            except Exception as exc:
                st.warning(str(exc))

elif page == "Upload Image":
    st.title("Upload a handwritten digit")
    st.write("Upload an image containing one clearly visible handwritten digit.")
    uploaded = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"],
    )
    st.caption("Maximum accepted image size: 10 MB. Files above this limit are rejected.")
    if uploaded is not None:
        MAX_UPLOAD_BYTES = 10 * 1024 * 1024
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("This image is larger than 10 MB. Please upload a smaller image.")
            st.stop()
        try:
            image = Image.open(uploaded)
            image.verify()
            uploaded.seek(0)
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", width=300)
            if st.button("Predict uploaded digit", type="primary"):
                model = model_required()
                if model is not None:
                    processed = preprocess_pil_image(image)
                    st.subheader("Processed model input (28 × 28)")
                    st.image(processed[0, :, :, 0], clamp=True, width=180)
                    show_prediction(model, processed)
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")

elif page == "Model Performance":
    st.title("Model performance")
    metrics_path = OUTPUT_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("Evaluation results are not available yet. Run `python train_model.py` first.")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.2f}%")
        b.metric("Test loss", f"{metrics['test_loss']:.4f}")
        c.metric("Epochs trained", str(metrics["epochs_trained"]))
        for filename, title in [
            ("accuracy_plot.png", "Training and validation accuracy"),
            ("loss_plot.png", "Training and validation loss"),
            ("confusion_matrix.png", "Confusion matrix"),
        ]:
            path = OUTPUT_DIR / filename
            if path.exists():
                st.subheader(title)
                st.image(str(path), use_column_width=True)
        report_path = OUTPUT_DIR / "classification_report.txt"
        if report_path.exists():
            with st.expander("View classification report"):
                st.code(report_path.read_text(encoding="utf-8"))

elif page == "About":
    st.title("About this project")
    st.write("""
    This project classifies handwritten numerical digits (0–9). It uses the
    MNIST dataset, a widely used collection of 28 × 28 grayscale digit images.
    A Convolutional Neural Network learns visual patterns such as edges,
    curves, and strokes, then maps an image to one of ten digit classes.
    """)
    st.subheader("CNN architecture")
    st.code("Input (28×28×1) → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool\n"
            "→ Flatten → Dense(128) → Dropout(0.3) → Softmax(10)")
    st.subheader("Technology stack")
    st.write("Python • TensorFlow/Keras • Streamlit • NumPy • Pillow • Matplotlib • Seaborn • scikit-learn")
    st.caption("Model metrics shown in this application are generated by your local training run.")
