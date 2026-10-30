"""Model loading and prediction helpers."""
from pathlib import Path
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / "models" / "digit_recognition_model.keras"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Open the VS Code terminal and run: python train_model.py"
        )
    return tf.keras.models.load_model(MODEL_PATH)

def predict_digit(model, processed_image: np.ndarray):
    """Return predicted digit, confidence percentage, and all ten probabilities."""
    if processed_image.shape != (1, 28, 28, 1):
        raise ValueError("Expected model input with shape (1, 28, 28, 1).")
    probabilities = model.predict(processed_image, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit] * 100)
    return predicted_digit, confidence, probabilities
