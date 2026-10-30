"""Image preprocessing helpers for uploaded images and drawing-canvas output."""
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def _prepare_grayscale(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    # Smooth tiny compression/noise artifacts without changing the digit shape much.
    return image.filter(ImageFilter.MedianFilter(size=3))

def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """
    Convert an image containing one handwritten digit into MNIST-like input.
    Returns a float32 array shaped (1, 28, 28, 1).
    """
    gray = _prepare_grayscale(image)
    arr = np.asarray(gray, dtype=np.uint8)

    # Estimate foreground polarity using image border brightness.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    background_is_light = float(np.median(border)) > 127
    ink = 255 - arr if background_is_light else arr

    # Threshold weak background noise and locate the digit's bounding box.
    threshold = max(25, int(np.percentile(ink, 75) * 0.35))
    mask = ink > threshold
    if not np.any(mask):
        raise ValueError("No clear digit was found. Try a darker, clearer image.")

    ys, xs = np.where(mask)
    cropped = Image.fromarray(ink[ys.min():ys.max()+1, xs.min():xs.max()+1])

    # Preserve aspect ratio and center the digit in a 20x20 area of a 28x28 canvas.
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2
    canvas.paste(cropped, (x, y))

    # Match MNIST's bright digit on a dark background and normalize.
    result = np.asarray(canvas, dtype=np.float32) / 255.0
    return result.reshape(1, 28, 28, 1)

def preprocess_canvas_rgba(canvas_rgba: np.ndarray) -> np.ndarray:
    """Convert streamlit-drawable-canvas RGBA pixels into model input."""
    if canvas_rgba is None or canvas_rgba.size == 0:
        raise ValueError("Draw a digit on the canvas before predicting.")
    rgba = np.asarray(canvas_rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        raise ValueError("The drawing canvas returned an unsupported image.")
    rgb = rgba[:, :, :3]
    # Canvas is configured with a black background and white brush.
    gray = np.max(rgb, axis=2).astype(np.uint8)
    return preprocess_pil_image(Image.fromarray(gray))
