# Handwritten Digit Recognition System

A deep-learning application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The Streamlit interface supports both drawing a digit and uploading an image.

## Features

- CNN model built with TensorFlow/Keras
- MNIST training and test evaluation
- Interactive drawing canvas
- PNG/JPG/JPEG upload prediction
- Confidence and probability distribution for all ten digits
- Accuracy/loss plots, confusion matrix, and classification report
- Modular Python source code

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 (recommended)
- VS Code with the Python extension
- Internet connection for the first MNIST download and package installation

## Setup in VS Code

Open this project folder in VS Code. Open Terminal → New Terminal and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

## Train the model

Run this once (or whenever you want to retrain):

```bash
python train_model.py
```

The script downloads MNIST through Keras if needed, trains the CNN, evaluates it on the test set, and saves:

- `models/digit_recognition_model.keras`
- `outputs/metrics.json`
- `outputs/classification_report.txt`
- `outputs/accuracy_plot.png`
- `outputs/loss_plot.png`
- `outputs/confusion_matrix.png`

Training time depends on your computer. The first run requires an internet connection to download MNIST.

## Run the application

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) and may open it in your browser.

## Project structure

```text
handwritten-digit-recognition/
├── app.py
├── train_model.py
├── predict.py
├── preprocess.py
├── requirements.txt
├── README.md
├── models/
├── notebooks/
│   └── model_exploration.ipynb
├── assets/
├── outputs/
└── screenshots/
```

## CNN architecture

Input (28×28×1) → Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(128) → Dropout(0.3) → Softmax(10).

The model uses Adam optimization and sparse categorical cross-entropy loss. Training uses a validation split and early stopping. Reported accuracy is generated from the actual test evaluation; no metric is hardcoded.

## Notes on uploaded images

The model was trained on centered MNIST digits. For best results, upload an image with one large, dark handwritten digit on a plain light background, or draw a single digit on the dark canvas using the white brush. Preprocessing attempts to invert the image when needed, isolate the foreground, and center it in a 28×28 image. Unusual backgrounds or multiple digits may reduce accuracy.

## Troubleshooting

- **Model not found:** run `python train_model.py` before opening prediction pages.
- **TensorFlow installation issue:** confirm Python is 3.10/3.11 and that the virtual environment is active.
- **Canvas dependency issue:** confirm `streamlit-drawable-canvas` installed successfully using `pip show streamlit-drawable-canvas`.
- **Port already in use:** run `streamlit run app.py --server.port 8502`.

## Testing checklist

- [ ] Dependencies install without errors.
- [ ] `python train_model.py` completes and saves the model.
- [ ] Streamlit starts with `streamlit run app.py`.
- [ ] A drawn digit returns a prediction.
- [ ] A single-digit PNG/JPG upload returns a prediction.
- [ ] Model performance page displays generated evaluation results.

## Future improvements

- Support multi-digit sequence recognition.
- Add image augmentation and compare model architectures.
- Add model export and deployment.
- Add explainability visualizations for CNN feature maps.

## Author

**Harsh Bind**  
B.Tech Computer Science – Data Science & Artificial Intelligence
