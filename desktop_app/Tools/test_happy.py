import numpy as np
import onnxruntime as ort
import cv2

# ========= CONFIG =========
MODEL_PATH = "models_store/best_fer_student.onnx"
IMAGE_PATH = "desktop_app/image.png"  # put a face image here
NORMALIZATION = "scale"  # change this: "imagenet", "half", "none"
APPLY_SOFTMAX = False  # try True / False
# ==========================

emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Load and preprocess image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Image not found. Place a face image as image.jpg")
    exit()

# Resize to model input size (128x128 assumed)
img_resized = cv2.resize(img, (128, 128))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)

# Apply normalization
if NORMALIZATION == "imagenet":
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb / 255.0 - mean) / std
elif NORMALIZATION == "scale":
    img_norm = img_rgb / 255.0
elif NORMALIZATION == "half":
    img_norm = (img_rgb / 255.0 - 0.5) / 0.5
elif NORMALIZATION == "none":
    img_norm = img_rgb
else:
    raise ValueError("Unknown NORMALIZATION")

# Transpose to (C, H, W) and add batch dimension
input_tensor = np.expand_dims(img_norm.transpose(2, 0, 1), axis=0)

# Run inference
outputs = session.run(None, {input_name: input_tensor})
logits = outputs[0][0]  # shape (7,)

# Apply softmax if needed
if APPLY_SOFTMAX:
    probs = np.exp(logits) / np.sum(np.exp(logits))
else:
    probs = logits

# Show results
print("Logits:", logits)
print("Probabilities:", probs)
pred_idx = np.argmax(probs)
print(f"Prediction: {emotions[pred_idx]} ({probs[pred_idx]:.4f})")
