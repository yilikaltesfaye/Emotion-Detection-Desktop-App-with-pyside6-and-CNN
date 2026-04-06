import os
import tensorflow as tf
import cv2
import numpy as np

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "image.png")

model = tf.keras.models.load_model(
    os.path.join(script_dir, "..", "models_store", "emotion_cnn.keras")
)

img = cv2.imread(image_path)
if img is None:
    print(f"ERROR: Could not load image at {image_path}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = cv2.resize(gray, (48, 48))
face_norm = face / 255.0
input_data = np.reshape(face_norm, (1, 48, 48, 1))
pred = model.predict(input_data, verbose=0)[0]
print("Raw predictions:", pred)
print("Index with highest:", np.argmax(pred))
