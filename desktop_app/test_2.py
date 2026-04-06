import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("models_store/emotion_cnn.keras")
# Create a dummy input (48x48 grayscale, all zeros)
dummy = np.zeros((1, 48, 48, 1))
pred = model.predict(dummy, verbose=0)
print("Output shape:", pred.shape)
print("Predictions for zero input:", pred[0])
# The highest value's index will be the "default" class for blank face
print("Index with highest confidence:", np.argmax(pred[0]))
