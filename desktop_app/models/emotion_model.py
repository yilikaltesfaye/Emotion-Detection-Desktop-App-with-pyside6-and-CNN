import cv2
import numpy as np
import tensorflow as tf


class EmotionModel:
    def __init__(self, model_path, emotion_labels):
        self.model = tf.keras.models.load_model(model_path)
        self.emotion_labels = emotion_labels
        self.inference_counter = 0

    def predict(self, face_roi):
        """Predict emotion from face ROI (grayscale)."""
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.reshape(face_normalized, (1, 48, 48, 1))

        predictions = self.model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]

        return emotion_idx, confidence

    def should_run_inference(self):
        """Frame skipping: run every 2nd frame."""
        self.inference_counter += 1
        return self.inference_counter % 2 == 0

    def reset_inference_counter(self):
        self.inference_counter = 0
