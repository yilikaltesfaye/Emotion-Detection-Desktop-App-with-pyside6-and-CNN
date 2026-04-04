import sys
import os
import time
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import tensorflow as tf

# Emotion labels (adjust if your model uses different order)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Emotion Detection Desktop Application")
        self.setMinimumSize(1000, 600)

        # Load face detector (Haar Cascade)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "..", "assets", "haarcascade_frontalface_default.xml"),
            os.path.join(script_dir, "assets", "haarcascade_frontalface_default.xml"),
            os.path.join("assets", "haarcascade_frontalface_default.xml"),
        ]
        cascade_path = None
        for path in possible_paths:
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                cascade_path = normalized
                break
        if cascade_path is None:
            print("ERROR: Face cascade file not found.")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("ERROR: Failed to load cascade classifier.")
                self.face_cascade = None

        # Load emotion model
        model_path = os.path.join("models_store", "emotion_cnn.h5")
        if not os.path.exists(model_path):
            # Try .keras extension
            model_path = os.path.join("models_store", "emotion_cnn.keras")
        if os.path.exists(model_path):
            print(f"Loading emotion model from {model_path}")
            self.emotion_model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        else:
            print(f"ERROR: Model not found at {model_path}")
            self.emotion_model = None

        # UI Elements
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - camera feed
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.camera_label = QLabel("Camera Feed Placeholder")
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.camera_label)

        # Right panel - controls and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Control buttons
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.exit_button = QPushButton("Exit")
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)
        right_layout.addWidget(self.exit_button)

        # Current detection results
        right_layout.addWidget(QLabel("--- Current Detection ---"))
        self.emotion_label = QLabel("Emotion: --")
        self.confidence_label = QLabel("Confidence: --")
        self.fps_label = QLabel("FPS: --")
        right_layout.addWidget(self.emotion_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addWidget(self.fps_label)

        right_layout.addStretch()

        # Add panels
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        # Connect buttons
        self.exit_button.clicked.connect(self.close)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        # Camera and timing variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.fps_timer = time.time()

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.camera_label.setText("Error: Cannot open camera")
                self.capture = None
                return
        self.timer.start(30)
        self.frame_count = 0
        self.fps_timer = time.time()

    def stop_camera(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        self.camera_label.setText("Camera Feed Placeholder")
        self.emotion_label.setText("Emotion: --")
        self.confidence_label.setText("Confidence: --")
        self.fps_label.setText("FPS: --")

    def update_frame(self):
        if self.capture is None:
            return
        ret, frame = self.capture.read()
        if not ret:
            return

        # Mirror frame
        frame = cv2.flip(frame, 1)

        # Face detection
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(48, 48)
            )

            for x, y, w, h in faces:
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Emotion detection on the face ROI
                if self.emotion_model is not None:
                    # Crop face, convert to grayscale (already gray from above, but we use original color frame then convert)
                    # Actually we already have gray image, but we need to crop from that
                    face_roi_gray = gray[y : y + h, x : x + w]
                    # Resize to 48x48
                    face_resized = cv2.resize(face_roi_gray, (48, 48))
                    # Normalize and reshape
                    face_normalized = face_resized / 255.0
                    face_input = np.reshape(face_normalized, (1, 48, 48, 1))
                    # Predict
                    predictions = self.emotion_model.predict(face_input, verbose=0)
                    emotion_idx = np.argmax(predictions[0])
                    confidence = predictions[0][emotion_idx]
                    emotion_text = EMOTION_LABELS[emotion_idx]

                    # Display on frame overlay
                    label = f"{emotion_text} ({confidence:.2f})"
                    cv2.putText(
                        frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                    # Update UI labels
                    self.emotion_label.setText(f"Emotion: {emotion_text}")
                    self.confidence_label.setText(f"Confidence: {confidence:.2f}")

        # Calculate FPS
        self.frame_count += 1
        if time.time() - self.fps_timer >= 1.0:
            fps = self.frame_count / (time.time() - self.fps_timer)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.fps_timer = time.time()

        # Convert frame to Qt format and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
