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
    QFileDialog,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import tensorflow as tf

# Correct label order from your notebook
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


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
        model_path = os.path.join("models_store", "emotion_cnn.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join("models_store", "emotion_model.h5")
        if os.path.exists(model_path):
            print(f"Loading emotion model from {model_path}")
            self.emotion_model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        else:
            print(f"ERROR: Model not found at {model_path}")
            self.emotion_model = None

        # UI setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - display area
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.display_label = QLabel("Camera Feed Placeholder")
        self.display_label.setStyleSheet("background-color: black; color: white;")
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.display_label)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Buttons
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.upload_button = QPushButton("Upload Image")
        self.exit_button = QPushButton("Exit")
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)
        right_layout.addWidget(self.upload_button)
        right_layout.addWidget(self.exit_button)

        # Results
        right_layout.addWidget(QLabel("--- Detection Results ---"))
        self.emotion_label = QLabel("Emotion: --")
        self.confidence_label = QLabel("Confidence: --")
        self.fps_label = QLabel("FPS: --")
        right_layout.addWidget(self.emotion_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addWidget(self.fps_label)
        right_layout.addStretch()

        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        # Connect buttons
        self.exit_button.clicked.connect(self.close)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.upload_button.clicked.connect(self.upload_image)

        # Camera and state variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.fps_timer = time.time()
        self.is_image_mode = False
        self.static_image = None
        self.frame_skip_counter = 0  # for skipping emotion inference on some frames

    def start_camera(self):
        self.is_image_mode = False
        self.static_image = None
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.display_label.setText("Error: Cannot open camera")
                self.capture = None
                return
        self.timer.start(30)  # ~33 FPS
        self.frame_count = 0
        self.fps_timer = time.time()
        self.frame_skip_counter = 0

    def stop_camera(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        self.is_image_mode = False
        self.static_image = None
        self.display_label.setText("Camera Feed Placeholder")
        self.emotion_label.setText("Emotion: --")
        self.confidence_label.setText("Confidence: --")
        self.fps_label.setText("FPS: --")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            print(f"Failed to load image: {file_path}")
            return
        if self.capture is not None:
            self.timer.stop()
            self.capture.release()
            self.capture = None
        self.is_image_mode = True
        self.static_image = img
        self.process_static_image()

    def process_static_image(self):
        if self.static_image is None:
            return
        frame = self.static_image.copy()
        self._detect_and_annotate(frame, is_static=True)
        self.display_static_frame(frame)
        self.fps_label.setText("FPS: --")

    def _detect_and_annotate(self, frame, is_static=False):
        """Shared face detection + emotion logic for both video and image."""
        if self.face_cascade is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Stricter parameters to reduce false positives
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # less sensitive to noise
            minNeighbors=6,  # higher = fewer false positives
            minSize=(80, 80),  # ignore very small regions
        )
        # Filter by aspect ratio (face should be roughly square)
        valid_faces = []
        for x, y, w, h in faces:
            aspect = w / h
            if 0.8 <= aspect <= 1.2:
                valid_faces.append((x, y, w, h))
        faces = valid_faces

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if self.emotion_model is not None:
                # Crop and preprocess face
                face_roi_gray = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi_gray, (48, 48))
                face_normalized = face_resized / 255.0
                face_input = np.reshape(face_normalized, (1, 48, 48, 1))
                # Predict
                predictions = self.emotion_model.predict(face_input, verbose=0)
                emotion_idx = np.argmax(predictions[0])
                confidence = predictions[0][emotion_idx]
                emotion_text = EMOTION_LABELS[emotion_idx]
                label = f"{emotion_text} ({confidence:.2f})"
                cv2.putText(
                    frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                # Update UI labels (only for the first face detected)
                self.emotion_label.setText(f"Emotion: {emotion_text}")
                self.confidence_label.setText(f"Confidence: {confidence:.2f}")
        return frame

    def display_static_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled_pixmap)

    def update_frame(self):
        if self.capture is None or self.is_image_mode:
            return
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # Run face detection and emotion every frame, but you can skip inference on some frames to improve FPS
        # Here we run inference every frame, but you can change to every 2nd frame if needed
        self._detect_and_annotate(frame, is_static=False)

        # FPS calculation
        self.frame_count += 1
        if time.time() - self.fps_timer >= 1.0:
            fps = self.frame_count / (time.time() - self.fps_timer)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.fps_timer = time.time()

        self.display_static_frame(frame)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
