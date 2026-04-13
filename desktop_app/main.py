import sys
import os
import time
import csv
from datetime import datetime
from collections import deque
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

        # Buttons - Row 1
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.upload_button = QPushButton("Upload Image")
        self.exit_button = QPushButton("Exit")
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)
        right_layout.addWidget(self.upload_button)
        right_layout.addWidget(self.exit_button)

        # Logging buttons
        right_layout.addWidget(QLabel("--- CSV Logging ---"))
        self.start_log_button = QPushButton("Start Logging")
        self.stop_log_button = QPushButton("Stop Logging")
        right_layout.addWidget(self.start_log_button)
        right_layout.addWidget(self.stop_log_button)

        # Session controls
        right_layout.addWidget(QLabel("--- Session ---"))
        self.reset_button = QPushButton("Reset Session")
        right_layout.addWidget(self.reset_button)

        # Results
        right_layout.addWidget(QLabel("--- Detection Results ---"))
        self.emotion_label = QLabel("Emotion: --")
        self.confidence_label = QLabel("Confidence: --")
        right_layout.addWidget(self.emotion_label)
        right_layout.addWidget(self.confidence_label)

        # Session stats
        right_layout.addWidget(QLabel("--- Session Stats ---"))
        self.session_timer_label = QLabel("Session Time: 0s")
        self.dominant_emotion_label = QLabel("Dominant Emotion: --")
        self.fps_label = QLabel("FPS: --")
        right_layout.addWidget(self.session_timer_label)
        right_layout.addWidget(self.dominant_emotion_label)
        right_layout.addWidget(self.fps_label)

        right_layout.addStretch()

        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        # Connect buttons
        self.exit_button.clicked.connect(self.close)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.upload_button.clicked.connect(self.upload_image)
        self.start_log_button.clicked.connect(self.start_logging)
        self.stop_log_button.clicked.connect(self.stop_logging)
        self.reset_button.clicked.connect(self.reset_session)

        # Camera and state variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.fps_timer = time.time()
        self.is_image_mode = False
        self.static_image = None

        # Face detection smoothing
        self.face_history = []
        self.no_face_counter = 0

        # Emotion smoothing
        self.emotion_history = deque(maxlen=5)
        self.inference_counter = 0

        # Session variables
        self.session_start_time = None
        self.session_active = False
        self.emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS}
        self.total_predictions = 0

        # Logging variables
        self.logging_active = False
        self.log_file = None
        self.log_writer = None

        # Current emotion for logging
        self.current_emotion = "None"
        self.current_confidence = 0.0

    def start_camera(self):
        self.is_image_mode = False
        self.static_image = None
        self.face_history = []
        self.no_face_counter = 0
        self.emotion_history.clear()

        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.display_label.setText("Error: Cannot open camera")
                self.capture = None
                return

        # Start session if not already active
        if not self.session_active:
            self.start_session()

        self.timer.start(30)
        self.frame_count = 0
        self.fps_timer = time.time()
        self.inference_counter = 0

    def stop_camera(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        self.is_image_mode = False
        self.static_image = None
        self.display_label.setText("Camera Feed Placeholder")
        # Stop session when camera stops
        if self.session_active:
            self.end_session()

    def start_session(self):
        self.session_start_time = time.time()
        self.session_active = True
        self.emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS}
        self.total_predictions = 0
        self.update_session_display()

    def end_session(self):
        self.session_active = False
        self.session_start_time = None
        self.update_session_display()

    def reset_session(self):
        # Stop logging if active
        if self.logging_active:
            self.stop_logging()
        # Reset all session data
        self.emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS}
        self.total_predictions = 0
        self.emotion_history.clear()
        self.current_emotion = "None"
        self.current_confidence = 0.0
        self.emotion_label.setText("Emotion: --")
        self.confidence_label.setText("Confidence: --")
        self.dominant_emotion_label.setText("Dominant Emotion: --")
        # Restart session timer if camera is active
        if self.capture is not None and self.capture.isOpened():
            self.session_start_time = time.time()
            self.session_active = True
        else:
            self.session_active = False
            self.session_start_time = None
        self.update_session_display()

    def start_logging(self):
        if self.logging_active:
            return
        # Create outputs folder if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"outputs/emotion_log_{timestamp}.csv"
        self.log_file = open(log_filename, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        # Write header
        self.log_writer.writerow(["timestamp", "emotion", "confidence", "session_time_seconds"])
        self.logging_active = True
        print(f"Logging started: {log_filename}")

    def stop_logging(self):
        if not self.logging_active:
            return
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.log_writer = None
        self.logging_active = False
        print("Logging stopped")

    def log_prediction(self, emotion, confidence):
        if not self.logging_active:
            return
        if self.log_writer and self.session_active:
            session_time = int(time.time() - self.session_start_time)
            self.log_writer.writerow(
                [datetime.now().isoformat(), emotion, f"{confidence:.4f}", session_time]
            )
            self.log_file.flush()  # Ensure data is written

    def update_session_display(self):
        if self.session_active and self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            self.session_timer_label.setText(f"Session Time: {elapsed}s")
        else:
            self.session_timer_label.setText("Session Time: 0s")

        # Update dominant emotion
        if self.total_predictions > 0:
            dominant = max(self.emotion_counts, key=self.emotion_counts.get)
            self.dominant_emotion_label.setText(f"Dominant Emotion: {dominant}")
        else:
            self.dominant_emotion_label.setText("Dominant Emotion: --")

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
        annotated_frame = self._detect_and_annotate(frame, is_static=True)
        self.display_static_frame(annotated_frame)
        self.fps_label.setText("FPS: --")

    def _detect_and_annotate(self, frame, is_static=False):
        if self.face_cascade is None:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        faces = self.face_cascade.detectMultiScale(
            gray_blurred,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(40, 40),
            maxSize=(400, 400),
        )

        valid_faces = []
        for x, y, w, h in faces:
            aspect = w / h
            if 0.7 <= aspect <= 1.3 and w > 40 and h > 40:
                valid_faces.append((x, y, w, h))

        current_face = None
        if valid_faces:
            current_face = max(valid_faces, key=lambda f: f[2] * f[3])
            self.no_face_counter = 0
            self.face_history.append(current_face)
            if len(self.face_history) > 5:
                self.face_history.pop(0)
        else:
            self.no_face_counter += 1

        show_face = False
        if self.face_history and self.no_face_counter < 10:
            avg_x = int(np.mean([f[0] for f in self.face_history]))
            avg_y = int(np.mean([f[1] for f in self.face_history]))
            avg_w = int(np.mean([f[2] for f in self.face_history]))
            avg_h = int(np.mean([f[3] for f in self.face_history]))
            faces_to_draw = [(avg_x, avg_y, avg_w, avg_h)]
            show_face = True
        else:
            faces_to_draw = []

        self.inference_counter += 1
        run_inference = (self.inference_counter % 2 == 0) or is_static

        for x, y, w, h in faces_to_draw:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self.emotion_model is not None and run_inference and show_face:
                face_roi_gray = gray[y : y + h, x : x + w]
                face_resized = cv2.resize(face_roi_gray, (48, 48))
                face_normalized = face_resized / 255.0
                face_input = np.reshape(face_normalized, (1, 48, 48, 1))

                predictions = self.emotion_model.predict(face_input, verbose=0)
                emotion_idx = np.argmax(predictions[0])
                confidence = predictions[0][emotion_idx]
                emotion_text = EMOTION_LABELS[emotion_idx]

                self.emotion_history.append(emotion_idx)

                if len(self.emotion_history) > 0:
                    smoothed_idx = max(set(self.emotion_history), key=self.emotion_history.count)
                    smoothed_emotion = EMOTION_LABELS[smoothed_idx]
                    smoothed_confidence = np.mean(
                        [predictions[0][idx] for idx in self.emotion_history]
                    )
                else:
                    smoothed_emotion = emotion_text
                    smoothed_confidence = confidence

                # Store for logging and session stats
                self.current_emotion = smoothed_emotion
                self.current_confidence = smoothed_confidence

                # Update session statistics
                if self.session_active:
                    self.emotion_counts[smoothed_emotion] += 1
                    self.total_predictions += 1
                    self.update_session_display()

                    # Log to CSV if logging is active
                    self.log_prediction(smoothed_emotion, smoothed_confidence)

                # Display emotion label on frame
                cv2.putText(
                    frame,
                    smoothed_emotion,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Update UI labels
                self.emotion_label.setText(f"Emotion: {smoothed_emotion}")
                self.confidence_label.setText(f"Confidence: {smoothed_confidence:.2f}")

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

        # Update session timer display
        if self.session_active and self.session_start_time:
            elapsed = int(time.time() - self.session_start_time)
            self.session_timer_label.setText(f"Session Time: {elapsed}s")

        annotated_frame = self._detect_and_annotate(frame, is_static=False)

        self.frame_count += 1
        if time.time() - self.fps_timer >= 1.0:
            fps = self.frame_count / (time.time() - self.fps_timer)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.fps_timer = time.time()

        self.display_static_frame(annotated_frame)

    def closeEvent(self, event):
        self.stop_logging()
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
