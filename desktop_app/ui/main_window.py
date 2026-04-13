import time
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
)
from PySide6.QtCore import QTimer, Qt
import cv2

from camera.webcam import Webcam
from detection.face_detector import FaceDetector
from models.emotion_model import EmotionModel
from utils.logger import CSVLogger
from utils.session import SessionManager
from utils.display import frame_to_pixmap, scale_pixmap

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


class MainWindow(QMainWindow):
    def __init__(self, face_cascade_path, model_path):
        super().__init__()
        self.setWindowTitle("Real-Time Emotion Detection Desktop Application")
        self.setMinimumSize(1000, 600)

        # Initialize components
        self.webcam = Webcam()
        self.face_detector = FaceDetector(face_cascade_path)
        self.emotion_model = EmotionModel(model_path, EMOTION_LABELS)
        self.logger = CSVLogger()
        self.session = SessionManager(EMOTION_LABELS)

        # UI state
        self.is_image_mode = False
        self.static_image = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # FPS tracking
        self.frame_count = 0
        self.fps_timer = time.time()

        self.setup_ui()

    def setup_ui(self):
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
        self.start_log_button.clicked.connect(self.logger.start)
        self.stop_log_button.clicked.connect(self.logger.stop)
        self.reset_button.clicked.connect(self.reset_session)

    def start_camera(self):
        self.is_image_mode = False
        self.static_image = None
        self.face_detector.reset()
        self.emotion_model.reset_inference_counter()

        if not self.webcam.start():
            self.display_label.setText("Error: Cannot open camera")
            return

        if not self.session.session_active:
            self.session.start()

        self.timer.start(30)
        self.frame_count = 0
        self.fps_timer = time.time()

    def stop_camera(self):
        self.timer.stop()
        self.webcam.stop()
        self.is_image_mode = False
        self.static_image = None
        self.display_label.setText("Camera Feed Placeholder")
        if self.session.session_active:
            self.session.stop()
        self.update_session_display()

    def reset_session(self):
        if self.logger.is_logging:
            self.logger.stop()
        self.session.reset()
        self.emotion_label.setText("Emotion: --")
        self.confidence_label.setText("Confidence: --")
        self.dominant_emotion_label.setText("Dominant Emotion: --")
        self.update_session_display()
        if self.webcam.is_opened():
            self.session.start()

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
        if self.webcam.is_opened():
            self.timer.stop()
            self.webcam.stop()
        self.is_image_mode = True
        self.static_image = img
        self.process_static_image()

    def process_static_image(self):
        if self.static_image is None:
            return
        frame = self.static_image.copy()
        annotated_frame, faces, gray, has_face = self.face_detector.detect_and_annotate(frame)

        if has_face and faces:
            for x, y, w, h in faces:
                face_roi = gray[y : y + h, x : x + w]
                emotion_idx, confidence = self.emotion_model.predict(face_roi)
                emotion_text = EMOTION_LABELS[emotion_idx]
                cv2.putText(
                    annotated_frame,
                    emotion_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                self.emotion_label.setText(f"Emotion: {emotion_text}")
                self.confidence_label.setText(f"Confidence: {confidence:.2f}")

        pixmap = frame_to_pixmap(annotated_frame)
        scaled = scale_pixmap(pixmap, self.display_label.size())
        self.display_label.setPixmap(scaled)
        self.fps_label.setText("FPS: --")

    def update_frame(self):
        if self.webcam.is_opened() and not self.is_image_mode:
            frame = self.webcam.read_frame()
            if frame is None:
                return

            # Face detection and annotation
            annotated_frame, faces, gray, has_face = self.face_detector.detect_and_annotate(frame)

            # Run emotion inference
            run_inference = self.emotion_model.should_run_inference()

            for x, y, w, h in faces:
                if has_face and run_inference:
                    face_roi = gray[y : y + h, x : x + w]
                    emotion_idx, confidence = self.emotion_model.predict(face_roi)

                    # Update session
                    smoothed_emotion, smoothed_confidence = self.session.update(
                        emotion_idx, confidence
                    )

                    # Display emotion on frame
                    cv2.putText(
                        annotated_frame,
                        smoothed_emotion,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    # Update UI
                    self.emotion_label.setText(f"Emotion: {smoothed_emotion}")
                    self.confidence_label.setText(f"Confidence: {smoothed_confidence:.2f}")

                    # Log to CSV
                    if self.logger.is_logging:
                        self.logger.log(
                            smoothed_emotion, smoothed_confidence, self.session.get_elapsed_time()
                        )

            # Update session display
            self.update_session_display()

            # FPS calculation
            self.frame_count += 1
            if time.time() - self.fps_timer >= 1.0:
                fps = self.frame_count / (time.time() - self.fps_timer)
                self.fps_label.setText(f"FPS: {fps:.1f}")
                self.frame_count = 0
                self.fps_timer = time.time()

            # Display
            pixmap = frame_to_pixmap(annotated_frame)
            scaled = scale_pixmap(pixmap, self.display_label.size())
            self.display_label.setPixmap(scaled)

    def update_session_display(self):
        elapsed = self.session.get_elapsed_time()
        self.session_timer_label.setText(f"Session Time: {elapsed}s")
        dominant = self.session.get_dominant_emotion()
        self.dominant_emotion_label.setText(f"Dominant Emotion: {dominant}")

    def closeEvent(self, event):
        self.logger.stop()
        self.stop_camera()
        event.accept()
