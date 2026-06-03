import sys
import os
import cv2
import onnxruntime as ort
import numpy as np
import logging
from pathlib import Path
from collections import Counter, deque
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QAction


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Suppress noisy TensorFlow/MediaPipe logs (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ---------------- Logging ----------------

log_file = Path("emotion_detection.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)

logging.info("Application started")


class GuiLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.callback = None

    def emit(self, record):
        if self.callback:
            msg = self.format(record)
            self.callback(msg)


gui_handler = GuiLogHandler()

gui_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

logging.getLogger().addHandler(gui_handler)


class ProcessingWorker(QThread):
    frame_ready = Signal(np.ndarray)
    results_ready = Signal(str, float)

    def __init__(
        self, emotion_model_path, yunet_model_path=resource_path("models_store/yunet.onnx")
    ):
        super().__init__()
        self.running = True

        # ---------- Lightweight face detector: YuNet ----------
        self.detector = cv2.FaceDetectorYN.create(
            model=yunet_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        # ---------- Emotion classifier (ONNX) ----------
        self.session = ort.InferenceSession(emotion_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        self.emotion_buffer = deque(maxlen=30)
        self.last_logged_emotion = None

    def predict_emotion(self, face_roi):

        face_resized = cv2.resize(face_roi, (128, 128))

        img_float = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img_norm = (img_float - mean) / std

        input_tensor = img_norm.transpose(2, 0, 1).reshape(1, 3, 128, 128)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]))

        idx = np.argmax(probs)

        return (self.emotions[idx], float(np.max(probs)))

    def run(self):
        # Try DirectShow for better resolution support on Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logging.error("Could not open webcam.")
            return

        # Request HD (1280x720). If the camera doesn't support it, it will fall back.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_w}x{actual_h}")
        logging.info(f"Camera initialized at {actual_w}x{actual_h}")
        if actual_w <= 640:
            logging.warning("Camera may only support 640x480.")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            emotion, conf = "Neutral", 0.0

            if faces is not None:
                for face in faces:
                    x, y, fw, fh = face[0:4].astype(int)
                    x, y = max(0, x), max(0, y)
                    fw, fh = min(fw, w - x), min(fh, h - y)

                    face_roi = frame[y : y + fh, x : x + fw]

                    if face_roi.size > 0:
                        current_emotion, current_conf = self.predict_emotion(face_roi)
                        self.emotion_buffer.append(current_emotion)
                        emotion = Counter(self.emotion_buffer).most_common(1)[0][0]

                        conf = current_conf
                        if emotion != self.last_logged_emotion:
                            logging.info(f"Detected Emotion Changed -> {emotion} ({conf:.2%})")
                            self.last_logged_emotion = emotion

                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            self.frame_ready.emit(frame)
            self.results_ready.emit(emotion, conf)

        cap.release()

    def stop(self):
        """Stop the worker loop."""
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self, emotion_model_path):
        super().__init__()
        self.setWindowTitle("Real-Time Facial Emotion Detection App")
        self.resize(1000, 700)  # Comfortable default size

        # Central widget with main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ---- Video feed ----
        self.display = QLabel("Loading...")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setMinimumSize(640, 480)
        self.display.setStyleSheet("background-color: #222; border: 1px solid #555;")
        main_layout.addWidget(self.display, 1)  # stretch factor 1 -> takes most space

        # ---- Bottom panel (emotion info + exit button) ----
        bottom_panel = QHBoxLayout()

        self.emo_label = QLabel("Emotion: N/A\nConfidence: N/A")
        self.emo_label.setFont(QFont("Segoe UI", 16))
        self.emo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bottom_panel.addWidget(self.emo_label, 1)
        upload_btn = QPushButton("Upload Image")
        upload_btn.setFont(QFont("Segoe UI", 12))
        upload_btn.clicked.connect(self.upload_image)

        bottom_panel.addWidget(upload_btn)
        quit_btn = QPushButton("Exit")
        quit_btn.setFont(QFont("Segoe UI", 12))
        quit_btn.setFixedWidth(100)
        quit_btn.clicked.connect(self.close)
        bottom_panel.addWidget(quit_btn)

        main_layout.addLayout(bottom_panel)
        self.status_history = []
        gui_handler.callback = self.add_footer_log

        self.log_footer = QLabel("System Ready")
        self.log_footer.setFont(QFont("Consolas", 10))
        self.log_footer.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.log_footer.setStyleSheet("""
        background-color:#111;
        border:1px solid #555;
        padding:6px;
        """)

        main_layout.addWidget(self.log_footer)

        # ---- Menu bar with a small hint ----
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        upload_action = QAction("Upload Image", self)

        upload_action.triggered.connect(self.upload_image)

        file_menu.addAction(upload_action)
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Esc")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("&View")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        view_menu.addAction(about_action)
        fullscreen_action = QAction("Toggle &Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # ---- Start the worker thread ----
        self.worker = ProcessingWorker(emotion_model_path)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.results_ready.connect(self.update_text)
        self.worker.start()

    @Slot(np.ndarray)
    def update_frame(self, frame):
        """Convert OpenCV frame to QPixmap and display."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display.setPixmap(pixmap)

    @Slot(str, float)
    def update_text(self, emotion, confidence):
        self.emo_label.setText(f"Emotion: {emotion}\n" f"Confidence: {confidence:.2f}")

    def add_footer_log(self, message):
        self.status_history.append(message)
        self.status_history = self.status_history[-2:]
        self.log_footer.setText("\n".join(self.status_history))

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_about(self):
        QMessageBox.about(
            self,
            "About",
            """
Real-Time Facial Emotion Detection Desktop App

    Face Detection:
        YuNet

    Emotion Classification:
        ONNX Runtime Knowledge Distilled Model

Version: 2.1.0

Made By Computer Engineering Students 
    1. Abriham Fikre
    2. Beza Mulugeta
    3. Ebisa Getachew
    4. Yilikal Tesfaye

Made in University of  Gondar, Ethiopia, 2026
""",
        )

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:

            logging.error("Failed to load image.")

            self.add_footer_log("Failed to load image.")

            return

        h, w, _ = image.shape
        upload_detector = cv2.FaceDetectorYN.create(
            model=resolved_yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        resolved_yunet_path = resource_path("models_store/yunet.onnx")
        upload_detector.setInputSize((w, h))
        _, faces = upload_detector.detect(image)

        emotion = "No Face"
        confidence = 0.0

        if faces is not None:

            face = faces[0]

            x, y, fw, fh = face[0:4].astype(int)

            face_roi = image[y : y + fh, x : x + fw]

            if face_roi.size > 0:

                emotion, confidence = self.worker.predict_emotion(face_roi)

                cv2.rectangle(image, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        self.update_text(emotion, confidence)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            self.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.display.setPixmap(pixmap)

        msg = f"Uploaded Image -> " f"{emotion} " f"({confidence:.2%})"

        logging.info(msg)

        self.add_footer_log(msg)

    def closeEvent(self, event):
        """Gracefully shut down the worker thread."""
        if hasattr(self, "worker"):
            logging.info("Application shutting down")
            self.worker.stop()
            self.worker.quit()
            self.worker.wait(5000)
        super().closeEvent(event)


if __name__ == "__main__":
    MODEL_PATH = resource_path("models_store/V7/best_fer_student.onnx")
    app = QApplication(sys.argv)
    window = MainWindow(MODEL_PATH)
    window.show()  # Normal window, not fullscreen
    sys.exit(app.exec())
