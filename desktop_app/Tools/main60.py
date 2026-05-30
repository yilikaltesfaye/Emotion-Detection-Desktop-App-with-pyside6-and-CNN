import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer
from tensorflow.keras.models import load_model
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QAction

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
#  REAL SpatialAttentionAnchor FROM YOUR TRAINING CODE
# ============================================================
class SpatialAttentionAnchor(Layer):
    """
    Your actual custom layer from training.
    Applies spatial attention by combining avg-pool and max-pool
    along the channel axis, then convolving with sigmoid activation.
    """

    def __init__(self, **kwargs):
        super(SpatialAttentionAnchor, self).__init__(**kwargs)
        self.conv = Conv2D(1, (3, 3), padding="same", activation="sigmoid")

    def call(self, inputs):
        avg_p = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_p = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.keras.layers.concatenate([avg_p, max_p], axis=-1)
        attn = self.conv(concat)
        return tf.keras.layers.multiply([inputs, attn])

    def get_config(self):
        config = super(SpatialAttentionAnchor, self).get_config()
        return config

    def build(self, input_shape):
        """Proper build method to suppress the warning."""
        super(SpatialAttentionAnchor, self).build(input_shape)


# ============================================================
#  USER SETTINGS
# ============================================================
DEBUG = True  # Set False after testing
APPLY_SOFTMAX = False  # Model already outputs softmax probabilities
# ============================================================


class ProcessingWorker(QThread):
    frame_ready = Signal(np.ndarray)
    results_ready = Signal(str, float)

    def __init__(self, keras_model_path, yunet_model_path="models_store/yunet.onnx"):
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

        # ---------- Load your trained student model ----------
        custom_objects = {"SpatialAttentionAnchor": SpatialAttentionAnchor}
        self.model = load_model(keras_model_path, custom_objects=custom_objects)
        self.emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.frame_count = 0

    def preprocess(self, face_roi):
        """
        Preprocess face crop to EXACTLY match training:
        - Grayscale (1 channel)
        - 48x48 pixels
        - Scale to [0, 1]
        """
        # 1. Convert BGR to grayscale (matches training color_mode='grayscale')
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # 2. Resize to 48x48 (NOT 128x128!)
        resized = cv2.resize(gray, (48, 48))

        # 3. Scale to [0, 1] (matches rescale=1./255)
        img = resized.astype(np.float32) / 255.0

        # 4. Add channel dimension: (48, 48) -> (48, 48, 1)
        img = np.expand_dims(img, axis=-1)

        # 5. Add batch dimension: (48, 48, 1) -> (1, 48, 48, 1)
        return np.expand_dims(img, axis=0)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_w}x{actual_h}")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            emotion, conf = "neutral", 0.0

            if faces is not None:
                for face in faces:
                    x, y, fw, fh = face[0:4].astype(int)
                    x, y = max(0, x), max(0, y)
                    fw, fh = min(fw, w - x), min(fh, h - y)

                    # Expand crop by 30% to include forehead/chin
                    expand_x = int(fw * 0.3)
                    expand_y = int(fh * 0.3)
                    x = max(0, x - expand_x)
                    y = max(0, y - expand_y)
                    fw = min(w - x, fw + 2 * expand_x)
                    fh = min(h - y, fh + 2 * expand_y)

                    face_roi = frame[y : y + fh, x : x + fw]

                    if face_roi.size > 0:
                        # Preprocess to match training (48x48 grayscale, [0,1])
                        input_tensor = self.preprocess(face_roi)

                        # Run inference
                        outputs = self.model.predict(input_tensor, verbose=0)

                        # Model already outputs softmax probabilities (0-1, sum=1)
                        probs = outputs[0]

                        # Make sure we have valid probabilities
                        # Sometimes the model outputs slightly off due to averaging
                        probs = np.clip(probs, 0, None)  # Clip negative values
                        probs = probs / probs.sum()  # Re-normalize

                        idx = np.argmax(probs)
                        emotion, conf = self.emotions[idx], float(probs[idx])

                        if DEBUG and self.frame_count % 30 == 0:
                            print(f"Probs:  {probs}")
                            print(f"Sum: {probs.sum():.4f}")
                            print(f"Prediction: {emotion} ({conf:.4f})")

                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            self.frame_count += 1
            self.frame_ready.emit(frame)
            self.results_ready.emit(emotion, conf)

        cap.release()

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self, keras_model_path):
        super().__init__()
        self.setWindowTitle("Emotion Monitor")
        self.resize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.display = QLabel("Loading...")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setMinimumSize(640, 480)
        self.display.setStyleSheet("background-color: #222; border: 1px solid #555;")
        main_layout.addWidget(self.display, 1)

        bottom_panel = QHBoxLayout()
        self.emo_label = QLabel("Emotion: --\nConfidence: --")
        self.emo_label.setFont(QFont("Segoe UI", 16))
        self.emo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bottom_panel.addWidget(self.emo_label, 1)

        quit_btn = QPushButton("Exit")
        quit_btn.setFont(QFont("Segoe UI", 12))
        quit_btn.setFixedWidth(100)
        quit_btn.clicked.connect(self.close)
        bottom_panel.addWidget(quit_btn)
        main_layout.addLayout(bottom_panel)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Esc")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("&View")
        fullscreen_action = QAction("Toggle &Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        self.worker = ProcessingWorker(keras_model_path)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.results_ready.connect(self.update_text)
        self.worker.start()

    @Slot(np.ndarray)
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display.setPixmap(pixmap)

    @Slot(str, float)
    def update_text(self, emotion, confidence):
        self.emo_label.setText(f"Emotion: {emotion}\nConfidence: {confidence:.4f}")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event):
        if hasattr(self, "worker"):
            self.worker.stop()
            self.worker.quit()
            self.worker.wait(5000)
        super().closeEvent(event)


if __name__ == "__main__":
    MODEL_PATH = "models_store/CNN_60.keras"  # or .h5
    app = QApplication(sys.argv)
    window = MainWindow(MODEL_PATH)
    window.show()
    sys.exit(app.exec())
