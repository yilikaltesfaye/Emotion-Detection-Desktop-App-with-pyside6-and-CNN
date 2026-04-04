import sys
import os
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Emotion Detection Desktop Application")
        self.setMinimumSize(1000, 600)

        # Load face detector - robust path handling
        # Get script directory (where main.py is located)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try multiple possible paths
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
            print("ERROR: Face cascade file not found in any expected location.")
            print("Expected locations:")
            for p in possible_paths:
                print(f"  - {os.path.normpath(p)}")
            self.face_cascade = None
        else:
            print(f"Loading face cascade from: {cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("ERROR: Failed to load cascade classifier.")
                self.face_cascade = None

        # Central widget and main layout
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

        # Right panel - controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.exit_button = QPushButton("Exit")
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)
        right_layout.addWidget(self.exit_button)
        right_layout.addStretch()

        # Add panels to main layout
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        # Connect buttons
        self.exit_button.clicked.connect(self.close)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        # Camera variables
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.camera_label.setText("Error: Cannot open camera")
                self.capture = None
                return
        self.timer.start(30)  # ~33 FPS

    def stop_camera(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        self.camera_label.setText("Camera Feed Placeholder")

    def update_frame(self):
        if self.capture is None:
            return
        ret, frame = self.capture.read()
        if not ret:
            return

        # Mirror the frame (so it looks like a mirror)
        frame = cv2.flip(frame, 1)

        # Face detection
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Loosened parameters for better detection
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # more sensitive than 1.1
                minNeighbors=3,  # lower = more detections
                minSize=(30, 30),
            )
            # Debug: print number of faces
            # if len(faces) > 0:
            #     print(f"Faces detected: {len(faces)}")

            # Draw rectangle around each face (green, thicker line)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Convert BGR (OpenCV) to RGB (Qt)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit label while keeping aspect ratio
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
