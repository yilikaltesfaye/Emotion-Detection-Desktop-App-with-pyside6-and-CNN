from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2


def frame_to_pixmap(frame):
    """Convert OpenCV BGR frame to QPixmap for Qt display."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qt_image)
    return pixmap


def scale_pixmap(pixmap, target_size):
    """Scale pixmap to fit target size while keeping aspect ratio."""
    return pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
