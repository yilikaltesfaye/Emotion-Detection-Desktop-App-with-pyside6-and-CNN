import cv2


class Webcam:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.capture = None
        self.is_running = False

    def start(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.camera_id)
            if not self.capture.isOpened():
                self.capture = None
                return False
        self.is_running = True
        return True

    def stop(self):
        self.is_running = False
        if self.capture:
            self.capture.release()
            self.capture = None

    def read_frame(self):
        if self.capture is None or not self.is_running:
            return None
        ret, frame = self.capture.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # Mirror for natural self-view

    def is_opened(self):
        return self.capture is not None and self.capture.isOpened()
