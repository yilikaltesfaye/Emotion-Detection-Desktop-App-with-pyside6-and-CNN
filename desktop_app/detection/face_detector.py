import cv2
import numpy as np


class FaceDetector:
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
        self.face_history = []
        self.no_face_counter = 0

    def detect_and_annotate(self, frame):
        """Detect faces and return annotated frame with bounding boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        faces = self.face_cascade.detectMultiScale(
            gray_blurred,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(40, 40),
            maxSize=(400, 400),
        )

        # Filter by aspect ratio
        valid_faces = []
        for x, y, w, h in faces:
            aspect = w / h
            if 0.7 <= aspect <= 1.3 and w > 40 and h > 40:
                valid_faces.append((x, y, w, h))

        # Temporal smoothing
        if valid_faces:
            largest = max(valid_faces, key=lambda f: f[2] * f[3])
            self.no_face_counter = 0
            self.face_history.append(largest)
            if len(self.face_history) > 5:
                self.face_history.pop(0)
        else:
            self.no_face_counter += 1

        # Determine if we should draw a face box
        if self.face_history and self.no_face_counter < 10:
            avg_x = int(np.mean([f[0] for f in self.face_history]))
            avg_y = int(np.mean([f[1] for f in self.face_history]))
            avg_w = int(np.mean([f[2] for f in self.face_history]))
            avg_h = int(np.mean([f[3] for f in self.face_history]))
            faces_to_draw = [(avg_x, avg_y, avg_w, avg_h)]
            has_face = True
        else:
            faces_to_draw = []
            has_face = False

        # Draw rectangles
        for x, y, w, h in faces_to_draw:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame, faces_to_draw, gray, has_face

    def reset(self):
        self.face_history = []
        self.no_face_counter = 0
