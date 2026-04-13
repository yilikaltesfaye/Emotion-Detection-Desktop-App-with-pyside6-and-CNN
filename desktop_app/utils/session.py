import time
from collections import deque


class SessionManager:
    def __init__(self, emotion_labels):
        self.emotion_labels = emotion_labels
        self.session_active = False
        self.session_start_time = None
        self.emotion_counts = {emotion: 0 for emotion in emotion_labels}
        self.total_predictions = 0
        self.emotion_history = deque(maxlen=5)  # For smoothing
        self.current_emotion = "None"
        self.current_confidence = 0.0

    def start(self):
        self.session_start_time = time.time()
        self.session_active = True
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        self.total_predictions = 0
        self.emotion_history.clear()

    def stop(self):
        self.session_active = False
        self.session_start_time = None

    def reset(self):
        self.stop()
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        self.total_predictions = 0
        self.emotion_history.clear()
        self.current_emotion = "None"
        self.current_confidence = 0.0

    def update(self, emotion_idx, confidence):
        """Update session with a new prediction."""
        emotion_text = self.emotion_labels[emotion_idx]
        self.emotion_history.append(emotion_idx)

        # Get smoothed emotion (majority vote)
        if len(self.emotion_history) > 0:
            smoothed_idx = max(set(self.emotion_history), key=self.emotion_history.count)
            smoothed_emotion = self.emotion_labels[smoothed_idx]
            smoothed_confidence = confidence
        else:
            smoothed_emotion = emotion_text
            smoothed_confidence = confidence

        self.current_emotion = smoothed_emotion
        self.current_confidence = smoothed_confidence

        if self.session_active:
            self.emotion_counts[smoothed_emotion] += 1
            self.total_predictions += 1

        return smoothed_emotion, smoothed_confidence

    def get_elapsed_time(self):
        if self.session_active and self.session_start_time:
            return int(time.time() - self.session_start_time)
        return 0

    def get_dominant_emotion(self):
        if self.total_predictions > 0:
            return max(self.emotion_counts, key=self.emotion_counts.get)
        return "--"
