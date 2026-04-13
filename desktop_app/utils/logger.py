import os
import csv
from datetime import datetime


class CSVLogger:
    def __init__(self):
        self.log_file = None
        self.log_writer = None
        self.is_logging = False
        self.current_log_path = None

    def start(self):
        if self.is_logging:
            return
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"outputs/emotion_log_{timestamp}.csv"
        self.log_file = open(log_filename, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["timestamp", "emotion", "confidence", "session_time_seconds"])
        self.is_logging = True
        self.current_log_path = log_filename
        print(f"Logging started: {log_filename}")

    def stop(self):
        if not self.is_logging:
            return
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        self.log_writer = None
        self.is_logging = False
        print("Logging stopped")

    def log(self, emotion, confidence, session_time):
        if not self.is_logging:
            return
        if self.log_writer:
            self.log_writer.writerow(
                [datetime.now().isoformat(), emotion, f"{confidence:.4f}", session_time]
            )
            self.log_file.flush()
