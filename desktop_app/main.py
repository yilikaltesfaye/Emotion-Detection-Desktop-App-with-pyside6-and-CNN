import sys
import os

# Add desktop_app to Python path so modules can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    # Find face cascade path (from project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cascade_path = os.path.join(base_dir, "assets", "haarcascade_frontalface_default.xml")

    if not os.path.exists(cascade_path):
        print(f"ERROR: Face cascade file not found at {cascade_path}")
        sys.exit(1)

    # Find model path
    model_path = os.path.join(base_dir, "models_store", "emotion_cnn.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models_store", "emotion_model.h5")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow(cascade_path, model_path)
    window.show()
    sys.exit(app.exec())
