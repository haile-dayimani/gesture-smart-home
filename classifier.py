"""
Loads the trained model and classifies one hand picture.

Uses the small TensorFlow Lite model when it can (fast on the Raspberry Pi)
and falls back to the full Keras model otherwise.
"""

import time

import cv2
import numpy as np

import config


def _load_tflite_interpreter(path):
    """Try the light TFLite runtimes first, then full TensorFlow."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf  # full TensorFlow
            Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


class GestureClassifier:
    def __init__(self, prefer_tflite=True):
        if not config.LABELS_PATH.exists():
            raise FileNotFoundError(
                f"{config.LABELS_PATH} not found. Run train.py first.")
        self.labels = config.LABELS_PATH.read_text().split()

        self.interpreter = None
        self.keras_model = None
        if prefer_tflite and config.TFLITE_MODEL_PATH.exists():
            self.interpreter = _load_tflite_interpreter(config.TFLITE_MODEL_PATH)
            details = self.interpreter.get_input_details()[0]
            self.input_index = details["index"]
            self.img_size = int(details["shape"][1])  # 128 (own CNN) or 160
            self.output_index = self.interpreter.get_output_details()[0]["index"]
            self.backend = "TFLite"
        elif config.KERAS_MODEL_PATH.exists():
            from tensorflow import keras
            self.keras_model = keras.models.load_model(config.KERAS_MODEL_PATH)
            self.img_size = int(self.keras_model.input_shape[1])
            self.backend = "Keras"
        else:
            raise FileNotFoundError("No trained model found. Run train.py first.")

    def predict(self, roi_bgr):
        """
        roi_bgr: the hand box from camera.crop_roi() (BGR, SAVE_SIZE square).
        Returns (label, confidence, all_probabilities, milliseconds).
        """
        # Same steps as training: RGB, resized to the model's input size,
        # 0-255 floats (the model rescales the pixels itself).
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        batch = np.expand_dims(rgb.astype(np.float32), axis=0)

        start = time.perf_counter()
        if self.interpreter is not None:
            self.interpreter.set_tensor(self.input_index, batch)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.output_index)[0]
        else:
            probs = self.keras_model(batch, training=False).numpy()[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        best = int(np.argmax(probs))
        return self.labels[best], float(probs[best]), probs, elapsed_ms
