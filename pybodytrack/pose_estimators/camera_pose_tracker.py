import cv2
import importlib
import numpy as np
import pandas as pd
import time

class CameraPoseTracker:
    """Tracks human pose using different models and stores data in a DataFrame."""

    STANDARD_LANDMARKS = [
        "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ]

    def __init__(self, processor_name="mediapipe_processor", camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.processor = self.load_processor(processor_name)
        self.data = []

    def load_processor(self, processor_name):
        """Dynamically loads the selected pose processor."""
        module = importlib.import_module(f"pybodytrack.pose_estimators.{processor_name}")
        class_name = processor_name.split('_')[0].capitalize() + "processor"
        return getattr(module, class_name)()

    def process_frame(self, frame):
        frame_data, processed_frame = self.processor.process(frame)
        frame_data["timestamp"] = time.time()

        for lm in self.STANDARD_LANDMARKS:
            if lm not in frame_data:
                frame_data[lm] = (np.nan, np.nan, np.nan, np.nan)

        self.data.append(frame_data)
        return processed_frame

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow("Pose Tracking", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def get_dataframe(self):
        """
        Returns a DataFrame with properly formatted column names.

        Returns:
            pandas.DataFrame: DataFrame with landmark positions and timestamp.
        """
        if not self.data:
            return pd.DataFrame()

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(self.data)

        # 📌 Separar los valores X, Y, Z y Confianza en columnas distintas
        columns = ["timestamp"]  # Iniciamos con la columna de tiempo

        for landmark in self.STANDARD_LANDMARKS:
            df[[f"{landmark}_X", f"{landmark}_Y", f"{landmark}_Z", f"{landmark}_CONFIDENCE"]] = pd.DataFrame(
                df[landmark].tolist(), index=df.index
            )
            columns.extend([f"{landmark}_X", f"{landmark}_Y", f"{landmark}_Z", f"{landmark}_CONFIDENCE"])

        # 📌 Eliminar las columnas que aún contienen las tuplas originales
        df = df.drop(columns=self.STANDARD_LANDMARKS, errors="ignore")

        # 📌 Reordenar columnas para mejor legibilidad
        df = df[columns]

        return df