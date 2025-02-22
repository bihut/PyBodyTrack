import cv2
import torch
import numpy as np
from ultralytics import YOLO

class YoloProcessor:
    """Pose detector using YOLOv8-Pose from Ultralytics."""

    def __init__(self, model_path=None, device='cpu'):
        """
        Inicializa el procesador de YOLO.

        Args:
            model_path (str, opcional): Ruta al modelo YOLO personalizado. Si es None, usa 'yolov8n-pose.pt'.
            device (str, opcional): Dispositivo a usar ('cuda' o 'cpu'). Si es None, se detecta automáticamente.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path if model_path else "yolov8n-pose.pt"

        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # 📌 Landmarks detectados por YOLOv8 (solo 17)
        self.STANDARD_LANDMARKS = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        # 📌 Conexiones del esqueleto para dibujar
        self.SKELETON_CONNECTIONS = [
            (0, 1), (1, 3), (0, 2), (2, 4),  # Cabeza
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Brazos
            (5, 11), (6, 12), (11, 12),  # Tronco
            (11, 13), (13, 15), (12, 14), (14, 16)  # Piernas
        ]

    def get_standard_landmarks(self):
        """Devuelve los landmarks estándar de YOLO"""
        return self.STANDARD_LANDMARKS

    def process(self, frame):
        results = self.model(frame)
        frame_height, frame_width, _ = frame.shape
        data = {key: (np.nan, np.nan, 0, np.nan) for key in self.STANDARD_LANDMARKS}  # 📌 Z = 0 por defecto
        keypoint_positions = {}

        for result in results:
            if result.keypoints is None:
                return data, frame

            keypoints = result.keypoints.xyn.cpu().numpy()

            for person in keypoints:
                for idx, kp in enumerate(person):
                    if idx < len(self.STANDARD_LANDMARKS):
                        kp = np.array(kp).flatten()
                        x, y = float(kp[0]), float(kp[1])
                        confidence = float(kp[2]) if len(kp) > 2 else np.nan

                        abs_x, abs_y = int(x * frame_width), int(y * frame_height)
                        data[self.STANDARD_LANDMARKS[idx]] = (abs_x, abs_y, 0, confidence)  # 📌 Z is always 0
                        keypoint_positions[idx] = (abs_x, abs_y)

                        # 📌 Dibujar keypoints
                        if confidence > 0.2:
                            cv2.circle(frame, (abs_x, abs_y), 5, (0, 255, 0), -1)

                # 📌 Dibujar conexiones del esqueleto
                for (p1, p2) in self.SKELETON_CONNECTIONS:
                    if p1 in keypoint_positions and p2 in keypoint_positions:
                        x1, y1 = keypoint_positions[p1]
                        x2, y2 = keypoint_positions[p2]

                        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return data, frame
