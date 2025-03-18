"""
PyBodyTrack - A Python library for motion quantification in videos.

Author: Angel Ruiz Zafra
License: Apache 2.0 License
Version: 2025.3.2
Repository: https://github.com/bihut/PyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""

import cv2
import numpy as np
import mediapipe as mp
from pybodytrack.bodyparts import body_parts

class MediaPipeProcessor:
    """Pose detector using MediaPipe."""

    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

        # 📌 Landmarks detectados por MediaPipe (33 puntos)
        self.STANDARD_LANDMARKS = body_parts.STANDARD_LANDMARKS

        self.SKELETON_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    def get_standard_landmarks(self):
        """Devuelve los landmarks estándar de MediaPipe"""
        return self.STANDARD_LANDMARKS

    def process(self, frame, selected_landmarks=None):
        data = {key: (np.nan, np.nan, np.nan, np.nan) for key in self.STANDARD_LANDMARKS}
        keypoint_positions = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx < len(self.STANDARD_LANDMARKS):
                    landmark_name = self.STANDARD_LANDMARKS[idx]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    z = float(landmark.z)
                    confidence = landmark.visibility

                    data[landmark_name] = (x, y, z, confidence)
                    keypoint_positions[idx] = (x, y)

                    # Dibujar solo si se cumple que:
                    # - No se especificó selección (pintar todos), o
                    # - El landmark actual está en la lista de seleccionados.
                    if (selected_landmarks is None or landmark_name in selected_landmarks) and confidence > 0.2:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Dibujar conexiones del esqueleto
            for p1, p2 in self.SKELETON_CONNECTIONS:
                if p1 in keypoint_positions and p2 in keypoint_positions:
                    # Si se ha especificado una selección, dibujar la conexión solo si ambos landmarks están seleccionados
                    if selected_landmarks is None or (
                            self.STANDARD_LANDMARKS[p1] in selected_landmarks and self.STANDARD_LANDMARKS[
                        p2] in selected_landmarks):
                        x1, y1 = keypoint_positions[p1]
                        x2, y2 = keypoint_positions[p2]
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return data, frame
