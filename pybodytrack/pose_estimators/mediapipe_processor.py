import cv2
import numpy as np
import mediapipe as mp

class MediaPipeProcessor:
    """Pose detector using MediaPipe."""

    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

        # 📌 Landmarks detectados por MediaPipe (33 puntos)
        self.STANDARD_LANDMARKS = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]

        self.SKELETON_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    def get_standard_landmarks(self):
        """Devuelve los landmarks estándar de MediaPipe"""
        return self.STANDARD_LANDMARKS

    def process(self, frame):
        data = {key: (np.nan, np.nan, np.nan, np.nan) for key in self.STANDARD_LANDMARKS}
        keypoint_positions = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx < len(self.STANDARD_LANDMARKS):
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    z = float(landmark.z)  # 📌 Asegurar que tomamos la coordenada Z real
                    confidence = landmark.visibility

                    data[self.STANDARD_LANDMARKS[idx]] = (x, y, z, confidence)
                    keypoint_positions[idx] = (x, y)

                    # 📌 Dibujar keypoints
                    if confidence > 0.2:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # 📌 Dibujar conexiones del esqueleto
            for p1, p2 in self.SKELETON_CONNECTIONS:
                if p1 in keypoint_positions and p2 in keypoint_positions:
                    x1, y1 = keypoint_positions[p1]
                    x2, y2 = keypoint_positions[p2]
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return data, frame
