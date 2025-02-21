import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import time


class MediaPipeProcessor:
    """Processor for MediaPipe pose estimation."""

    LANDMARKS = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]

    def __init__(self):
        """Initialize MediaPipe Pose model."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def process(self, frame):
        """Processes the frame and returns a normalized DataFrame."""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        data = {lm: (np.nan, np.nan, np.nan, np.nan) for lm in self.LANDMARKS}
        timestamp = time.time()

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                if idx < len(self.LANDMARKS):
                    std_name = self.LANDMARKS[idx]
                    data[std_name] = (landmark.x, landmark.y, landmark.z, landmark.visibility)

        df = pd.DataFrame.from_dict(data, orient="index", columns=["x", "y", "z", "confidence"])
        df.index.name = "landmark"
        df.reset_index(inplace=True)
        df["timestamp"] = timestamp
        return df
