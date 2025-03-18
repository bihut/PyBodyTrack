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
from pybodytrack.bodyparts import body_parts as bodyparts
import pyopenpose as op

class OpenPoseProcessor:
    """Pose detector using OpenPose."""

    def __init__(self, model_path="/path/to/openpose/models/"):
        params = {
            "model_folder": model_path,
            "hand": False,
            "face": False,
            "net_resolution": "-1x368"
        }
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        # 📌 OpenPose detecta 25 keypoints
        self.STANDARD_LANDMARKS = bodyparts.STANDARD_LANDMARKS_OPENPOSE

        # 📌 Conexiones del esqueleto en OpenPose
        self.SKELETON_CONNECTIONS = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
            (10, 14), (14, 15), (15, 19), (15, 21), (15, 17),
            (13, 16), (16, 22), (16, 20), (16, 18)
        ]

    def get_standard_landmarks(self):
        """Devuelve los landmarks estándar de OpenPose"""
        return self.STANDARD_LANDMARKS

    def process(self, frame):
        datum = op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([datum])

        data = {key: (np.nan, np.nan, np.nan, np.nan) for key in self.STANDARD_LANDMARKS}
        keypoint_positions = {}

        if datum.poseKeypoints is not None:
            for person in datum.poseKeypoints:
                for idx, keypoint in enumerate(person):
                    if idx < len(self.STANDARD_LANDMARKS):
                        x, y, confidence = keypoint
                        x, y = int(x), int(y)
                        data[self.STANDARD_LANDMARKS[idx]] = (x, y, 0, confidence)
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
