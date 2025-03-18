"""
PyBodyTrack - A Python library for motion quantification in videos.

Author: Angel Ruiz Zafra
License: Apache 2.0 License
Version: 2025.3.2
Repository: https://github.com/bihut/PyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""

from enum import Enum

class PoseProcessor(Enum):
    YOLO= 0
    MEDIAPIPE= 1
    OPENPOSE=2