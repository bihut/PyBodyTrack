"""
pyBodyTrack - A Python package for motion quantification in videos.

Author: Angel Ruiz Zafra
License: MIT License
Version: 2025.2.1
Repository: https://github.com/bihut/pyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""
from enum import Enum

class PoseProcessor(Enum):
    YOLO= 0
    MEDIAPIPE= 1
    OPENPOSE=2