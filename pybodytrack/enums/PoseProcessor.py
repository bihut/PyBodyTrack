from enum import Enum


class PoseProcessor(Enum):
    YOLO= 0
    MEDIAPIPE= 1
    OPENPOSE=2