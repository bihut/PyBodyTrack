# config.py - Configuration file for pyBodyTrack
LANDMARKS = [
    "PoseLandmark.NOSE", "PoseLandmark.LEFT_EYE_INNER", "PoseLandmark.LEFT_EYE",
    "PoseLandmark.LEFT_EYE_OUTER", "PoseLandmark.RIGHT_EYE_INNER", "PoseLandmark.RIGHT_EYE",
    "PoseLandmark.RIGHT_EYE_OUTER", "PoseLandmark.LEFT_EAR", "PoseLandmark.RIGHT_EAR",
    "PoseLandmark.MOUTH_LEFT", "PoseLandmark.MOUTH_RIGHT", "PoseLandmark.LEFT_SHOULDER",
    "PoseLandmark.RIGHT_SHOULDER", "PoseLandmark.LEFT_ELBOW", "PoseLandmark.RIGHT_ELBOW",
    "PoseLandmark.LEFT_WRIST", "PoseLandmark.RIGHT_WRIST", "PoseLandmark.LEFT_PINKY",
    "PoseLandmark.RIGHT_PINKY", "PoseLandmark.LEFT_INDEX", "PoseLandmark.RIGHT_INDEX",
    "PoseLandmark.LEFT_THUMB", "PoseLandmark.RIGHT_THUMB", "PoseLandmark.LEFT_HIP",
    "PoseLandmark.RIGHT_HIP", "PoseLandmark.LEFT_KNEE", "PoseLandmark.RIGHT_KNEE",
    "PoseLandmark.LEFT_ANKLE", "PoseLandmark.RIGHT_ANKLE", "PoseLandmark.LEFT_HEEL",
    "PoseLandmark.RIGHT_HEEL", "PoseLandmark.LEFT_FOOT_INDEX", "PoseLandmark.RIGHT_FOOT_INDEX",
    "timestamp"
]

LANDMARKS_HEAD = [
    "PoseLandmark.NOSE",
    "PoseLandmark.LEFT_EYE_INNER",
    "PoseLandmark.LEFT_EYE",
    "PoseLandmark.LEFT_EYE_OUTER",
    "PoseLandmark.RIGHT_EYE_INNER",
    "PoseLandmark.RIGHT_EYE",
    "PoseLandmark.RIGHT_EYE_OUTER",
    "PoseLandmark.LEFT_EAR",
    "PoseLandmark.RIGHT_EAR",
    "PoseLandmark.MOUTH_LEFT",
    "PoseLandmark.MOUTH_RIGHT",
    "timestamp"
]
LANDMARKS_LEFT_ARM = [
    "PoseLandmark.LEFT_SHOULDER",
    "PoseLandmark.LEFT_ELBOW",
    "PoseLandmark.LEFT_WRIST",
    "PoseLandmark.LEFT_PINKY",
    "PoseLandmark.LEFT_INDEX",
    "PoseLandmark.LEFT_THUMB",
    "timestamp"
]
LANDMARKS_RIGHT_ARM = [
    "PoseLandmark.RIGHT_SHOULDER",
    "PoseLandmark.RIGHT_ELBOW",
    "PoseLandmark.RIGHT_WRIST",
    "PoseLandmark.RIGHT_PINKY",
    "PoseLandmark.RIGHT_INDEX",
    "PoseLandmark.RIGHT_THUMB",
    "timestamp"
]

LANDMARKS_LEFT_LEG = [
    "PoseLandmark.LEFT_HIP",
    "PoseLandmark.LEFT_KNEE",
    "PoseLandmark.LEFT_ANKLE",
    "PoseLandmark.LEFT_HEEL",
    "PoseLandmark.LEFT_FOOT_INDEX",
    "timestamp"
]

LANDMARKS_RIGHT_LEG= [
    "PoseLandmark.RIGHT_HIP",
    "PoseLandmark.RIGHT_KNEE",
    "PoseLandmark.RIGHT_ANKLE",
    "PoseLandmark.RIGHT_HEEL",
    "PoseLandmark.RIGHT_FOOT_INDEX",
    "timestamp"

]

LANDMARKS_TRUNK = [
    "PoseLandmark.LEFT_SHOULDER",
    "PoseLandmark.RIGHT_SHOULDER",
    "PoseLandmark.LEFT_HIP",
    "PoseLandmark.RIGHT_HIP",
    "timestamp"
]

LANDMARKS_UPPER_BODY = [
    # HEAD
    "PoseLandmark.NOSE",
    "PoseLandmark.LEFT_EYE_INNER",
    "PoseLandmark.LEFT_EYE",
    "PoseLandmark.LEFT_EYE_OUTER",
    "PoseLandmark.RIGHT_EYE_INNER",
    "PoseLandmark.RIGHT_EYE",
    "PoseLandmark.RIGHT_EYE_OUTER",
    "PoseLandmark.LEFT_EAR",
    "PoseLandmark.RIGHT_EAR",
    "PoseLandmark.MOUTH_LEFT",
    "PoseLandmark.MOUTH_RIGHT",
    # ARMS
    "PoseLandmark.LEFT_SHOULDER",
    "PoseLandmark.RIGHT_SHOULDER",
    "PoseLandmark.LEFT_ELBOW",
    "PoseLandmark.RIGHT_ELBOW",
    "PoseLandmark.LEFT_WRIST",
    "PoseLandmark.RIGHT_WRIST",
    "PoseLandmark.LEFT_PINKY",
    "PoseLandmark.RIGHT_PINKY",
    "PoseLandmark.LEFT_INDEX",
    "PoseLandmark.RIGHT_INDEX",
    "PoseLandmark.LEFT_THUMB",
    "PoseLandmark.RIGHT_THUMB",
    # TRUNK
    "PoseLandmark.LEFT_HIP",
    "PoseLandmark.RIGHT_HIP",
    "timestamp"
]

LANDMARKS_LOWER_BODY = [
    # LEFT LEG
    "PoseLandmark.LEFT_HIP",
    "PoseLandmark.LEFT_KNEE",
    "PoseLandmark.LEFT_ANKLE",
    "PoseLandmark.LEFT_HEEL",
    "PoseLandmark.LEFT_FOOT_INDEX",
    # RIGHT LEG
    "PoseLandmark.RIGHT_HIP",
    "PoseLandmark.RIGHT_KNEE",
    "PoseLandmark.RIGHT_ANKLE",
    "PoseLandmark.RIGHT_HEEL",
    "PoseLandmark.RIGHT_FOOT_INDEX",
    "timestamp"
]
#from pybodytrack.config import LANDMARKS