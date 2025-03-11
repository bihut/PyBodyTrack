# body_parts.py

# Standard list of landmarks (excluding timestamp)
STANDARD_LANDMARKS = [
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
STANDARD_LANDMARKS_YOLO = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

STANDARD_LANDMARKS_OPENPOSE = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear"
]

# Landmarks for the head (timestamp omitted)
HEAD_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right"
]

# Landmarks for the left arm (timestamp omitted)
LEFT_ARM_LANDMARKS = [
    "left_shoulder", "left_elbow", "left_wrist",
    "left_pinky", "left_index", "left_thumb"
]

# Landmarks for the right arm (timestamp omitted)
RIGHT_ARM_LANDMARKS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "right_pinky", "right_index", "right_thumb"
]

# Landmarks for the left leg (timestamp omitted)
LEFT_LEG_LANDMARKS = [
    "left_hip", "left_knee", "left_ankle",
    "left_heel", "left_foot_index"
]

# Landmarks for the right leg (timestamp omitted)
RIGHT_LEG_LANDMARKS = [
    "right_hip", "right_knee", "right_ankle",
    "right_heel", "right_foot_index"
]

# Landmarks for the trunk (timestamp omitted)
TRUNK_LANDMARKS = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip"
]

# Landmarks for the upper body (combining head, arms and part of trunk; timestamp omitted)
UPPER_BODY_LANDMARKS = [
    # Head
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    # Arms
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    # Trunk (upper part)
    "left_hip", "right_hip"
]

# Landmarks for the lower body (timestamp omitted)
LOWER_BODY_LANDMARKS = [
    # Left leg
    "left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index",
    # Right leg
    "right_hip", "right_knee", "right_ankle", "right_heel", "right_foot_index"
]

# Diccionario que asocia el nombre de la parte del cuerpo con su lista de landmarks
BODY_PARTS = {
    "standard": STANDARD_LANDMARKS,
    "head": HEAD_LANDMARKS,
    "left_arm": LEFT_ARM_LANDMARKS,
    "right_arm": RIGHT_ARM_LANDMARKS,
    "left_leg": LEFT_LEG_LANDMARKS,
    "right_leg": RIGHT_LEG_LANDMARKS,
    "trunk": TRUNK_LANDMARKS,
    "upper_body": UPPER_BODY_LANDMARKS,
    "lower_body": LOWER_BODY_LANDMARKS,
}

def get_columns_for_part(part: str, coordinate_suffixes=["_x", "_y"]):
    """
    Returns a list of column names for the specified body part.

    Parameters:
        part (str): The body part to filter (e.g., "upper_body" or "lower_body").
        coordinate_suffixes (list): List of coordinate suffixes to append for each landmark.

    Returns:
        List[str]: The list of columns corresponding to the landmarks of the specified body part.
    """
    landmarks = BODY_PARTS.get(part.lower())
    if landmarks is None:
        raise ValueError(
            f"Body part '{part}' is not defined. Available parts: {list(BODY_PARTS.keys())}"
        )

    columns = []
    for landmark in landmarks:
        for suffix in coordinate_suffixes:
            columns.append(f"{landmark}{suffix}")
    return columns
