import importlib
import numpy as np
import pandas as pd
import time

class HumanPoseProcessor:
    """General processor to unify pose detection from different models."""

    # 📌 Standardized landmark names based on MediaPipe's 33 keypoints
    STANDARD_LANDMARKS = [
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

    def __init__(self, processor_name):
        """
        Initializes the pose processor by dynamically loading the corresponding module.

        Args:
            processor_name (str): The processor module name (e.g., 'mediapipe_processor').
        """
        self.processor_name = processor_name
        self.processor = self.load_processor(processor_name)

    def load_processor(self, processor_name):
        """
        Dynamically loads the selected pose processor.

        Args:
            processor_name (str): The processor module name.

        Returns:
            object: An instance of the processor class.
        """
        module = importlib.import_module(processor_name)
        class_name = processor_name.split('_')[0].capitalize() + "Processor"
        return getattr(module, class_name)()

    def process_frame(self, frame):
        """
        Processes a video frame using the selected pose detection model
        and returns a standardized DataFrame.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            pandas.DataFrame: Normalized landmark coordinates.
        """
        # Get timestamp
        timestamp = time.time()

        # Get raw landmark data from the processor
        df_raw = self.processor.process(frame)

        # Create an empty dictionary with NaN values for all standard landmarks
        data = {lm: (np.nan, np.nan, np.nan, np.nan) for lm in self.STANDARD_LANDMARKS}

        # Fill the data dictionary with values from the detected landmarks
        for _, row in df_raw.iterrows():
            if row["landmark"] in self.STANDARD_LANDMARKS:
                data[row["landmark"]] = (row["x"], row["y"], row["z"], row["confidence"])

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data, orient="index", columns=["x", "y", "z", "confidence"])
        df.index.name = "landmark"
        df.reset_index(inplace=True)

        # Add timestamp
        df["timestamp"] = timestamp

        return df
