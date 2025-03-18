"""
pyBodyTrack - A Python package for motion quantification in videos.

Author: Angel Ruiz Zafra
License: MIT License
Version: 2025.2.1
Repository: https://github.com/bihut/pyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""
import json
import threading
import time
import pandas as pd

from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods
from pybodytrack.observer.Observer import Observer

# List to store movement data results
results_json = []

class CustomObserver(Observer):
    """
    Custom observer class to process landmark data received from BodyTracking.
    It processes data in blocks of frames and calculates movement metrics.
    """

    def __init__(self, frame_block_size=30):
        """
        Initializes the observer with a specified frame block size.

        Args:
            frame_block_size (int): Number of frames processed per batch.
        """
        super().__init__()
        self.cont_packages = 0  # Counter for processed frames
        self.frame_block_size = frame_block_size
        self.buffer = []  # Buffer to store incoming landmark data

    def handleMessage(self, msg):
        """
        Handles incoming messages from the BodyTracking process.

        Args:
            msg: The received message object.
        """
        if msg.what == 1:  # New landmark data received
            block = msg.obj
            # Process the data in a separate thread to avoid blocking
            threading.Thread(target=self.process_buffer, args=(block,), daemon=True).start()
        else:
            print("Received error message:", msg.obj)

    def process_buffer(self, block):
        """
        Processes a block of landmark data and calculates movement metrics.

        Args:
            block (list): List of landmark data frames.
        """
        df_buffer = pd.DataFrame(block)
        self.cont_packages += self.frame_block_size  # Update frame count

        # Calculate movement metrics using Euclidean distance
        movement = Methods.euclidean_distance(df_buffer, filter=True, distance_threshold=0.0)
        nmi = body_tracking.normalized_movement_index(movement, len(bodyparts.STANDARD_LANDMARKS))

        # Store results in JSON format
        result = {"time": self.cont_packages, "movement": movement, "nmi": nmi}
        results_json.append(result)

# Define output file for storing results
output_file = "results_falldetection.json"

# Initialize the BodyTracking instance with MediaPipe and camera input
body_tracking = BodyTracking(
    processor=PoseProcessor.MEDIAPIPE,  # Use MediaPipe for pose estimation
    mode=VideoMode.CAMERA,              # Process real-time camera input
    path_video=None,                     # No pre-recorded video used
    selected_landmarks=bodyparts.STANDARD_LANDMARKS  # Track standard body landmarks
)

# Define frames per second for processing
fps = 8

# Initialize the observer and start its processing loop
observer = CustomObserver(frame_block_size=fps)
observer.startLoop()

# Start the tracking process in a separate thread
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={'observer': observer, 'fps': fps})
tracker_thread.start()

try:
    # Keep the main thread active while tracking is running
    while tracker_thread.is_alive():
        time.sleep(1)  # Sleep to prevent high CPU usage
except KeyboardInterrupt:
    print("Stopping tracking...")
    body_tracking.stop()

# Ensure proper termination of the tracking process
body_tracking.stop()
tracker_thread.join(timeout=5)
if tracker_thread.is_alive():
    print("Warning: Tracker thread is still alive. Forcing stop...")
    body_tracking.stop()

# Save results as a JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(results_json, file, indent=4)  # Pretty-print JSON output for readability
