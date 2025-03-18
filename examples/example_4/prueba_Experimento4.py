"""
pyBodyTrack - A Python package for motion quantification in videos.

Author: Angel Ruiz Zafra
License: MIT License
Version: 2025.2.1
Repository: https://github.com/bihut/pyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""
import json
import os
import threading
import time
import cv2

from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods

# Path to the JSON file containing experiment configuration
json_data_path = "/home/bihut/dev/Proyectos/pyBodyTrack/examples/example_4/experiment4.json"

# Load experiment configuration from JSON file
with open(json_data_path, "r") as file:
    data = json.load(file)

# Define video file path
video_path = "PATH TO VIDEO"

# Initialize results dictionary
results_json = {"exercise": "EXERCISE"}

# Select body landmarks based on predefined settings
landmark_type = "lower"
landmarks = bodyparts.STANDARD_LANDMARKS
if "lower" in landmark_type.lower():
    landmarks = bodyparts.LOWER_BODY_LANDMARKS
elif "upper" in landmark_type.lower():
    landmarks = bodyparts.UPPER_BODY_LANDMARKS

# Initialize the BodyTracking instance
body_tracking = BodyTracking(
    processor=PoseProcessor.MEDIAPIPE,  # Use MediaPipe for pose estimation
    mode=VideoMode.VIDEO,               # Process video mode
    path_video=video_path,               # Path to the input video
    selected_landmarks=landmarks         # Selected body landmarks
)

# Define start and end times for processing (if specified)
start_time = None
end_time = None

# Check if the video file exists before proceeding
if os.path.exists(video_path):
    print("Video file exists.")
else:
    print("Error: Video file does not exist.")
    exit(1)

# Set the time range for processing
body_tracking.set_times(start_time, end_time)

# Start the tracking process in a separate thread
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={'observer': None, 'fps': 30})
tracker_thread.start()

try:
    # Keep the main thread active while tracking is running
    while tracker_thread.is_alive():
        time.sleep(1)  # Sleep to prevent high CPU usage
except KeyboardInterrupt:
    print("Stopping tracking...")
    body_tracking.stop()

# Ensure proper termination of the tracking process
tracker_thread.join(timeout=5)
if tracker_thread.is_alive():
    print("Warning: Tracker thread is still alive. Forcing stop...")
    body_tracking.stop()

# Retrieve the movement data from the tracker
df = body_tracking.getData()
df.to_csv("df2.csv")  # Save raw data to CSV for debugging or analysis

# Compute movement metrics using Euclidean distance
movement = Methods.euclidean_distance(df, filter=True, distance_threshold=2.0)
normalized_movement = body_tracking.normalized_movement_index(movement, len(landmarks))

# Store movement metrics in the results dictionary
results_json['ram'] = movement  # Raw Amount of Movement (RAM)
results_json['nmi'] = normalized_movement  # Normalized Movement Index (NMI)
results_json['mol'] = body_tracking.movement_per_landmark(movement, len(bodyparts.STANDARD_LANDMARKS))  # Movement per Landmark (MOL)
results_json['mof'] = body_tracking.movement_per_frame(movement)  # Movement per Frame (MOF)
results_json['mos'] = body_tracking.movement_per_second(movement)  # Movement per Second (MOS)

# Define output file paths
output_json_path = "file_output_euclidean.json"
frame_output_path = ""

# Remove existing files before saving new results
if os.path.exists(output_json_path):
    os.remove(output_json_path)

if os.path.exists(frame_output_path):
    os.remove(frame_output_path)

# Save the results as a JSON file
with open(output_json_path, "w", encoding="utf-8") as file:
    json.dump(results_json, file, indent=4)  # Pretty-print JSON output for readability

# Save a random frame from the processed video
body_tracking.save_random_frame(frame_output_path)
