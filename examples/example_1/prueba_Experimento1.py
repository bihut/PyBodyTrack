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
import sys
import threading
import time
import cv2
from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods

# Define the video file path and output directory
video_path = "PATH TO VIDEO"
output_directory = "PATH TO STORE JSON WITH METRICS"

# Initialize a dictionary to store results
results = {}
results['exercise'] = video_path  # Assuming the video file is associated with an exercise

# Select body landmarks based on user specification
landmarks = bodyparts.STANDARD_LANDMARKS
if "lower" in video_path.lower():
    landmarks = bodyparts.LOWER_BODY_LANDMARKS
elif "upper" in video_path.lower():
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
if "start" in video_path:
    start_time = video_path['start']
if "end" in video_path:
    end_time = video_path['end']

# Check if the video file exists before proceeding
if os.path.exists(video_path):
    print("Video file exists.")
else:
    print("Error: Video file does not exist.")
    sys.exit(1)

# Set the time range for processing
body_tracking.set_times(start_time, end_time)

# Start the tracking process in a separate thread
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={'observer': None, 'fps': 30})
tracker_thread.start()

try:
    # Keep the main thread active while tracking runs
    while tracker_thread.is_alive():
        time.sleep(1)  # Sleep to reduce CPU usage
except KeyboardInterrupt:
    print("Stopping tracking...")
    body_tracking.stop()

# Ensure proper termination of the tracking process
body_tracking.stop()
tracker_thread.join(timeout=5)
if tracker_thread.is_alive():
    print("Warning: Tracker thread is still alive. Forcing stop...")
    body_tracking.stop()

# Retrieve the movement data from the tracker
df = body_tracking.getData()
df.to_csv("df.csv")  # Save raw data to CSV for debugging or analysis

# Compute movement metrics using Chebyshev distance
movement = Methods.chebyshev_distance(df, filter=True, distance_threshold=2.0)
normalized_movement = body_tracking.normalized_movement_index(movement, len(landmarks))

# Store movement metrics in the results dictionary
results['ram'] = movement  # Raw Amount of Movement (RAM)
results['nmi'] = normalized_movement  # Normalized Movement Index (NMI)
results['mol'] = body_tracking.movement_per_landmark(movement, len(bodyparts.STANDARD_LANDMARKS))  # Movement per Landmark (MOL)
results['mof'] = body_tracking.movement_per_frame(movement)  # Movement per Frame (MOF)
results['mos'] = body_tracking.movement_per_second(movement)  # Movement per Second (MOS)

# Define output file paths
json_output_path = os.path.join(output_directory, "output_cheby.json")
frame_output_path = os.path.join(output_directory, "random_frame.jpg")

# Remove existing files before saving new results
if os.path.exists(json_output_path):
    os.remove(json_output_path)

if os.path.exists(frame_output_path):
    os.remove(frame_output_path)

# Save the results as a JSON file
with open(json_output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)  # Pretty-print for better readability

# Save a random frame from the processed video
body_tracking.save_random_frame(frame_output_path)

print("Processing completed for:", video_path)
print("----------------------------")

# Ensure OpenCV windows are properly closed
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensure OpenCV processes the window closure correctly
except:
    pass

# Small pause to prevent rapid iterations causing issues
time.sleep(1)
