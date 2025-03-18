"""
pyBodyTrack - A Python package for motion quantification in videos.

Author: Angel Ruiz Zafra
License: MIT License
Version: 2025.2.1
Repository: https://github.com/bihut/pyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""
import threading
import time
from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods

# Path to the input video file
path_video = "PATH TO VIDEO"

# Initialize the BodyTracking object with the selected processor and mode
body_tracking = BodyTracking(
    processor=PoseProcessor.MEDIAPIPE,  # Use MediaPipe for pose estimation
    mode=VideoMode.VIDEO,               # Set mode to process a video file
    path_video=path_video,              # Path to the video file
    selected_landmarks=bodyparts.STANDARD_LANDMARKS  # Use standard body landmarks
)

# Define the time range for processing (in seconds)
start = 10
end = 40
body_tracking.set_times(start, end)

# Create and start a separate thread for tracking
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={})
tracker_thread.start()

try:
    # Main thread stays active while the tracking thread runs
    while tracker_thread.is_alive():
        time.sleep(1)  # Prevents busy-waiting by sleeping 1 second per loop
except KeyboardInterrupt:
    print("Stopping tracking...")
    body_tracking.stop()

# Ensure proper shutdown of the tracking thread
tracker_thread.join(timeout=1)
if tracker_thread.is_alive():
    print("Tracker thread still alive. Force stopping...")
    body_tracking.stop()

# Retrieve movement data
df = body_tracking.getData()

# Compute movement metrics using Chebyshev distance
movement = Methods.chebyshev_distance(df, filter=True, distance_threshold=2.0)

# Initialize the result JSON dictionary
res_json = {}

# Compute Normalized Movement Index (NMI)
norm = body_tracking.normalized_movement_index(movement, len(bodyparts.STANDARD_LANDMARKS))
res_json['ram'] = movement  # Raw Amount of Movement (RAM)
res_json['nmi'] = norm      # Normalized Movement Index (NMI)

# Compute Movement per Landmark (MOL)
movl = body_tracking.movement_per_landmark(movement, len(bodyparts.STANDARD_LANDMARKS))
res_json['mol'] = movl

# Compute Movement per Frame (MOF)
aux = body_tracking.movement_per_frame(movement)
res_json['mof'] = aux

# Compute Movement per Second (MOS)
aux = body_tracking.movement_per_second(movement)
res_json['mos'] = aux

# Print the results
print("Raw movement:", movement, " - NMI:", norm)
