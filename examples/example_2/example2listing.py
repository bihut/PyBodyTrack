import threading
import time
from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods
path_video = "PATH TO VIDEO"
body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.VIDEO,
                             path_video=path_video,
                             selected_landmarks=bodyparts.STANDARD_LANDMARKS)

start = 10
end = 40

body_tracking.set_times(start, end)
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={})
tracker_thread.start()
try:
    while tracker_thread.is_alive():
        time.sleep(1)  # Main thread idle loop
        #print("dentro de la hebraq")
except KeyboardInterrupt:
    print("Stopping tracking...")
    body_tracking.stop()

# Detener el procesamiento correctamente
body_tracking.stop()
tracker_thread.join(timeout=5)
if tracker_thread.is_alive():
    print("Tracker thread still alive. Force stopping...")
    body_tracking.stop()
df = body_tracking.getData()
df.to_csv("df2.csv")
movement = Methods.chebyshev_distance(df, filter=True, distance_threshold=2.0)
norm = body_tracking.normalized_movement_index(movement, len(bodyparts.STANDARD_LANDMARKS))
print("Raw movement:",movement, " - NMI:",norm)
