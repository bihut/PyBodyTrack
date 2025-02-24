from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods
from pybodytrack.utils.utils import Utils

body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.VIDEO, path_video="/home/bihut/Imágenes/squat.mp4",
                             selected_landmarks=bodyparts.TRUNK_LANDMARKS)
body_tracking.start()
df = body_tracking.getData()
#columns=bodyparts.get_columns_for_part("lower_body")
#df2=Utils.get_sub_landmark(df,columns)
movement = Methods.euclidean_distance(df,filter=True,distance_threshold=2.0)
print("cantidad de movimiento euclidean:",movement)
'''
import queue
import threading
import time

import cv2
from pybodytrack.pose_estimators.yolo_processor import YoloProcessor
from pybodytrack.pose_estimators.mediapipe_processor import MediaPipeProcessor
from pybodytrack.pose_estimators.camera_pose_tracker import CameraPoseTracker
custom_model_path = "/home/bihut/dev/Proyectos/pyBodyTrack/yolov8n-pose.pt"  # 🛠️ Cambia esto si tienes un modelo diferente

#processor = YoloProcessor(model_path=custom_model_path)
processor = MediaPipeProcessor()
# 📌 Selecciona el modelo: Mediapipe o YOLO
#processor = YoloProcessor()  # Cambia a YoloProcessor() si quieres usar YOLO
text= "YOLO" if isinstance(processor, YoloProcessor) else "MediaPipe"
# 📌 Inicializar tracker
tracker = CameraPoseTracker(processor)

mode = 1  # Change to 1 to use the video file
path_video = "/home/bihut/Imágenes/squat.mp4"  # Replace with your video file path

# Open video source
if mode == 0:
    cap = cv2.VideoCapture(0)
    fps = 30  # Assume 30 fps for camera
elif mode == 1:
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default if unable to get FPS
else:
    raise ValueError("Invalid mode selected. Use 0 for camera or 1 for video file.")

frame_interval = 1.0 / fps

# Shared variables and a lock for synchronization
latest_frame_lock = threading.Lock()
frame_to_process = None         # Latest frame available for processing
latest_processed_frame = None    # Latest processed frame (with skeleton)
stop_processing = False

def processing_thread_func():
    global frame_to_process, latest_processed_frame, stop_processing
    while not stop_processing:
        # Copy the latest frame for processing
        with latest_frame_lock:
            if frame_to_process is not None:
                frame = frame_to_process.copy()
            else:
                frame = None
        if frame is not None:
            # Process the frame (tracker.process_frame should draw the skeleton on the frame)
            tracker.process_frame(frame)
            # Save the processed frame in the shared variable
            with latest_frame_lock:
                latest_processed_frame = frame
        else:
            time.sleep(0.001)  # small delay to avoid busy waiting

# Start the processing thread
processing_thread = threading.Thread(target=processing_thread_func)
processing_thread.start()

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Update the frame to process with the latest capture
    with latest_frame_lock:
        frame_to_process = frame.copy()
        # Use the processed frame if available; otherwise, use the raw frame
        display_frame = latest_processed_frame.copy() if latest_processed_frame is not None else frame.copy()

    cv2.imshow("Pose Tracking", display_frame)

    elapsed_time = time.time() - start_time
    remaining_time = frame_interval - elapsed_time
    if remaining_time > 0:
        time.sleep(remaining_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_processing = True
processing_thread.join()
cap.release()
cv2.destroyAllWindows()

tracker.save_to_csv("pose_data" + text + ".csv")
'''