import json
import os.path
import sys
import threading
import time

import cv2
import pandas as pd

from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods
from pybodytrack.observer.Observer import Observer
from pybodytrack.utils.utils import Utils


class CustomObserver(Observer):
    def __init__(self, frame_block_size=30):
        """
        Parameters:
            frame_block_size (int): Number of frames to accumulate before processing.
        """
        super().__init__()
        self.frame_block_size = frame_block_size
        self.buffer = []  # Buffer to store incoming landmark data

    def handleMessage(self, msg):
        if msg.what == 1:  # New landmark data received
            self.buffer.append(msg.obj)
            if len(self.buffer) >= self.frame_block_size:
                # Get a block of frames (non-overlapping)
                block = self.buffer[:self.frame_block_size]
                # Remove the processed frames from the buffer
                self.buffer = self.buffer[self.frame_block_size:]
                # Offload processing to another thread
                threading.Thread(target=self.processBuffer, args=(block,), daemon=True).start()
        else:
            # Handle other message types if needed
            print("Received error message:", msg.obj)

    def processBuffer(self, block):
        """
        Process a block of landmark data on a separate thread.

        Converts the block (list of rows) into a DataFrame and applies any heavy processing
        (for example, computing movement). This runs in a separate thread so that the video loop is not blocked.
        """
        df_buffer = pd.DataFrame(block)
        start_time = df_buffer.iloc[0]['timestamp']
        end_time = df_buffer.iloc[-1]['timestamp']
        # Perform heavy processing here (for instance, calculating movement)
        # Example: movement = Methods.euclidean_distance(df_buffer)
        # For demonstration, we simply print the information:
        print(f"Processing block from {start_time} to {end_time} with {len(df_buffer)} frames.")
        movement = Methods.euclidean_distance(df_buffer)
        print("Cantidad de movimiento euclidean:",movement)

json_data ="/home/bihut/dev/Proyectos/pyBodyTrack/examples/example_1/experiment1.json"

with open(json_data, "r") as file:
    data = json.load(file)
path_videos = data['path_videos']
path_output = data['path_output']
video = data['videos'][27]
#for video in data['videos']:
path_video = path_videos + "/" + video['name']
name_video = video['name']
print("Voy con ",name_video)
res_json = {}
res_json['exercise']=video['exercise']

landmark = bodyparts.STANDARD_LANDMARKS
out_res = path_output
if "lower" in video['landmarks'].lower():
    landmark = bodyparts.LOWER_BODY_LANDMARKS
elif "upper" in video['landmarks'].lower():
    landmark = bodyparts.UPPER_BODY_LANDMARKS
print("PATH VIDEO :",path_videos + "/" + name_video)
body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.VIDEO,
                             path_video=path_videos + "/" + name_video,
                             selected_landmarks=landmark)

start = None
end = None
if "start" in video:
    start = video['start']
if "end" in video:
    end = video['end']
if os.path.exists(path_videos + "/" + name_video):
    print("El fichero existe")
else:
    print("El fichero no existe")
body_tracking.set_times(start, end)
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={
    'observer': None,
    'fps': 30
})
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
#df=Utils.remove_empty_rows(df)
#df = Utils.recalculate_timestamps(df,body_tracking.fps)
df.to_csv("df.csv")
movement = Methods.chebyshev_distance(df, filter=True, distance_threshold=2.0)
norm = body_tracking.normalized_movement_index(movement, len(landmark))
res_json['ram'] = movement
res_json['nmi'] = norm
# print("normalized_movement_index:",norm)
movl = body_tracking.movement_per_landmark(movement, len(bodyparts.STANDARD_LANDMARKS))
res_json['mol'] = movl
aux = body_tracking.movement_per_frame(movement)
res_json['mof'] = aux
aux = body_tracking.movement_per_second(movement)
res_json['mos'] = aux
#body_tracking.stop()
# Guardar en un fichero JSON
output_res = path_output + "/" + name_video + "_output_cheby.json"
path_frame = path_output + "/" + name_video + "_frame.jpg"
if os.path.exists(output_res):
    os.remove(output_res)

if os.path.exists(path_frame):
    os.remove(path_frame)

with open(output_res, "w", encoding="utf-8") as file:
    json.dump(res_json, file, indent=4)  # `indent=4` para formato legible

body_tracking.save_random_frame(path_frame)
#print(video['output'])
#print(video['output']+"_frame.jpg")
print("Terminado con ",name_video)
print("-------------------")
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Asegurar que OpenCV procesa correctamente el cierre

    # Pequeña pausa para evitar bloqueos en iteraciones rápidas

except:
    pass
time.sleep(1)
if 1==1:
    sys.exit()

path_videos = "/home/bihut/Documentos/UGR/Papers/pyBodyTrack - Software X/Experimentos Videos/Experimento1/Experimento1"
video = "Barbell_bicep_curl 10.mp4"
path_frame = path_videos+"/"+video+"_frame.jpg"
output_res = path_videos+"/"+video+"_output_cheby.json"
res_json = {}

#path_videos = "/home/bihut/Imágenes/egipto/video7.mp4"
landmarks = bodyparts.UPPER_BODY_LANDMARKS
res_json["landmarks"] = "Upper Body"
#observer = CustomObserver()
#observer.startLoop()

body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.VIDEO, path_video=path_videos+"/"+video,
                             selected_landmarks=landmarks)
#body_tracking.start()
#body_tracking.set_times(2,17)

#FUNCIONALIDAD - METER VARIOS VIDEOS Y QUE LOS ORDENE POR CANTIDAD DE MOVIMIENTO
#FUNCIONALIDAD - METER DOS VIDEOS Y DECIR CUAL TIENE MAS MOVIMIENTO y LA PROPORCION
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={
    'observer': None,
    'fps': 30
})
tracker_thread.start()
# Start the tracking in a separate thread (since start() is blocking)

#tracker_thread = threading.Thread(target=body_tracking.start, kwargs={
#        'observer': None,
#        'fps': 30
#    })
#tracker_thread.start()


try:
    while tracker_thread.is_alive():
        time.sleep(1)  # Main thread idle loop
except KeyboardInterrupt:
    print("Stopping tracking...")
    body_tracking.stop()

tracker_thread.join()
print("HA FINALIZADO, vamos a obtener los datos")
df = body_tracking.getData()
movement = Methods.chebyshev_distance(df,filter=True,distance_threshold=2.0)
norm=body_tracking.normalized_movement_index(movement,len(landmarks))
res_json['ram'] = movement
res_json['nmi'] = norm
#print("normalized_movement_index:",norm)
movl=body_tracking.movement_per_landmark(movement, len(bodyparts.STANDARD_LANDMARKS))
res_json['mol'] = movl
aux = body_tracking.movement_per_frame(movement)
res_json['mof'] = aux
aux= body_tracking.movement_per_second(movement)
res_json['mos'] = aux

# Guardar en un fichero JSON
with open(output_res, "w", encoding="utf-8") as file:
    json.dump(res_json, file, indent=4)  # `indent=4` para formato legible

body_tracking.save_random_frame(path_frame)
#print("movement_per_landmark:",movl)
#body_tracking.stats_summary(movement)
if 1==1:
    sys.exit()

observer = MovementObserver()
body_tracking.start()

df = body_tracking.getData()
#df = body_tracking.filter_interval(10,45)
#55,75
#columns=bodyparts.get_columns_for_part("lower_body")
#df2=Utils.get_sub_landmark(df,columns)
movement = Methods.euclidean_distance(df,filter=True,distance_threshold=2.0)
norm=body_tracking.normalized_movement_index(movement,len(landmarks))
print("normalized_movement_index:",norm)
movl=body_tracking.movement_per_landmark(movement, len(bodyparts.STANDARD_LANDMARKS))
print("movement_per_landmark:",movl)
body_tracking.stats_summary(movement)
'''
print("cantidad de movimiento euclidean:",movement)
a=Utils.movement_per_second(movement,df)
print("cantidad de movimiento por segundo:",a)
a = Utils.movement_per_frame(movement, df)
print("cantidad de movimiento por frame:",a)
a = Utils.movement_per_landmark(movement, len(bodyparts.TRUNK_LANDMARKS))
print("cantidad de movimiento por landmark:",a)
a = Utils.normalized_movement_index(movement,df,len(bodyparts.TRUNK_LANDMARKS))
print("cantidad de movimiento normalizado:",a)

num_landmarks = (len(df.columns) - 1) // 4
frame_movements = []

# For each consecutive pair of frames, sum the Euclidean movement for each landmark.
for i in range(1, len(df)):
    frame_distance = 0.0
    for lm in range(num_landmarks):
        base = lm * 4
        col_x = df.columns[1 + base]
        col_y = df.columns[1 + base + 1]
        col_z = df.columns[1 + base + 2]
        dx = df.iloc[i][col_x] - df.iloc[i-1][col_x]
        dy = df.iloc[i][col_y] - df.iloc[i-1][col_y]
        dz = df.iloc[i][col_z] - df.iloc[i-1][col_z]
        frame_distance += np.sqrt(dx**2 + dy**2 + dz**2)
    frame_movements.append(frame_distance)

stats = Utils.frame_movement_statistics(frame_movements)

# Compute movement per second.
duration = df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']
movement_per_second = movement / duration if duration > 0 else 0.0
print("---------------------")
print("Movement Per Second:", movement_per_second)
print("Frame Movement Statistics:")
print(f"  Average: {stats.get('average'):.2f}")
print(f"  Std Dev: {stats.get('std_dev'):.2f}")
print(f"  Median: {stats.get('median'):.2f}")
print(f"  95th Percentile: {stats.get('p95'):.2f}")
'''
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