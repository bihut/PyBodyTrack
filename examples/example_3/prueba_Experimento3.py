import json
import os.path

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

res_json=[]

class CustomObserver(Observer):

    def __init__(self, frame_block_size=30):
        """
        Parameters:
            frame_block_size (int): Number of frames to accumulate before processing.
        """
        super().__init__()
        self.conttime = None
        self.cont_packages = 0
        self.frame_block_size = frame_block_size
        self.buffer = []  # Buffer to store incoming landmark data

    def handleMessage(self, msg):
        print("MENSAJE RECIBIDO HANDLE")
        if msg.what == 1:  # New landmark data received
            #self.buffer = []  # Buffer to store incoming landmark data
            #self.buffer.append(msg.obj)
            #if len(self.buffer) >= self.frame_block_size:
                # Get a block of frames (non-overlapping)
            #block = self.buffer[:self.frame_block_size]
                # Remove the processed frames from the buffer
            #self.buffer = self.buffer[self.frame_block_size:]
                # Offload processing to another thread
            block = msg.obj
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
        print("MENSAJE RECIBIDO HANDLER","PROCESAR")
        if self.conttime is None:
            self.conttime = time.time()
        df_buffer = pd.DataFrame(block)
        #start_time = df_buffer.iloc[0]['timestamp']
        #end_time = df_buffer.iloc[-1]['timestamp']
        # Perform heavy processing here (for instance, calculating movement)
        # Example: movement = Methods.euclidean_distance(df_buffer)
        # For demonstration, we simply print the information:
        self.cont_packages += self.frame_block_size
        #print(f"Processing block from {start_time} to {end_time} with {len(df_buffer)} frames.")
        movement = Methods.euclidean_distance(df_buffer, filter=True, distance_threshold=0.0)
        print("Cantidad de movimiento euclidean:",movement)
        nmi = body_tracking.normalized_movement_index(movement, len(bodyparts.STANDARD_LANDMARKS))
        print("NMI:",nmi)
        res = {"time":self.cont_packages,"movement":movement,"nmi":nmi}
        res_json.append(res)
        print("PAquete ",self.cont_packages)

        self.conttime = time.time()


path_video_before = "/home/bihut/Documentos/UGR/Papers/pyBodyTrack-SoftwareX/ExperimentosVideos/Experimento3/andres.mp4"

if not os.path.exists(path_video_before):
    print("El no fichero existe")
output = "/home/bihut/dev/Proyectos/pyBodyTrack/examples/example_3"
body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.VIDEO,
                             path_video=path_video_before,
                             selected_landmarks=bodyparts.STANDARD_LANDMARKS)

#start = 1
#end = 7



#body_tracking.set_times(start, end)
fps=8
observer = CustomObserver(frame_block_size=fps)
observer.startLoop()
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={
    'observer': observer,
    'fps': fps
})
conttime = time.time()
cont_packages=0
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

output_res = output + "/video_output_euclidean.json"
path_frame = output + "/video_frame.jpg"
if os.path.exists(output_res):
    os.remove(output_res)

if os.path.exists(path_frame):
    os.remove(path_frame)

with open(output_res, "w", encoding="utf-8") as file:
    json.dump(res_json, file, indent=4)  # `indent=4` para formato legible

#df = body_tracking.getData()
#df=Utils.remove_empty_rows(df)
#df = Utils.recalculate_timestamps(df,body_tracking.fps)
'''
df.to_csv("df2.csv")

movement = Methods.chebyshev_distance(df, filter=True, distance_threshold=2.0)
norm = body_tracking.normalized_movement_index(movement, len(bodyparts.STANDARD_LANDMARKS))
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
output_res = output + "/video_output_cheby.json"
path_frame = output + "/video_frame.jpg"
if os.path.exists(output_res):
    os.remove(output_res)

if os.path.exists(path_frame):
    os.remove(path_frame)

with open(output_res, "w", encoding="utf-8") as file:
    json.dump(res_json, file, indent=4)  # `indent=4` para formato legible

body_tracking.save_random_frame(path_frame)
#print(video['output'])
#print(video['output']+"_frame.jpg")
print("Terminado con",video)
'''
print("-------------------")
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Asegurar que OpenCV procesa correctamente el cierre

    # Pequeña pausa para evitar bloqueos en iteraciones rápidas

except:
    pass
time.sleep(1)
