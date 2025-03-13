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

res_json=[]

class CustomObserver(Observer):

    def __init__(self, frame_block_size=30):
        super().__init__()
        self.conttime = None
        self.cont_packages = 0
        self.frame_block_size = frame_block_size
        self.buffer = []  # Buffer to store incoming landmark data

    def handleMessage(self, msg):
        if msg.what == 1:  # New landmark data received

            block = msg.obj
            threading.Thread(target=self.processBuffer, args=(block,), daemon=True).start()
        else:
            # Handle other message types if needed
            print("Received error message:", msg.obj)

    def processBuffer(self, block):
        df_buffer = pd.DataFrame(block)
        self.cont_packages += self.frame_block_size
        movement = Methods.euclidean_distance(df_buffer, filter=True, distance_threshold=0.0)
        nmi = body_tracking.normalized_movement_index(movement, len(bodyparts.STANDARD_LANDMARKS))
        res = {"time":self.cont_packages,"movement":movement,"nmi":nmi}
        res_json.append(res)



output="results_falldetection.json"
body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.CAMERA,
                             path_video=None,
                             selected_landmarks=bodyparts.STANDARD_LANDMARKS)

fps=8
observer = CustomObserver(frame_block_size=fps)
observer.startLoop()
tracker_thread = threading.Thread(target=body_tracking.start, kwargs={
    'observer': observer,
    'fps': fps
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


with open(output, "w", encoding="utf-8") as file:
    json.dump(res_json, file, indent=4)  # `indent=4` para formato legible
