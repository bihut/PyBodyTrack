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



video = "PATH TO VIDEO"
path_output = "PATH TO STORE JSON WITH METRICS"
#for video in data['videos']:
path_video = "" #path_videos + "/" + video['name']
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

body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.VIDEO,
                             path_video=path_video,
                             selected_landmarks=landmark)

start = None
end = None
if "start" in video:
    start = video['start']
if "end" in video:
    end = video['end']
if os.path.exists(path_video):
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
if 1==1:
    sys.exit(0)
df = body_tracking.getData()

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

print("Terminado con ",name_video)
print("-------------------")
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Asegurar que OpenCV procesa correctamente el cierre

    # Pequeña pausa para evitar bloqueos en iteraciones rápidas

except:
    pass
time.sleep(1)
