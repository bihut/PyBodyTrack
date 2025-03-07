import json
import os.path

import threading
import time

import cv2

from pybodytrack.BodyTracking import BodyTracking
from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.bodyparts import body_parts as bodyparts
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.methods.methods import Methods
video=1
path_video_before = "/home/bihut/Documentos/NRP-EnsayosJunio/ensayosFinales/kylian/kylian_14/videos/ai_camera/kylian_14_1_center.avi"
if not os.path.exists(path_video_before):
    print("El fichero no existe")
output = "/home/bihut/dev/Proyectos/pyBodyTrack/examples/example_2"
body_tracking = BodyTracking(processor=PoseProcessor.MEDIAPIPE, mode=VideoMode.VIDEO,
                             path_video=path_video_before,
                             selected_landmarks=bodyparts.STANDARD_LANDMARKS)

start = 10
end = 40
res_json={}
res_json['videoid']=video

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
output_res = output + "/video_" + str(video) + "_output_cheby.json"
path_frame = output + "/video_" + str(video) + "_frame.jpg"
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
print("-------------------")
try:
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Asegurar que OpenCV procesa correctamente el cierre

    # Pequeña pausa para evitar bloqueos en iteraciones rápidas

except:
    pass
time.sleep(1)
