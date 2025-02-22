import cv2
from pybodytrack.pose_estimators.yolo_processor import YoloProcessor
from pybodytrack.pose_estimators.mediapipe_processor import MediaPipeProcessor
from pybodytrack.pose_estimators.camera_pose_tracker import CameraPoseTracker
custom_model_path = "/home/bihut/dev/Proyectos/pyBodyTrack/yolov8n-pose.pt"  # 🛠️ Cambia esto si tienes un modelo diferente

processor = YoloProcessor(model_path=custom_model_path)
# 📌 Selecciona el modelo: Mediapipe o YOLO
#processor = YoloProcessor()  # Cambia a YoloProcessor() si quieres usar YOLO
text= "YOLO" if isinstance(processor, YoloProcessor) else "MediaPipe"
# 📌 Inicializar tracker
tracker = CameraPoseTracker(processor)

# 📌 Capturar video
cap = cv2.VideoCapture(0)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    tracker.process_frame(frame)
    frame_count += 1

    # 📌 Mostrar la imagen procesada con el esqueleto
    cv2.imshow("Pose Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count > 100:
        break  # 📌 Salir después de 100 frames para pruebas

cap.release()
cv2.destroyAllWindows()

# 📌 Guardar CSV
tracker.save_to_csv("pose_data"+text+".csv")
