import cv2
import numpy as np

class YoloProcessor:
    def __init__(self, model_cfg="models/yolo/yolov4.cfg", model_weights="models/yolo/yolov4.weights"):
        self.net = cv2.dnn.readNet(model_weights, model_cfg)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def process(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward(self.output_layers)

        data = {}
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    data[f"YOLO_POINT_{class_id}"] = (x, y, 0, confidence)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return data, frame
