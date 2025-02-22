import cv2
import numpy as np

class OpenposeProcessor:
    def __init__(self, model_folder="models/openpose/"):
        self.net = cv2.dnn.readNetFromCaffe(
            model_folder + "pose_deploy_linevec.prototxt",
            model_folder + "pose_iter_440000.caffemodel"
        )
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1

    def process(self, frame):
        frameHeight, frameWidth = frame.shape[:2]
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.inWidth, self.inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        output = self.net.forward()

        data = {}
        for i in range(output.shape[1]):
            heatMap = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = int((frameWidth * point[0]) / output.shape[3])
            y = int((frameHeight * point[1]) / output.shape[2])
            if conf > self.threshold:
                data[f"KEYPOINT_{i}"] = (x, y, 0, conf)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        return data, frame
