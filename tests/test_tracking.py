import pytest

import cv2

from pybodytrack.processors.humanposeprocessor import HumanPoseProcessor


# Use MediaPipe, YOLO, or OpenPose as needed
processor = HumanPoseProcessor("mediapipeprocessor")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    df = processor.process_frame(frame)
    print(df)  # Output normalized pose DataFrame

    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# processor = HumanPoseProcessor("yolo_processor")
# processor = HumanPoseProcessor("openpose_processor")
