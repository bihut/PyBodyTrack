import cv2
import mediapipe as mp

class MediapipeProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose()

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        data = {}

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                data[self.mp_pose.PoseLandmark(idx).name] = (landmark.x, landmark.y, landmark.z, landmark.visibility)

        return data, frame
