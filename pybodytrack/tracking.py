import cv2
import numpy as np
import mediapipe as mp
from typing import Callable, Optional
class Tracking:
    @staticmethod
    def show_camera_with_model(
        camera_index: int = 0,
        model_processor: Optional[Callable[[cv2.Mat], cv2.Mat]] = None
    ):
        """
        Displays the camera feed with an optional model overlay.

        Args:
            camera_index (int, optional): The index of the camera to use. Defaults to 0.
            model_processor (Callable[[cv2.Mat], cv2.Mat], optional):
                A function that takes a frame, processes it, and returns the modified frame.
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Process frame with the model if provided
                if model_processor:
                    frame = model_processor(frame)

                # Show frame
                cv2.imshow("Camera Feed", frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()



def track_motion(video_path: str, algorithm: str = "optical_flow") -> dict:
    """
    Track motion in a video using the specified algorithm.

    :param video_path: Path to the video file.
    :param algorithm: Algorithm to use for tracking ("optical_flow", "deep_learning", etc.).
    :return: Dictionary containing motion statistics.
    """
    # Implementación del tracking aquí
    motion_data = {"total_movement": 0, "frames_analyzed": 0}
    return motion_data