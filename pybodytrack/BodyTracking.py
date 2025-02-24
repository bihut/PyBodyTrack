import cv2
import time
import threading

from pybodytrack.enums.PoseProcessor import PoseProcessor
from pybodytrack.enums.VideoMode import VideoMode
from pybodytrack.pose_estimators.camera_pose_tracker import CameraPoseTracker
from pybodytrack.pose_estimators.mediapipe_processor import MediaPipeProcessor
from pybodytrack.pose_estimators.yolo_processor import YoloProcessor


class BodyTracking:
    def __init__(self, processor="mediapipe", mode=VideoMode.CAMERA, path_video="",custom_model_path="",selected_landmarks=None):
        """
        Initializes the BodyTracking object.

        Parameters:
            processor: An instance of the processor (e.g., YoloProcessor or MediaPipe).
            mode (int): 0 for camera, 1 for video file.
            path_video (str): The path to the video file if mode is 1.
        """
        if processor == PoseProcessor.MEDIAPIPE:
            self.processor = MediaPipeProcessor()
        elif processor == PoseProcessor.YOLO:
            self.processor = YoloProcessor(model_path=custom_model_path)
        #self.processor = processor
        self.tracker = CameraPoseTracker(self.processor,selected_landmarks=selected_landmarks)
        self.mode = mode
        if mode == VideoMode.VIDEO:
            self.path_video = path_video

        # Determine the video source and FPS based on mode
        if self.mode == VideoMode.CAMERA:
            self.cap = cv2.VideoCapture(0)
            self.fps = 30  # Default FPS for camera
        elif self.mode == VideoMode.VIDEO:
            self.cap = cv2.VideoCapture(self.path_video)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps == 0:
                self.fps = 30  # Default if unable to determine FPS
        else:
            raise ValueError("Invalid mode selected. Use 0 for camera or 1 for video file.")

        self.frame_interval = 1.0 / self.fps

        # Text identifier for saving CSV (YOLO vs. MediaPipe)
        self.text = "YOLO" if isinstance(self.processor, YoloProcessor) else "MediaPipe"

        # Shared variables and lock for frame processing
        self.latest_frame_lock = threading.Lock()
        self.frame_to_process = None  # Latest frame available for processing
        self.latest_processed_frame = None  # Latest processed frame (with skeleton)
        self.stop_processing = False

        # Processing thread
        self.processing_thread = threading.Thread(target=self._processing_thread_func)

    def _processing_thread_func(self):
        """
        Thread function to continuously process frames.
        It always processes the latest frame available.
        """
        while not self.stop_processing:
            with self.latest_frame_lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process.copy()
                else:
                    frame = None
            if frame is not None:
                # Process the frame (this should draw the skeleton on the frame)
                self.tracker.process_frame(frame)
                # Store the processed frame for display
                with self.latest_frame_lock:
                    self.latest_processed_frame = frame
            else:
                time.sleep(0.001)  # small delay to avoid busy waiting

    def start(self):
        """
        Starts the processing thread and the main loop for reading, processing,
        and displaying the video frames.
        """
        self.processing_thread.start()
        while self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            # Update the frame for processing
            with self.latest_frame_lock:
                self.frame_to_process = frame.copy()
                # Use the processed frame if available; otherwise, show the raw frame
                if self.latest_processed_frame is not None:
                    display_frame = self.latest_processed_frame.copy()
                else:
                    display_frame = frame.copy()

            cv2.imshow("Pose Tracking", display_frame)
            elapsed_time = time.time() - start_time
            remaining_time = self.frame_interval - elapsed_time
            if remaining_time > 0:
                time.sleep(remaining_time)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self):
        """
        Stops the processing thread, releases the video source,
        and closes all OpenCV windows.
        """
        self.stop_processing = True
        self.processing_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

    def getData(self):
        return self.tracker.get_dataframe()

    def save_csv(self, filename=None):
        """
        Saves the tracking data to a CSV file.

        Parameters:
            filename (str): Optional filename for the CSV. If None, a default
                            name based on the processor type is used.
        """
        if filename is None:
            filename = "pose_data" + self.text + "_"+str(time.time())+".csv"
        self.tracker.save_to_csv(filename)

