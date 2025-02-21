"""
pyBodyTrack - A Python package for motion quantification in videos.

Author: Angel Ruiz Zafra
License: MIT License
Version: 2025.2.1
Repository: https://github.com/bihut/pyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""

import numpy as np
import scipy.spatial.distance as dist
from scipy.signal import convolve2d
import cv2
class Methods:

    @staticmethod
    def euclidean_distance(dataframe):
        """
        Calculate the total Euclidean distance of body landmarks between consecutive frames.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :return: Total movement as the sum of Euclidean distances between consecutive frames.
        """
        total_movement = 0.0

        # Iterate through consecutive frames
        for i in range(len(dataframe) - 1):
            current_frame = dataframe.iloc[i]
            next_frame = dataframe.iloc[i + 1]

            # Compute Euclidean distance for each landmark
            for j in range(len(current_frame) - 1):
                try:
                    current_point = np.array([float(coord) for coord in current_frame[j].split()])
                    next_point = np.array([float(coord) for coord in next_frame[j].split()])

                    # Compute Euclidean distance
                    distance = np.linalg.norm(next_point - current_point)

                    if not np.isnan(distance):
                        total_movement += distance
                except (ValueError, IndexError):
                    continue  # Skip invalid data points

        return total_movement


    import numpy as np

    @staticmethod
    def manhattan_distance(dataframe):
        """
        Calculate the total Manhattan distance of body landmarks between consecutive frames.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :return: Total movement as the sum of Manhattan distances between consecutive frames.
        """
        total_movement = 0.0

        # Iterate through consecutive frames
        for i in range(len(dataframe) - 1):
            current_frame = dataframe.iloc[i]
            next_frame = dataframe.iloc[i + 1]

            # Compute Manhattan distance for each landmark
            for j in range(len(current_frame) - 1):
                try:
                    current_point = np.array([float(coord) for coord in current_frame[j].split()])
                    next_point = np.array([float(coord) for coord in next_frame[j].split()])

                    # Compute Manhattan distance
                    distance = np.sum(np.abs(next_point - current_point))

                    if not np.isnan(distance):
                        total_movement += distance
                except (ValueError, IndexError):
                    continue  # Skip invalid data points

        return total_movement

    @staticmethod
    def chebyshev_distance(dataframe):
        """
        Calculate the total Chebyshev distance of body landmarks between consecutive frames.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :return: Total movement as the sum of Chebyshev distances between consecutive frames.
        """
        total_movement = 0.0

        # Iterate through consecutive frames
        for i in range(len(dataframe) - 1):
            current_frame = dataframe.iloc[i]
            next_frame = dataframe.iloc[i + 1]

            # Compute Chebyshev distance for each landmark
            for j in range(len(current_frame) - 1):
                try:
                    current_point = np.array([float(coord) for coord in current_frame[j].split()])
                    next_point = np.array([float(coord) for coord in next_frame[j].split()])

                    # Compute Chebyshev distance
                    distance = np.max(np.abs(next_point - current_point))

                    if not np.isnan(distance):
                        total_movement += distance
                except (ValueError, IndexError):
                    continue  # Skip invalid data points

        return total_movement

    @staticmethod
    def minkowski_distance(dataframe, p):
        """
        Calculate the total Minkowski distance of body landmarks between consecutive frames.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :param p: Order of the Minkowski distance (e.g., p=1 for Manhattan, p=2 for Euclidean).
        :return: Total movement as the sum of Minkowski distances between consecutive frames.
        """
        total_movement = 0.0

        # Iterate through consecutive frames
        for i in range(len(dataframe) - 1):
            current_frame = dataframe.iloc[i]
            next_frame = dataframe.iloc[i + 1]

            # Compute Minkowski distance for each landmark
            for j in range(len(current_frame) - 1):
                try:
                    current_point = np.array([float(coord) for coord in current_frame[j].split()])
                    next_point = np.array([float(coord) for coord in next_frame[j].split()])

                    # Compute Minkowski distance
                    distance = np.power(np.sum(np.power(np.abs(next_point - current_point), p)), 1 / p)

                    if not np.isnan(distance):
                        total_movement += distance
                except (ValueError, IndexError, ZeroDivisionError):
                    continue  # Skip invalid data points

        return total_movement

    @staticmethod
    def mahalanobis_distance(dataframe):
        """
        Calculate the total Mahalanobis distance of body landmarks between consecutive frames.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :return: Total movement as the sum of Mahalanobis distances between consecutive frames.
        """
        total_movement = 0.0

        # Convert dataframe landmarks to numeric arrays
        data_array = []
        for i in range(len(dataframe)):
            row_points = []
            for j in range(len(dataframe.iloc[i]) - 1):  # Exclude timestamp if applicable
                try:
                    row_points.append([float(coord) for coord in dataframe.iloc[i, j].split()])
                except ValueError:
                    row_points.append(None)  # Handle missing or invalid values
            data_array.append(row_points)

        data_array = np.array(data_array, dtype=object)  # Keep as object for flexibility

        # Compute covariance matrix (needed for Mahalanobis distance)
        flattened_data = np.vstack([row for row in data_array if row is not None])
        covariance_matrix = np.cov(flattened_data.T)  # Compute covariance matrix across landmarks
        inv_cov_matrix = np.linalg.pinv(covariance_matrix)  # Compute pseudo-inverse in case of singularity

        # Iterate through consecutive frames
        for i in range(len(data_array) - 1):
            current_frame = data_array[i]
            next_frame = data_array[i + 1]

            # Compute Mahalanobis distance for each landmark
            for j in range(len(current_frame)):
                if current_frame[j] is None or next_frame[j] is None:
                    continue  # Skip invalid data points

                try:
                    current_point = np.array(current_frame[j])
                    next_point = np.array(next_frame[j])

                    # Compute Mahalanobis distance
                    distance = dist.mahalanobis(current_point, next_point, inv_cov_matrix)

                    if not np.isnan(distance):
                        total_movement += distance
                except (ValueError, np.linalg.LinAlgError):
                    continue  # Skip cases where distance calculation fails

        return total_movement


    @staticmethod
    def differential_acceleration(dataframe, fps):
        """
        Calculate the total movement based on differential acceleration of body landmarks.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :param fps: Frames per second of the video, used to compute velocity and acceleration.
        :return: Total movement as the sum of acceleration differences across all frames and landmarks.
        """
        total_movement = 0.0

        # Iterate through frames, excluding first and last
        for i in range(1, len(dataframe) - 1):
            prev_frame = dataframe.iloc[i - 1]
            current_frame = dataframe.iloc[i]
            next_frame = dataframe.iloc[i + 1]

            # Compute acceleration differences for each landmark
            for j in range(len(current_frame) - 1):  # Exclude timestamp if present
                try:
                    prev_point = np.array([float(coord) for coord in prev_frame[j].split()])
                    current_point = np.array([float(coord) for coord in current_frame[j].split()])
                    next_point = np.array([float(coord) for coord in next_frame[j].split()])

                    # Compute velocity for each dimension (x, y, z)
                    prev_velocity = (current_point - prev_point) * fps
                    next_velocity = (next_point - current_point) * fps

                    # Compute acceleration difference for each dimension
                    acceleration_diff = next_velocity - prev_velocity

                    # Sum absolute differences in acceleration across all dimensions
                    total_movement += np.sum(np.abs(acceleration_diff))

                except (ValueError, IndexError):
                    continue  # Skip invalid data points

        return total_movement

    @staticmethod
    def angular_displacement(dataframe):
        """
        Calculate the total angular displacement of body landmarks between consecutive frames.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :return: Total angular displacement summed over all frames and landmarks.
        """
        total_angular_movement = 0.0

        # Iterate through frames, excluding the first and last
        for i in range(1, len(dataframe) - 1):
            prev_frame = dataframe.iloc[i - 1]
            current_frame = dataframe.iloc[i]
            next_frame = dataframe.iloc[i + 1]

            # Compute angular displacement for each landmark
            for j in range(len(current_frame) - 1):  # Exclude timestamp if present
                try:
                    prev_vector = np.array([float(coord) for coord in prev_frame[j].split()])
                    current_vector = np.array([float(coord) for coord in current_frame[j].split()])
                    next_vector = np.array([float(coord) for coord in next_frame[j].split()])

                    # Compute cosine of the angle between vectors
                    dot_product = np.dot(prev_vector, next_vector)
                    norm_product = np.linalg.norm(prev_vector) * np.linalg.norm(next_vector)

                    if norm_product == 0:
                        continue  # Skip if one of the vectors is zero to avoid division by zero

                    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)  # Ensure valid range for arccos

                    # Compute angular displacement (in radians)
                    angular_displacement = np.arccos(cos_theta)

                    # Accumulate total angular displacement
                    total_angular_movement += angular_displacement

                except (ValueError, IndexError, FloatingPointError):
                    continue  # Skip invalid data points

        return total_angular_movement

    @staticmethod
    def lucas_kanade_optical_flow(dataframe, window_size):
        """
        Compute total movement using the Lucas-Kanade optical flow method.

        :param dataframe: Pandas DataFrame containing 33 body landmarks per frame along with timestamps.
                          Each landmark should be stored as a string with space-separated coordinates.
        :param window_size: Window size for smoothing the motion estimation.
        :return: Total movement as the sum of velocity magnitudes across all frames and landmarks.
        """
        total_movement = 0.0

        # Iterate through frames, excluding the first and last
        for i in range(1, len(dataframe) - 1):
            prev_frame = dataframe.iloc[i - 1]
            current_frame = dataframe.iloc[i]
            next_frame = dataframe.iloc[i + 1]

            # Compute Lucas-Kanade optical flow for each landmark
            for j in range(len(current_frame) - 1):  # Exclude timestamp if present
                try:
                    prev_point = np.array([float(coord) for coord in prev_frame[j].split()])
                    current_point = np.array([float(coord) for coord in current_frame[j].split()])
                    next_point = np.array([float(coord) for coord in next_frame[j].split()])

                    # Apply Lucas-Kanade method
                    vx, vy, vz = Methods._lucas_kanade(prev_point, current_point, next_point, window_size)

                    # Compute velocity magnitude
                    velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

                    # Accumulate total movement
                    if not np.isnan(velocity_magnitude):
                        total_movement += velocity_magnitude

                except (ValueError, IndexError):
                    continue  # Skip invalid data points

        return total_movement
    @staticmethod
    def _lucas_kanade(prev, current, next, window_size):
        """
        Estimate optical flow velocity using the Lucas-Kanade method.

        :param prev: Landmark positions in the previous frame.
        :param current: Landmark positions in the current frame.
        :param next: Landmark positions in the next frame.
        :param window_size: Window size for the smoothing operation.
        :return: Estimated velocities (vx, vy, vz).
        """
        # Convert landmark points to 2D arrays
        prev = np.array(prev).reshape(-1, 3)
        current = np.array(current).reshape(-1, 3)
        next = np.array(next).reshape(-1, 3)

        # Apply smoothing filter (mean filter) to reduce noise
        kernel = np.ones((window_size, window_size)) / window_size**2
        prev_smoothed = np.array([convolve2d(prev[:, i].reshape(1, -1), kernel, mode='same') for i in range(3)])
        current_smoothed = np.array([convolve2d(current[:, i].reshape(1, -1), kernel, mode='same') for i in range(3)])
        next_smoothed = np.array([convolve2d(next[:, i].reshape(1, -1), kernel, mode='same') for i in range(3)])

        # Compute positional differences between frames
        dx_prev = current_smoothed[0] - prev_smoothed[0]
        dy_prev = current_smoothed[1] - prev_smoothed[1]
        dz_prev = current_smoothed[2] - prev_smoothed[2]

        dx_next = next_smoothed[0] - current_smoothed[0]
        dy_next = next_smoothed[1] - current_smoothed[1]
        dz_next = next_smoothed[2] - current_smoothed[2]

        # Compute average velocity
        vx = (dx_prev + dx_next) / 2
        vy = (dy_prev + dy_next) / 2
        vz = (dz_prev + dz_next) / 2

        return vx, vy, vz
