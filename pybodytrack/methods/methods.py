"""
pyBodyTrack - A Python package for motion quantification in videos.

Author: Angel Ruiz Zafra
License: MIT License
Version: 2025.2.1
Repository: https://github.com/bihut/pyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
import cv2
class Methods:
    @staticmethod
    def _kalman_filter(data, Q=1e-5, R=0.1):
        """
        A simple 1D Kalman filter for smoothing a time series.

        :param data: 1D numpy array of measurements.
        :param Q: Process variance.
        :param R: Measurement variance.
        :return: A 1D numpy array of filtered values.
        """
        n = len(data)
        xhat = np.zeros(n)  # a posteri estimate of x
        P = np.zeros(n)  # a posteri error estimate
        xhatminus = np.zeros(n)  # a priori estimate of x
        Pminus = np.zeros(n)  # a priori error estimate
        K = np.zeros(n)  # gain or blending factor

        # Initialize with the first measurement.
        xhat[0] = data[0]
        P[0] = 1.0

        for k in range(1, n):
            # Time update (prediction)
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q

            # Measurement update (correction)
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        return xhat

    @staticmethod
    def euclidean_distance(df, filter=True, distance_threshold=2.0, Q=1e-5, R=0.1):
        """
        Calculate the total Euclidean distance of body landmarks between consecutive frames.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, z)
        and small movements below `distance_threshold` are ignored.
        When False, the pure Euclidean distance is computed from raw data.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        :param df: Pandas DataFrame with the landmark data.
        :param filter: Boolean flag to apply Kalman filtering and thresholding if True.
        :param distance_threshold: Minimum Euclidean distance change to be considered valid movement when filtering.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of Euclidean distances between consecutive frames.
        """
        total_movement = 0.0
        columns = list(df.columns)
        landmark_columns = columns[1:]  # Exclude timestamp
        num_landmarks = len(landmark_columns) // 4

        if filter:
            # Create a filtered copy of the DataFrame.
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)

            # Compute Euclidean distance using the filtered data.
            for i in range(len(df_filtered) - 1):
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df_filtered.iloc[i + 1][col_x] - df_filtered.iloc[i][col_x]
                    dy = df_filtered.iloc[i + 1][col_y] - df_filtered.iloc[i][col_y]
                    dz = df_filtered.iloc[i + 1][col_z] - df_filtered.iloc[i][col_z]
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if distance >= distance_threshold:
                        total_movement += distance
        else:
            # Compute pure Euclidean distance using raw data.
            for i in range(len(df) - 1):
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df.iloc[i + 1][col_x] - df.iloc[i][col_x]
                    dy = df.iloc[i + 1][col_y] - df.iloc[i][col_y]
                    dz = df.iloc[i + 1][col_z] - df.iloc[i][col_z]
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    total_movement += distance

        return total_movement
    @staticmethod
    def euclidean_distanceborrar(df, filter=True, distance_threshold=2.0, Q=1e-5, R=0.1):
        """
        Calculates the total Euclidean movement (sum of distances between consecutive frames)
        and returns a DataFrame with the movement details for each pair of frames along with their timestamps.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, z) and
        movements smaller than `distance_threshold` are ignored.
        When False, the pure Euclidean distance is computed using raw data.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        :param df: Pandas DataFrame containing the landmark data.
        :param filter: Boolean flag to apply Kalman filtering and distance thresholding.
        :param distance_threshold: Minimum Euclidean distance change to be considered a valid movement when filtering.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: A tuple (total_movement, movement_df) where:
                 - total_movement is the sum of the Euclidean distances between consecutive frames.
                 - movement_df is a DataFrame with columns:
                       'timestamp_start', 'timestamp_end', and 'movement',
                     representing the movement (sum of distances for each landmark) between each pair of frames.
        """
        total_movement = 0.0
        movements = []  # List to store movement details for each pair of frames

        columns = list(df.columns)
        landmark_columns = columns[1:]  # Exclude the 'timestamp' column
        num_landmarks = len(landmark_columns) // 4

        if filter:
            # Create a filtered copy of the DataFrame
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]
                # Apply the Kalman filter to the x, y, and z coordinates for each landmark
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)

            # Compute the Euclidean distance using the filtered data
            for i in range(len(df_filtered) - 1):
                frame_movement = 0.0
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df_filtered.iloc[i + 1][col_x] - df_filtered.iloc[i][col_x]
                    dy = df_filtered.iloc[i + 1][col_y] - df_filtered.iloc[i][col_y]
                    dz = df_filtered.iloc[i + 1][col_z] - df_filtered.iloc[i][col_z]
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if distance >= distance_threshold:
                        frame_movement += distance
                total_movement += frame_movement
                movements.append({
                    'timestamp_start': df_filtered.iloc[i]['timestamp'],
                    'timestamp_end': df_filtered.iloc[i + 1]['timestamp'],
                    'movement': frame_movement
                })
        else:
            # Compute the pure Euclidean distance using the raw data (no filtering)
            for i in range(len(df) - 1):
                frame_movement = 0.0
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df.iloc[i + 1][col_x] - df.iloc[i][col_x]
                    dy = df.iloc[i + 1][col_y] - df.iloc[i][col_y]
                    dz = df.iloc[i + 1][col_z] - df.iloc[i][col_z]
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    frame_movement += distance
                total_movement += frame_movement
                movements.append({
                    'timestamp_start': df.iloc[i]['timestamp'],
                    'timestamp_end': df.iloc[i + 1]['timestamp'],
                    'movement': frame_movement
                })

        # Convert the list of movements into a DataFrame
        movement_df = pd.DataFrame(movements)
        return total_movement, movement_df

    @staticmethod
    def euclidean_distancebackup(df, filter=True, distance_threshold=2.0, Q=1e-5, R=0.1):
        """
        Calculate the total Euclidean distance of body landmarks between consecutive frames.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, z)
        and small movements below `distance_threshold` are ignored.
        When False, the pure Euclidean distance is computed from raw data.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        :param df: Pandas DataFrame with the landmark data.
        :param filter: Boolean flag to apply Kalman filtering and thresholding if True.
        :param distance_threshold: Minimum Euclidean distance change to be considered valid movement when filtering.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of Euclidean distances between consecutive frames.
        """
        total_movement = 0.0
        columns = list(df.columns)
        landmark_columns = columns[1:]  # Exclude timestamp
        num_landmarks = len(landmark_columns) // 4

        if filter:
            # Create a filtered copy of the DataFrame.
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)

            # Compute Euclidean distance using the filtered data.
            for i in range(len(df_filtered) - 1):
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df_filtered.iloc[i + 1][col_x] - df_filtered.iloc[i][col_x]
                    dy = df_filtered.iloc[i + 1][col_y] - df_filtered.iloc[i][col_y]
                    dz = df_filtered.iloc[i + 1][col_z] - df_filtered.iloc[i][col_z]
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if distance >= distance_threshold:
                        total_movement += distance
        else:
            # Compute pure Euclidean distance using raw data.
            for i in range(len(df) - 1):
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df.iloc[i + 1][col_x] - df.iloc[i][col_x]
                    dy = df.iloc[i + 1][col_y] - df.iloc[i][col_y]
                    dz = df.iloc[i + 1][col_z] - df.iloc[i][col_z]
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    total_movement += distance

        return total_movement

    @staticmethod
    def manhattan_distance(df, filter=True, distance_threshold=2.0, Q=1e-5, R=0.1):
        """
        Calculate the total Manhattan distance of body landmarks between consecutive frames.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, z)
        and movements below `distance_threshold` are ignored.
        When False, the pure Manhattan distance is computed from raw data.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        :param df: Pandas DataFrame with the landmark data.
        :param filter: Boolean flag to apply Kalman filtering and thresholding if True.
        :param distance_threshold: Minimum Manhattan distance change to be considered valid movement when filtering.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of Manhattan distances between consecutive frames.
        """
        total_movement = 0.0
        columns = list(df.columns)
        landmark_columns = columns[1:]  # Exclude timestamp
        num_landmarks = len(landmark_columns) // 4

        if filter:
            # Create a filtered copy of the DataFrame.
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)

            # Compute Manhattan distance using the filtered data.
            for i in range(len(df_filtered) - 1):
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df_filtered.iloc[i + 1][col_x] - df_filtered.iloc[i][col_x]
                    dy = df_filtered.iloc[i + 1][col_y] - df_filtered.iloc[i][col_y]
                    dz = df_filtered.iloc[i + 1][col_z] - df_filtered.iloc[i][col_z]
                    distance = abs(dx) + abs(dy) + abs(dz)
                    if distance >= distance_threshold:
                        total_movement += distance
        else:
            # Compute pure Manhattan distance using raw data.
            for i in range(len(df) - 1):
                for lm in range(num_landmarks):
                    base = lm * 4
                    col_x = landmark_columns[base]
                    col_y = landmark_columns[base + 1]
                    col_z = landmark_columns[base + 2]
                    dx = df.iloc[i + 1][col_x] - df.iloc[i][col_x]
                    dy = df.iloc[i + 1][col_y] - df.iloc[i][col_y]
                    dz = df.iloc[i + 1][col_z] - df.iloc[i][col_z]
                    distance = abs(dx) + abs(dy) + abs(dz)
                    total_movement += distance

        return total_movement

    @staticmethod
    def chebyshev_distance(df, filter=True, distance_threshold=0.0, Q=1e-5, R=0.1):
        """
        Calculate the total Chebyshev distance of body landmarks between consecutive frames.

        This method supports two modes:
          - If filter is True, it applies a Kalman filter to each coordinate and then computes
            the Chebyshev distance, optionally ignoring small differences below distance_threshold.
          - If filter is False, it computes the pure Chebyshev distance using the raw data.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, four columns in order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        :param df: Pandas DataFrame with the landmark data.
        :param filter: Boolean flag to apply filtering (Kalman filter and threshold) if True.
        :param distance_threshold: Minimum Chebyshev distance to be considered valid movement when filtering.
                                   (Set to 0.0 to disable thresholding.)
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of Chebyshev distances between consecutive frames.
        """
        total_movement = 0.0
        columns = list(df.columns)
        landmark_columns = columns[1:]  # Exclude 'timestamp'
        num_landmarks = len(landmark_columns) // 4

        if filter:
            # Create a copy of the DataFrame to store the filtered values.
            df_filtered = df.copy()
            # Apply the Kalman filter to each coordinate (x, y, z) for each landmark.
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]

                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)

            # Compute Chebyshev distance on the filtered data.
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]

                dx = df_filtered[col_x].diff().fillna(0)
                dy = df_filtered[col_y].diff().fillna(0)
                dz = df_filtered[col_z].diff().fillna(0)

                # Chebyshev distance: maximum of the absolute differences (x, y, z)
                distances = pd.concat([dx.abs(), dy.abs(), dz.abs()], axis=1).max(axis=1)
                if distance_threshold > 0.0:
                    distances = distances.where(distances >= distance_threshold, 0)
                total_movement += distances.sum()
        else:
            # Compute pure Chebyshev distance on the raw data.
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]

                dx = df[col_x].diff().fillna(0)
                dy = df[col_y].diff().fillna(0)
                dz = df[col_z].diff().fillna(0)

                distances = pd.concat([dx.abs(), dy.abs(), dz.abs()], axis=1).max(axis=1)
                total_movement += distances.sum()

        return total_movement

    @staticmethod
    def minkowski_distance(df, p=2, filter=True, distance_threshold=2.0, Q=1e-5, R=0.1):
        """
        Calculate the total Minkowski distance of body landmarks between consecutive frames,
        optionally applying a Kalman filter and a threshold to remove small noise-induced movements.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order:
            <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        When filter is True, a Kalman filter is applied to each coordinate (x, y, and z) and any
        distance value below the distance_threshold is ignored. When filter is False, the pure Minkowski
        distance is computed using the raw data.

        The Minkowski distance is computed using the formula:
             d = (|dx|^p + |dy|^p + |dz|^p)^(1/p)

        :param df: Pandas DataFrame with the landmark data.
        :param p: Order of the Minkowski distance (e.g., p=1 for Manhattan, p=2 for Euclidean).
        :param filter: Boolean flag to apply Kalman filtering and thresholding if True.
        :param distance_threshold: Minimum Minkowski distance change to be considered valid movement.
                                   (Measured in the same units as the landmark coordinates.)
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of Minkowski distances between consecutive frames.
        """
        total_movement = 0.0

        # Exclude the 'timestamp' column.
        columns = list(df.columns)
        landmark_columns = columns[1:]
        # Each landmark has 4 columns: x, y, z, confidence.
        num_landmarks = len(landmark_columns) // 4

        if filter:
            # Create a filtered copy of the DataFrame.
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)

            # Compute Minkowski distance on the filtered data.
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]

                # Calculate differences between consecutive frames.
                dx = df_filtered[col_x].diff().fillna(0)
                dy = df_filtered[col_y].diff().fillna(0)
                dz = df_filtered[col_z].diff().fillna(0)

                # Compute Minkowski distance for each frame.
                distances = np.power(np.abs(dx) ** p + np.abs(dy) ** p + np.abs(dz) ** p, 1.0 / p)
                # Apply threshold: ignore distances below the threshold.
                distances = distances.where(distances >= distance_threshold, 0)
                total_movement += distances.sum()
        else:
            # Compute pure Minkowski distance using raw data.
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = landmark_columns[base]
                col_y = landmark_columns[base + 1]
                col_z = landmark_columns[base + 2]

                dx = df[col_x].diff().fillna(0)
                dy = df[col_y].diff().fillna(0)
                dz = df[col_z].diff().fillna(0)

                distances = np.power(np.abs(dx) ** p + np.abs(dy) ** p + np.abs(dz) ** p, 1.0 / p)
                total_movement += distances.sum()

        return total_movement

    @staticmethod
    def mahalanobis_distance(df, filter=True, distance_threshold=0.0, Q=1e-5, R=0.1):
        """
        Calculate the total Mahalanobis distance of body landmarks between consecutive frames,
        with an option to apply Kalman filtering to reduce noise.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, and z) for each landmark.
        An optional distance_threshold can be used to ignore small changes (set to 0.0 to disable thresholding).

        The method builds an array of shape (n_frames, num_landmarks, 3) using the x, y, and z values,
        computes the global covariance matrix (across all landmarks and frames), and then sums the Mahalanobis
        distances between consecutive frames for each landmark.

        :param df: Pandas DataFrame with the landmark data.
        :param filter: Boolean flag indicating whether to apply Kalman filtering.
        :param distance_threshold: Minimum Mahalanobis distance to be considered as valid movement.
                                   (Measured in the same units as the landmark coordinates.)
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of Mahalanobis distances between consecutive frames.
        """


        total_movement = 0.0
        n_frames = len(df)
        if n_frames < 2:
            return 0.0

        # Exclude the timestamp column.
        n_total_cols = len(df.columns) - 1
        # Each landmark has 4 columns (x, y, z, confidence)
        num_landmarks = n_total_cols // 4

        # Build an array of shape (n_frames, num_landmarks, 3) using x, y, and z values.
        points = np.empty((n_frames, num_landmarks, 3), dtype=float)
        for lm in range(num_landmarks):
            col_x = df.columns[1 + lm * 4]  # Skip 'timestamp'
            col_y = df.columns[1 + lm * 4 + 1]
            col_z = df.columns[1 + lm * 4 + 2]
            points[:, lm, 0] = df[col_x].values
            points[:, lm, 1] = df[col_y].values
            points[:, lm, 2] = df[col_z].values

        if filter:
            # Apply the Kalman filter to each coordinate (x, y, and z) for every landmark.
            for lm in range(num_landmarks):
                for coord in range(3):
                    points[:, lm, coord] = Methods._kalman_filter(points[:, lm, coord], Q, R)

        # Flatten the data to shape (n_frames * num_landmarks, 3) to compute the covariance matrix.
        flattened = points.reshape(-1, 3)
        covariance_matrix = np.cov(flattened.T)
        inv_cov_matrix = np.linalg.pinv(covariance_matrix)  # Use pseudo-inverse in case the matrix is singular

        # Iterate through consecutive frames and compute Mahalanobis distance for each landmark.
        for i in range(n_frames - 1):
            for lm in range(num_landmarks):
                current_point = points[i, lm, :]
                next_point = points[i + 1, lm, :]
                distance = mahalanobis(current_point, next_point, inv_cov_matrix)
                if not np.isnan(distance):
                    if distance_threshold > 0:
                        if distance >= distance_threshold:
                            total_movement += distance
                    else:
                        total_movement += distance

        return total_movement

    @staticmethod
    def differential_acceleration(df, fps=30, filter=True, Q=1e-5, R=0.1):
        """
        Calculate the total movement based on differential acceleration of body landmarks between consecutive frames.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, and z) time series to smooth
        out noise before computing velocities and acceleration differences.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order:
            <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        The method constructs an array of shape (n_frames, num_landmarks, 3) using the x, y, and z values,
        computes the velocities between frames (multiplied by fps), then computes the difference in velocity
        (acceleration) and sums the absolute differences across all frames and landmarks.

        :param df: Pandas DataFrame with the landmark data.
        :param fps: Frames per second of the video.
        :param filter: Boolean flag to apply Kalman filtering to the coordinate data.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of acceleration differences across all frames and landmarks.
        """
        import numpy as np

        total_movement = 0.0
        n_frames = len(df)
        if n_frames < 3:
            return 0.0  # Need at least three frames to compute differential acceleration.

        # Exclude the 'timestamp' column.
        n_total_cols = len(df.columns) - 1
        # Each landmark has 4 columns (x, y, z, confidence)
        num_landmarks = n_total_cols // 4

        # Optionally apply Kalman filter to each coordinate column.
        if filter:
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = df.columns[1 + base]
                col_y = df.columns[1 + base + 1]
                col_z = df.columns[1 + base + 2]
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)
        else:
            df_filtered = df

        # Build an array of points of shape (n_frames, num_landmarks, 3) using the (filtered) x, y, z values.
        points = np.empty((n_frames, num_landmarks, 3), dtype=float)
        for lm in range(num_landmarks):
            col_x = df_filtered.columns[1 + lm * 4]  # Skip 'timestamp'
            col_y = df_filtered.columns[1 + lm * 4 + 1]
            col_z = df_filtered.columns[1 + lm * 4 + 2]
            points[:, lm, 0] = df_filtered[col_x].values
            points[:, lm, 1] = df_filtered[col_y].values
            points[:, lm, 2] = df_filtered[col_z].values

        # For each intermediate frame, compute the differential acceleration for each landmark.
        for i in range(1, n_frames - 1):
            for lm in range(num_landmarks):
                prev_point = points[i - 1, lm, :]
                current_point = points[i, lm, :]
                next_point = points[i + 1, lm, :]

                # Compute velocities (difference in position multiplied by fps).
                prev_velocity = (current_point - prev_point) * fps
                next_velocity = (next_point - current_point) * fps

                # Compute acceleration difference (change in velocity).
                acceleration_diff = next_velocity - prev_velocity

                # Accumulate the sum of absolute acceleration differences over x, y, and z.
                total_movement += np.sum(np.abs(acceleration_diff))

        return total_movement

    @staticmethod
    def angular_displacement(df, filter=True, Q=1e-5, R=0.1):
        """
        Calculate the total angular displacement of body landmarks between consecutive frames.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, z) time series
        to smooth out noise before computing the angular displacement.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        For each landmark and for each frame (except the first and last), the function computes the angle
        (in radians) between the landmark's position in the previous frame and its position in the next frame.
        The total angular displacement is the sum of these angles over all frames and landmarks.

        :param df: Pandas DataFrame with the landmark data.
        :param filter: Boolean flag to apply Kalman filtering if True.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total angular displacement (in radians).
        """
        import numpy as np

        total_angular_movement = 0.0
        n_frames = len(df)
        if n_frames < 3:
            return 0.0  # At least three frames are needed to compute angular displacement.

        # Exclude the 'timestamp' column.
        n_total_cols = len(df.columns) - 1
        # Each landmark has 4 columns: x, y, z, confidence.
        num_landmarks = n_total_cols // 4

        # If filtering is enabled, create a filtered copy of the DataFrame.
        if filter:
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = df.columns[1 + base]
                col_y = df.columns[1 + base + 1]
                col_z = df.columns[1 + base + 2]
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)
        else:
            df_filtered = df

        # Iterate over intermediate frames (from the second to the second-last).
        for i in range(1, n_frames - 1):
            for lm in range(num_landmarks):
                col_x = df_filtered.columns[1 + lm * 4]
                col_y = df_filtered.columns[1 + lm * 4 + 1]
                col_z = df_filtered.columns[1 + lm * 4 + 2]

                # Extract vectors (x, y, z) from the previous and next frames.
                prev_vector = np.array([
                    df_filtered.iloc[i - 1][col_x],
                    df_filtered.iloc[i - 1][col_y],
                    df_filtered.iloc[i - 1][col_z]
                ])
                next_vector = np.array([
                    df_filtered.iloc[i + 1][col_x],
                    df_filtered.iloc[i + 1][col_y],
                    df_filtered.iloc[i + 1][col_z]
                ])

                # Avoid division by zero if any vector is zero.
                norm_prev = np.linalg.norm(prev_vector)
                norm_next = np.linalg.norm(next_vector)
                if norm_prev == 0 or norm_next == 0:
                    continue

                # Calculate the dot product and the cosine of the angle.
                dot_product = np.dot(prev_vector, next_vector)
                cos_theta = np.clip(dot_product / (norm_prev * norm_next), -1.0, 1.0)

                # Calculate the angle (in radians) and add it to the total.
                angle = np.arccos(cos_theta)
                total_angular_movement += angle

        return total_angular_movement

    @staticmethod
    def lucas_kanade_optical_flow(df, window_size, filter=True, Q=1e-5, R=0.1):
        """
        Compute total movement using the Lucas-Kanade optical flow method.

        When `filter` is True, a Kalman filter is applied to each coordinate (x, y, z)
        to smooth out noise before computing the velocity, and then the velocity magnitude
        is computed using the simplified Lucas-Kanade method. When False, the raw data is used.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - For each landmark, there are four columns in order:
            <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        The method iterates over frames (excluding the first and last) and for each landmark computes:
            velocity = (next_point - prev_point) / 2
        Then the magnitude of the velocity is added to the total movement.

        :param df: Pandas DataFrame with the landmark data.
        :param window_size: Window size for smoothing (not used in this simplified version).
        :param filter: Boolean flag to apply Kalman filtering if True.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of velocity magnitudes across all frames and landmarks.
        """
        import numpy as np

        total_movement = 0.0
        n_frames = len(df)
        if n_frames < 3:
            return 0.0

        # Exclude the 'timestamp' column.
        columns = list(df.columns)
        landmark_columns = columns[1:]
        num_landmarks = len(landmark_columns) // 4

        # Apply Kalman filtering if requested.
        if filter:
            df_filtered = df.copy()
            for lm in range(num_landmarks):
                base = lm * 4
                col_x = df.columns[1 + base]
                col_y = df.columns[1 + base + 1]
                col_z = df.columns[1 + base + 2]
                df_filtered[col_x] = Methods._kalman_filter(df[col_x].values, Q, R)
                df_filtered[col_y] = Methods._kalman_filter(df[col_y].values, Q, R)
                df_filtered[col_z] = Methods._kalman_filter(df[col_z].values, Q, R)
        else:
            df_filtered = df

        # Iterate over frames (from second to second-last).
        for i in range(1, n_frames - 1):
            for lm in range(num_landmarks):
                col_x = df_filtered.columns[1 + lm * 4]
                col_y = df_filtered.columns[1 + lm * 4 + 1]
                col_z = df_filtered.columns[1 + lm * 4 + 2]
                try:
                    prev_point = np.array([
                        df_filtered.iloc[i - 1][col_x],
                        df_filtered.iloc[i - 1][col_y],
                        df_filtered.iloc[i - 1][col_z]
                    ])
                    current_point = np.array([
                        df_filtered.iloc[i][col_x],
                        df_filtered.iloc[i][col_y],
                        df_filtered.iloc[i][col_z]
                    ])
                    next_point = np.array([
                        df_filtered.iloc[i + 1][col_x],
                        df_filtered.iloc[i + 1][col_y],
                        df_filtered.iloc[i + 1][col_z]
                    ])

                    # Apply the simplified Lucas-Kanade method:
                    # velocity = (next_point - prev_point) / 2
                    vx, vy, vz = Methods._lucas_kanade(prev_point, current_point, next_point, window_size)
                    velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                    if not np.isnan(velocity_magnitude):
                        total_movement += velocity_magnitude
                except (ValueError, IndexError):
                    continue

        return total_movement

    @staticmethod
    def _lucas_kanade(prev, current, next, window_size):
        """
        Estimate optical flow velocity using the Lucas-Kanade method adapted for a single landmark point.

        Since for each landmark we only have one point per frame, the velocity is estimated as:
             velocity = (next - prev) / 2

        The window_size parameter is included for compatibility but not used.

        :param prev: Landmark position in the previous frame (array with [x, y, z]).
        :param current: Landmark position in the current frame (not used here).
        :param next: Landmark position in the next frame (array with [x, y, z]).
        :param window_size: Window size for smoothing (unused in this simple implementation).
        :return: Estimated velocities (vx, vy, vz).
        """
        vx = (next[0] - prev[0]) / 2.0
        vy = (next[1] - prev[1]) / 2.0
        vz = (next[2] - prev[2]) / 2.0
        return vx, vy, vz

    @staticmethod
    def farneback_optical_flow(df, image_size=(100, 100), filter=True, Q=1e-5, R=0.1):
        """
        Compute total movement using the Farnebäck optical flow method.

        When `filter` is True, a Kalman filter is applied to each numeric landmark column
        (x, y, z, and confidence) to smooth out noise before converting the landmarks
        into a synthetic image. When False, raw data is used.

        Assumes the DataFrame has the following structure:
          - The first column is 'timestamp'.
          - The remaining columns are numeric and grouped in blocks of 4 for each landmark,
            in the order: <landmark>_x, <landmark>_y, <landmark>_z, <landmark>_confidence.

        For each consecutive pair of frames (rows), the function:
          1. Excludes the timestamp column.
          2. Converts the numeric landmark data into a synthetic image.
          3. Converts the image to grayscale.
          4. Computes the optical flow between the two images using the Farnebäck method.
          5. Accumulates the total movement as the sum of the flow magnitude.

        :param df: Pandas DataFrame containing landmark data.
        :param image_size: Tuple (width, height) for the synthetic image.
        :param filter: Boolean flag; if True, apply Kalman filtering to smooth the landmark data.
        :param Q: Process variance for the Kalman filter.
        :param R: Measurement variance for the Kalman filter.
        :return: Total movement as the sum of optical flow magnitudes over all frame pairs.
        """
        total_movement = 0.0

        # If filtering is enabled, create a filtered copy of the DataFrame.
        if filter:
            df_filtered = df.copy()
            columns = list(df.columns)
            # Exclude the 'timestamp' (first) column.
            landmark_columns = columns[1:]
            # Apply the Kalman filter to each numeric column.
            for col in landmark_columns:
                df_filtered[col] = Methods._kalman_filter(df[col].values, Q, R)
        else:
            df_filtered = df

        # Iterate over consecutive frames (rows).
        for i in range(1, len(df_filtered)):
            try:
                prev_row = df_filtered.iloc[i - 1]
                current_row = df_filtered.iloc[i]

                # Exclude the timestamp column (assumed to be the first column).
                prev_landmarks = prev_row.iloc[1:]
                current_landmarks = current_row.iloc[1:]

                # Convert the numeric landmark data into synthetic images.
                prev_img = Methods._convert_landmarks_to_image(prev_landmarks, image_size)
                current_img = Methods._convert_landmarks_to_image(current_landmarks, image_size)

                # Convert images to grayscale.
                prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

                # Compute optical flow using the Farnebäck method.
                flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

                # Compute flow magnitude (Euclidean norm per pixel).
                flow_magnitude = np.linalg.norm(flow, axis=-1)
                total_movement += np.sum(flow_magnitude)
            except Exception:
                continue

        return total_movement

    @staticmethod
    def _convert_landmarks_to_image(landmarks, image_size=(100, 100)):
        """
        Private function to convert a row of numeric landmark data into a synthetic image.

        Assumes that the input 'landmarks' (a Pandas Series) contains numeric values in groups of 4:
          [landmark_x, landmark_y, landmark_z, landmark_confidence, ...].
        This function creates a blank image and draws a small circle for each landmark,
        using the x and y values scaled to the provided image size.

        :param landmarks: Pandas Series containing numeric landmark data (excluding timestamp).
        :param image_size: Tuple (width, height) specifying the size of the output image.
        :return: An image (numpy array) with the landmarks drawn.
        """
        # Create a blank image (black background).
        img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        n = len(landmarks)
        num_landmarks = n // 4  # Each landmark has 4 values: x, y, z, and confidence.

        for i in range(num_landmarks):
            try:
                # Extract x and y coordinates from the appropriate columns.
                x = float(landmarks.iloc[i * 4])
                y = float(landmarks.iloc[i * 4 + 1])
                # Scale the coordinates to the image dimensions.
                x_int = int(x * image_size[0])
                y_int = int(y * image_size[1])
                cv2.circle(img, (x_int, y_int), 2, (255, 255, 255), -1)
            except (ValueError, IndexError):
                continue  # Skip if any conversion fails.

        return img
