import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Optional
#source pybodytrack_env/bin/activate

class Utils:
    @staticmethod
    def filter_dataframe_by_time(df, start_time, end_time):
        """
        Filters a DataFrame based on a given time range.

        Args:
            df (pandas.DataFrame): DataFrame containing a 'timestamp' column.
            start_time (float): Start time of the range.
            end_time (float): End time of the range.

        Returns:
            pandas.DataFrame: Subset of the DataFrame where 'timestamp' is within the specified range.
        """
        try:
            # Ensure 'timestamp' is of float type
            df = df.copy()  # Avoid modifying the original DataFrame
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

            # Filter DataFrame by time range
            filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

            return filtered_df.dropna(subset=['timestamp'])  # Remove any rows with NaN timestamps

        except Exception as e:
            print(f"Error filtering DataFrame by time: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on failure

    @staticmethod
    def clean_dataframe(df):
        """
        Cleans a DataFrame by removing rows where the 'timestamp' column contains non-numeric values.

        Args:
            df (pandas.DataFrame): DataFrame containing a 'timestamp' column.

        Returns:
            pandas.DataFrame: A cleaned DataFrame with only valid numeric timestamps.
        """
        df_cleaned = df.copy()

        # Convert 'timestamp' to float, setting invalid values to NaN
        df_cleaned['timestamp'] = pd.to_numeric(df_cleaned['timestamp'], errors='coerce')

        # Count removed values
        removed_values = df_cleaned['timestamp'].isna().sum()

        # Drop rows with NaN timestamps
        df_cleaned = df_cleaned.dropna(subset=['timestamp'])

        print(f"Removed invalid timestamp values: {removed_values}")

        return df_cleaned

    @staticmethod
    def get_subframes_list(df, interval):
        """
        Splits a DataFrame into subframes based on a given time interval.

        Args:
            df (pandas.DataFrame): DataFrame containing a 'timestamp' column.
            interval (float): Time interval for dividing the DataFrame.

        Returns:
            list of pandas.DataFrame: A list of subframes based on the specified time interval.
        """
        df = Utils.clean_dataframe(df)  # Sanitize the DataFrame
        subframes = []
        try:
            first_timestamp = df['timestamp'].iloc[0]
            last_timestamp = df['timestamp'].iloc[-1]
            start = first_timestamp
            end = start + interval
            continue_splitting = True
            subframe_count = 0

            while continue_splitting:
                filtered_df = Utils.filter_dataframe_by_time(df, start, end)

                if not filtered_df.empty:
                    subframe_count += 1
                    subframes.append(filtered_df)

                    if end > last_timestamp:
                        filtered_df = Utils.filter_dataframe_by_time(df, start, last_timestamp)
                        subframes.append(filtered_df)
                        continue_splitting = False

                start = end
                end += interval

        except Exception as e:
            print(f"Error in get_subframes_list: {e}")

        return subframes
    @staticmethod
    def get_sub_landmark(df, columns):
        """
        Returns a sub-DataFrame containing only the specified columns.

        Args:
            df (pandas.DataFrame): Original DataFrame.
            columns (list): List of column names to include in the sub-DataFrame.

        Returns:
            pandas.DataFrame: Sub-DataFrame with only the specified columns.
        """
        valid_columns = [col for col in columns if col in df.columns]
        return df[valid_columns]

    @staticmethod
    def convert_timestamp(timestamp):
        """
        Converts a timestamp to the format "dd/MM/yyyy hh:mm:ss".

        Args:
            timestamp (float): Timestamp in seconds.

        Returns:
            str: The formatted date and time as "dd/MM/yyyy hh:mm:ss".
        """
        try:
            # Convert the timestamp to a datetime object
            formatted_datetime = datetime.datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S")
            return formatted_datetime
        except (ValueError, OSError) as e:
            print(f"Error converting timestamp: {e}")
            return None  # Return None if conversion fails

    @staticmethod
    def get_day_and_month(timestamp: float) -> tuple[int, int]:
        """
        Extracts the day and month from a timestamp.

        Args:
            timestamp (float): Unix timestamp in seconds.

        Returns:
            tuple[int, int]: A tuple containing the day and month as integers (day, month).
        """
        date_time = datetime.fromtimestamp(timestamp)
        return date_time.day, date_time.month


    @staticmethod
    def get_object_by_key(objects: List[Dict], key: str) -> Optional[Dict]:
        """
        Retrieves an object from a list by its key.

        Args:
            objects (List[Dict]): A list of dictionaries with the structure
                                  {"key": <key_value>, "values": []}.
            key (str): The key to search for in the list.

        Returns:
            Optional[Dict]: The object with the specified key, or None if not found.
        """
        return next((obj for obj in objects if obj["key"] == key), None)

    @staticmethod
    def movement_per_second(total_movement, df):
        """
        Calculate the average movement per second.

        Assumes the DataFrame has a 'timestamp' column as the first column,
        with timestamps expressed in seconds.

        :param total_movement: Total movement value (from any motion method).
        :param df: Pandas DataFrame containing the landmark data, including 'timestamp'.
        :return: Movement per second.
        """
        # Extract timestamps (assumed to be the first column)
        timestamps = df.iloc[:, 0].values
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return 0.0
        return total_movement / duration

    @staticmethod
    def movement_per_frame(total_movement, df):
        """
        Calculate the average movement per frame.

        :param total_movement: Total movement value computed using any motion method.
        :param df: Pandas DataFrame containing the landmark data.
        :return: Average movement per frame.
        """
        n_frames = len(df)
        if n_frames <= 1:
            return 0.0
        return total_movement / (n_frames - 1)

    @staticmethod
    def movement_per_landmark(total_movement, num_landmarks):
        """
        Calculate the average movement per landmark.

        :param total_movement: Total movement computed using any motion method.
        :param num_landmarks: Total number of landmarks.
        :return: Average movement per landmark.
        """
        if num_landmarks <= 0:
            return 0.0
        return total_movement / num_landmarks

    @staticmethod
    def frame_movement_statistics(frame_movements):
        """
        Calculate statistical metrics for movement across frames.

        :param frame_movements: List or array of movement values per frame.
        :return: Dictionary containing average, standard deviation, median, and 95th percentile.
        """
        if len(frame_movements) == 0:
            return {}
        stats = {
            "average": np.mean(frame_movements),
            "std_dev": np.std(frame_movements),
            "median": np.median(frame_movements),
            "p95": np.percentile(frame_movements, 95)
        }
        return stats

    @staticmethod
    def normalized_movement_index(total_movement, df, num_landmarks):
        """
        Calculate a normalized movement index by dividing the total movement by both the
        duration (in seconds) and the number of landmarks. This yields a dimensionless index,
        facilitating comparison across videos with different durations or landmark counts.

        Assumes the DataFrame has a 'timestamp' column as the first column.

        :param total_movement: Total movement computed using any motion method.
        :param df: Pandas DataFrame with the landmark data (including 'timestamp').
        :param num_landmarks: Total number of landmarks.
        :return: Normalized movement index.
        """
        # Extract the timestamp column (assumed to be the first column)
        timestamps = df.iloc[:, 0].values
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0 or num_landmarks <= 0:
            return 0.0
        return total_movement / (duration * num_landmarks)
