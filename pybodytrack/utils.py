import pandas as pd
import datetime

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
