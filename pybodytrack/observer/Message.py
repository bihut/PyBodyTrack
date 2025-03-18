"""
PyBodyTrack - A Python library for motion quantification in videos.

Author: Angel Ruiz Zafra
License: Apache 2.0 License
Version: 2025.3.2
Repository: https://github.com/bihut/PyBodyTrack
Created on 4/2/25 by Angel Ruiz Zafra
"""

class Message:
    def __init__(self, what, obj=None):
        """
        Parameters:
            what (int): A code indicating the type of message (e.g., 1 for new data, 2 for error).
            obj: The object carrying the message data (e.g., a landmark data row).
        """
        self.what = what
        self.obj = obj