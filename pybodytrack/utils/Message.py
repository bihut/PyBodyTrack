class Message:
    def __init__(self, what, obj=None):
        """
        Parameters:
            what (int): A code indicating the type of message (e.g., 1 for new data, 2 for error).
            obj: The object carrying the message data (e.g., a landmark data row).
        """
        self.what = what
        self.obj = obj