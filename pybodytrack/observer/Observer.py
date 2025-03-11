import queue
import threading


class Observer:
    def __init__(self):
        # Thread-safe queue to hold messages
        self.msg_queue = queue.Queue()

    def sendMessage(self, msg):
        """
        Called by the BodyTracking class to send a new message.
        """
        self.msg_queue.put(msg)

    def handleMessage(self, msg):
        """
        Override this method to handle incoming messages.
        """
        pass

    def startLoop(self):
        """
        Starts the message loop in a daemon thread.
        """
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            print("DENTRO DEL LOOP")
            msg = self.msg_queue.get()
            print("VALOR DE MSG",msg)
            if msg is None:  # Use None as a signal to stop the loop if desired.
                print("BUCLE OBSERVER","MSG ES NONE")
                break
            print("OBSERVEr","HAY ALGUNMENSAJE")
            self.handleMessage(msg)
