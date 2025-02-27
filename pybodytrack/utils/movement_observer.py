class MovementObserver:
    def __init__(self):
        # List to store movement data updates
        self.movements = []

    def update(self, movement):
        """
        This method will be called each time a new movement measurement is computed.
        It appends the movement value to a list and prints it.
        """
        self.movements.append(movement)
        print(f"Movement update: {movement}")