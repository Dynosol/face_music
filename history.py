import numpy as np

class HistoryBuffer:
    """
    Maintains a rolling history of notes, velocities, and other data.
    """

    def __init__(self, size):
        self.size = size  # Size of the history buffer
        self.notes = np.zeros(size, np.int16)  # Initialize note history array
        self.velocities = np.full(size, -2, np.int16)  # Initialize velocity history array

    def push(self, notes=None, velocities=None):
        """Adds new data to the history buffers."""
        if notes is not None:
            self.notes = np.roll(self.notes, 1)  # Shift notes array to the right
            self.notes[0] = int(notes)           # Insert new note at the beginning
        if velocities is not None:
            self.velocities = np.roll(self.velocities, 1)  # Shift velocities array
            self.velocities[0] = int(velocities)           # Insert new velocity
