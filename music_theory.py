import numpy as np

# Generate a dictionary mapping note names to MIDI note numbers
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

N = {}  # shorthand for notes

for midi_number in range(128):
    note_in_octave = midi_number % 12
    octave = (midi_number // 12) - 1
    note_name = note_names[note_in_octave] + str(octave)
    N[note_name] = midi_number

# Define scale intervals in semitones
SCALE_SET = {
    'Major': [0, 2, 4, 5, 7, 9, 11],       # Intervals for Major scale
    'Minor': [0, 2, 3, 5, 7, 8, 10],       # Intervals for Natural Minor scale
    'HarmMinor': [0, 2, 3, 5, 7, 8, 11],   # Intervals for Harmonic Minor scale
    'PentMajor': [0, 2, 4, 7, 9],          # Intervals for Major Pentatonic scale
    'PentMinor': [0, 2, 3, 7, 9],          # Intervals for Minor Pentatonic scale
    'justFifthLol': [0, 7],                # Intervals for a perfect fifth
    'No.': [0]                             # Single note (unison)
}

# Assign relative priorities to each note in the scales
SEVEN_NOTE_PRIORITY = [0.99, 0.3, 0.8, 0.7, 0.9, 0.3, 0.1]  # Priorities for seven-note scales
PENT_NOTE_PRIORITY = [0.9, 0.7, 0.8, 0.8, 0.7]              # Priorities for pentatonic scales
SCALE_ARBITRARY_PRIORITIES = {
    'Major': SEVEN_NOTE_PRIORITY,
    'Minor': SEVEN_NOTE_PRIORITY,
    'HarmMinor': SEVEN_NOTE_PRIORITY,
    'PentMajor': PENT_NOTE_PRIORITY,
    'PentMinor': PENT_NOTE_PRIORITY,
    'justFifthLol': [0.9, 0.8],  # Priorities for perfect fifth intervals
    'No.': [0.1],                # Priority for unison
}

# Define chord intervals in semitones
CHORD_INTERVALS = {
    'Major': [0, 4, 7],  # Root, Major Third, Perfect Fifth
    'Minor': [0, 3, 7],  # Root, Minor Third, Perfect Fifth
    'HarmMinor': [0, 3, 7],  # Root, Minor Third, Perfect Fifth
    'PentMajor': [0, 4, 7],  # Root, Major Third, Perfect Fifth
    'PentMinor': [0, 3, 7],  # Root, Minor Third, Perfect Fifth
    'justFifthLol': [0, 7],  # Root, Perfect Fifth
}

# Define a chord sequence with root notes and corresponding scales
CHORD_SEQUENCE = [
    [N["A2"], 'Minor'],
    [N["D2"], 'Minor'],
    [N["G2"], 'Major'],
    [N["C2"], 'Major'],
]

class NoteSet:
    """
    Represents a set of MIDI notes and their associated priorities based on a given scale and root note.
    """
    def __init__(self, base_note, scale_name='Major', min_note=24, max_note=87):
        self.base_note = base_note
        self.scale_name = scale_name
        self.min_note = min_note
        self.max_note = max_note
        self.notes, self.priorities = self.generate_note_set()

    def generate_note_set(self):
        interval_set = SCALE_SET[self.scale_name]
        arbitrary_interval_priority = SCALE_ARBITRARY_PRIORITIES[self.scale_name]

        base_note = self.base_note
        while base_note >= self.min_note:
            base_note -= 12

        out_notes = []
        out_priority = []
        while base_note < self.max_note:
            for i in range(len(interval_set)):
                note = base_note + interval_set[i]
                if note < self.min_note or note > self.max_note:
                    continue
                out_notes.append(int(note))
                out_priority.append(arbitrary_interval_priority[i % len(arbitrary_interval_priority)])
            base_note += 12

        notes_array = np.array(out_notes, dtype=int)
        priorities_array = np.array(out_priority, dtype=float)
        return notes_array, priorities_array
