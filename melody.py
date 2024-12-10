import mido                # Import the mido library for MIDI input/output operations
import time                # Import the time module for timing control in the main loop
from random import random, choice  # Import random functions for probabilities and random selection
import numpy as np         # Import NumPy library for numerical operations on arrays
import math
import MIDI_Funcs as MIDI_Funcs          # Import a custom module for MIDI utilities (assumed to handle MIDI cleanup)

# Constants defining the MIDI note range
NOTE_RANGE_MIN = 24  # Lowest MIDI note to use (C1, MIDI note number 24)
NOTE_RANGE_MAX = 87  # Highest MIDI note to use (D#6, MIDI note number 87)

# Define BPM and timing settings
BPM = 240           # Set the tempo of the music to 120 beats per minute
TICKS_PER_BEAT = 4   # Number of ticks per beat (representing sixteenth notes)
BEATS_PER_MEASURE = 4  # Number of beats per measure (standard 4/4 time signature)

# Calculate derived timing constants
SECONDS_PER_BEAT = 60.0 / BPM  # Duration of one beat in seconds
TIME_DELAY = SECONDS_PER_BEAT / TICKS_PER_BEAT  # Delay between ticks (duration of one tick)
MEASURE_DURATION = BEATS_PER_MEASURE * TICKS_PER_BEAT  # Total number of ticks in one measure

# Generate a dictionary mapping note names to MIDI note numbers
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

N = {}# shorthand for notes

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
    [N["C2"], 'Major'],  # D minor pentatonic (MIDI note 50 is D3)
    [N["D2"], 'Minor'],      # G major (MIDI note 55 is G3)
    [N["G2"], 'Major'],      # C major (MIDI note 60 is C4, Middle C)
]

# Function to convert MIDI note numbers to note names
def midi_note_to_name(midi_note):
    if midi_note is None:
        return 'Rest'  # Return 'Rest' if the note is None (silence)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']  # Note names for 12 semitones
    octave = (midi_note // 12) - 1  # Calculate the octave number (MIDI note 0 is C-1)
    note = midi_note % 12  # Find the note within the octave
    note_name = note_names[note] + str(octave)  # Combine note name and octave number
    return note_name  # Return the note name (e.g., 'C4')

class NoteSet:
    """
    Represents a set of MIDI notes and their associated priorities based on a given scale and root note.
    """
    def __init__(self, base_note, scale_name='Major', min_note=NOTE_RANGE_MIN, max_note=NOTE_RANGE_MAX):
        self.base_note = base_note  # Store the base note (root note of the chord)
        self.scale_name = scale_name  # Store the scale name (e.g., 'Major', 'Minor')
        self.min_note = min_note  # Minimum MIDI note number for this note set
        self.max_note = max_note  # Maximum MIDI note number for this note set
        # Generate the note set and priorities upon initialization
        self.notes, self.priorities = self.generate_note_set()

    def generate_note_set(self):
        """
        Generates all the notes and their priorities for the scale starting from the base note,
        covering the specified MIDI note range.
        """
        interval_set = SCALE_SET[self.scale_name]  # Get the intervals for the specified scale
        arbitrary_interval_priority = SCALE_ARBITRARY_PRIORITIES[self.scale_name]  # Get priorities for the scale

        base_note = self.base_note  # Start from the base note
        # Adjust the base note to be below the minimum note range
        while base_note >= self.min_note:
            base_note -= 12  # Decrease by one octave until below minimum note range

        out_notes = []  # Initialize list to store generated MIDI note numbers
        out_priority = []  # Initialize list to store corresponding note priorities
        # Generate notes across octaves until reaching the maximum note range
        while base_note < self.max_note:
            for ii in range(len(interval_set)):
                note = base_note + interval_set[ii]  # Calculate the MIDI note number
                if note < self.min_note:
                    continue  # Skip notes below the minimum note range
                if note > self.max_note:
                    continue     # Skip notes above the maximum note range
                out_notes.append(int(note))  # Add the note to the list
                out_priority.append(arbitrary_interval_priority[ii % len(arbitrary_interval_priority)])  # Add the corresponding priority
            base_note += 12  # Move to the next octave

        # Store the generated notes and priorities
        notes_array = np.array(out_notes, dtype=int)
        priorities_array = np.array(out_priority, dtype=float)
        return notes_array, priorities_array

class MidiHandler:
    """
    Handles MIDI output ports and sending MIDI messages.
    """

    def __init__(self):
        # Open MIDI output ports for melody and choir
        self.melody_out = mido.open_output('IAC Driver Bus 1')      # Melody output port
        self.choir_out1 = mido.open_output('IAC Driver Bus 2')      # Choir output port 1
        self.choir_out2 = mido.open_output('IAC Driver Bus 3')      # Choir output port 2
        self.choir_out3 = mido.open_output('IAC Driver Bus 4')      # Choir output port 3
        self.brass_out = mido.open_output('IAC Driver Bus 5')       # Brass output port
        self.organ_out = mido.open_output('IAC Driver Bus 6')       # Organ output port
        self.outport_sets = [self.melody_out, self.choir_out1, self.choir_out2, self.choir_out3, self.brass_out, self.organ_out]  # List of all ports
        # Ensure clean exit by turning off any lingering notes
        MIDI_Funcs.niceMidiExit(self.outport_sets)  # Call MIDI cleanup function

    def send_note_on(self, port, note, velocity=100):
        """Sends a 'note_on' MIDI message to the specified port."""
        port.send(mido.Message('note_on', note=int(note), velocity=int(velocity)))  # Send note_on message

    def send_note_off(self, port, note):
        """Sends a 'note_off' MIDI message to the specified port."""
        port.send(mido.Message('note_off', note=int(note)))  # Send note_off message

    def close_ports(self):
        """Closes all MIDI ports and ensures all notes are turned off."""
        for outport in self.outport_sets:
            outport.send(mido.Message('note_off', note=0, velocity=0))  # Turn off all notes
            outport.close()  # Close the port

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

class ProceduralMusicGenerator:
    """
    Main class responsible for generating procedural music.
    """

    def __init__(self):
        self.midi_handler = MidiHandler()  # Initialize MIDI handler
        self.chord_sequence = CHORD_SEQUENCE  # Use predefined chord sequence
        self.current_chord_index = 0  # Start with the first chord
        self.notes_since_chord_change = 0  # Ticks since last chord change
        self.current_tick = -1  # Initialize tick counter

        # Timing settings
        self.ticks_per_beat = TICKS_PER_BEAT                  # Ticks per beat (e.g., 4 for sixteenth notes)
        self.beats_per_measure = BEATS_PER_MEASURE            # Beats per measure (e.g., 4 for 4/4 time)
        self.measure_duration = MEASURE_DURATION              # Total ticks per measure
        self.time_delay = TIME_DELAY                          # Delay between ticks
        self.chord_sequence_duration = self.measure_duration  # Ticks per chord change

        # Initialize history buffers for each instrument
        self.history_size = self.measure_duration * 3         # Size of the history buffers
        self.melody_history = HistoryBuffer(self.history_size)  # Melody history
        self.choir_history = HistoryBuffer(self.history_size)   # Choir history
        self.brass_history = HistoryBuffer(self.history_size)   # Brass history
        self.organ_history = HistoryBuffer(self.history_size)   # Organ history

        # Initialize velocity sums for dynamics
        self.melody_vel_sum = 100  # Melody velocity sum
        self.choir_vel_sum = 100   # Choir velocity sum
        self.brass_vel_sum = 100   # Brass velocity sum
        self.organ_vel_sum = 100   # Organ velocity sum

        # Initialize the first chord and note sets for each voice
        self.initialize_note_sets()

        # Initialize the first note for each instrument
        self.melody_note = int(self.melody_note_set.notes[0])
        # self.choir_note = int(self.choir_note_set.notes[0])
        # Initialize previous choir notes with root, third, and fifth of the first chord
        self.choir_notes = [
            self.chord_sequence[0][0] + interval
            for interval in CHORD_INTERVALS[self.chord_sequence[0][1]]
        ]
        self.brass_note = None  # Start with no note
        self.organ_note = int(self.organ_note_set.notes[0])

    def initialize_note_sets(self):
        """Initializes separate NoteSets for each voice with appropriate ranges based on current chord."""
        base_note, scale_name = self.chord_sequence[self.current_chord_index]

        # Determine min_note and max_note for each voice based on base_note
        # Ensure they are within the allowed MIDI note range
        # Melody Voice: One to three octaves above the base_note
        melody_min_note = max(base_note + 12, NOTE_RANGE_MIN)
        melody_max_note = min(base_note + 36, NOTE_RANGE_MAX)
        self.melody_note_set = NoteSet(base_note, scale_name, min_note=melody_min_note, max_note=melody_max_note)

        # Choir Voice: From base_note up to two octaves above
        choir_min_note = max(base_note + 12, NOTE_RANGE_MIN)
        choir_max_note = min(base_note + 36, NOTE_RANGE_MAX)
        self.choir_note_set = NoteSet(base_note, scale_name, min_note=choir_min_note, max_note=choir_max_note)

        # Brass Voice: One octave below to one octave above the base_note
        brass_min_note = max(base_note - 12, NOTE_RANGE_MIN)
        brass_max_note = min(base_note + 12, NOTE_RANGE_MAX)
        self.brass_note_set = NoteSet(base_note, scale_name, min_note=brass_min_note, max_note=brass_max_note)

        # Organ Voice: Two octaves below up to the base_note
        organ_min_note = max(base_note + 12, NOTE_RANGE_MIN)
        organ_max_note = min(base_note + 24, NOTE_RANGE_MAX)
        self.organ_note_set = NoteSet(base_note, scale_name, min_note=organ_min_note, max_note=organ_max_note)

    def run(self):
        """
        Main loop that runs the procedural music generation.
        """
        try:
            while True:
                start_time = time.time()  # Start time of the tick
                self.current_tick += 1    # Increment tick counter

                # Handle chord changes
                self.handle_chord_changes()

                # Process each instrument
                self.process_melody()
                self.process_choir()
                self.process_brass()
                self.process_organ()

                # Print out the current state
                self.print_current_state()

                end_time = time.time()  # End time after processing

                # Delay to maintain consistent timing
                sleep_time = self.time_delay - (end_time - start_time)  # Calculate sleep time
                if sleep_time > 0:
                    time.sleep(sleep_time)  # Sleep to maintain timing
                else:
                    print("TIMING ERROR!!! Not enough time to process generation between ticks")  # Timing warning

        except KeyboardInterrupt:
            # Graceful shutdown on KeyboardInterrupt
            print("\nKeyboardInterrupt detected, stopping gracefully...")
            self.midi_handler.close_ports()  # Close MIDI ports
            print("All MIDI ports have been closed successfully.")

    def handle_chord_changes(self):
        """
        Manages chord progression.
        """
        self.notes_since_chord_change += 1  # Increment ticks since last chord change
        if self.notes_since_chord_change >= self.chord_sequence_duration:
            # Time to change the chord
            self.notes_since_chord_change = 0  # Reset counter
            self.current_chord_index = (self.current_chord_index + 1) % len(self.chord_sequence)  # Next chord
            # Re-initialize note sets for the new chord
            self.initialize_note_sets()

    def process_melody(self):
        """
        Generates and plays notes for the melody (solo) voice.
        """
        previous_note = self.melody_note  # Previous note for reference
        notes_since_change = self.notes_since_chord_change  # Ticks since last chord change

        play_note_odds = random()  # Base probability to play a note

        density_factor = 0.5       # Higher values increase the likelihood of playing a note
        neighborhood_factor = 0.7  # Values between 0 and 1; lower values favor closer notes

        # Increase odds on chord change or beat
        if notes_since_change == 0:
            play_note_odds *= 5  # Higher chance on chord change
        elif notes_since_change % self.ticks_per_beat == 0:
            play_note_odds *= 2  # Higher chance on beat

        do_play_note = play_note_odds > density_factor  # Decide whether to play a note

        curr_chord = self.melody_note_set.notes  # Get melody note set
        curr_priority = self.melody_note_set.priorities  # Get melody priorities


        adjusted_priorities = curr_priority.copy()  # Copy priorities to adjust

        # Adjust priorities based on distance from previous note
        for i in range(len(curr_chord)):
            distance = abs(curr_chord[i] - previous_note)          # Calculate distance from previous note
            adjusted_priorities[i] *= pow(neighborhood_factor, distance)  # Adjust priority based on distance

        priority_sum = sum(adjusted_priorities)  # Sum of adjusted priorities
        adjusted_priorities /= priority_sum  # Normalize priorities

        if do_play_note:
            # Select a note based on adjusted priorities
            cumulative_priorities = np.cumsum(adjusted_priorities)  # Cumulative sum for selection
            random_value = random()  # Random value for selection
            selected_index = np.searchsorted(cumulative_priorities, random_value)  # Find note index
            selected_note = int(curr_chord[selected_index])  # Selected note

            note_vel = int(self.melody_vel_sum / 4 + 30 * random() + 30)  # Calculate note velocity

            # Play the note
            if previous_note is not None:
                self.midi_handler.send_note_off(self.midi_handler.melody_out, previous_note)  # Turn off previous note
            self.midi_handler.send_note_on(self.midi_handler.melody_out, selected_note, velocity=note_vel)  # Play new note

            self.melody_history.push(notes=selected_note, velocities=note_vel)  # Update history
            self.melody_note = selected_note  # Update current note

            self.melody_vel_sum = (note_vel + self.melody_vel_sum * 3) / 4  # Update velocity sum
        else:
            pass  # Hold the note

    def process_choir(self):
        """
        Generates and plays three choir notes (root, third, fifth), each to different MIDI outputs.
        """
        previous_notes = self.choir_notes  # Previous choir notes

        # Play new notes on chord change
        if self.notes_since_chord_change == 0:
            base_note, scale_name = self.chord_sequence[self.current_chord_index]

            # Get chord intervals from the CHORD_INTERVALS dictionary
            chord_intervals = CHORD_INTERVALS.get(scale_name, CHORD_INTERVALS['Major'])  # Default to Major

            # Compute the chord notes
            chord_notes = [base_note + interval for interval in chord_intervals]

            # Ensure the bass note is within choir_note_set
            choir_notes_set = set(self.choir_note_set.notes)
            bass_note = chord_notes[0]

            # Adjust bass_note upwards by octaves until it is in choir_note_set
            while bass_note not in choir_notes_set:
                bass_note += 12  # Increase by one octave
                if bass_note > NOTE_RANGE_MAX:
                    # If bass_note exceeds the maximum note range, select the closest note from choir_note_set
                    bass_note = min(self.choir_note_set.notes, key=lambda x: abs(x - bass_note))
                    break

            # Update the bass note in chord_notes
            chord_notes[0] = bass_note

            # Map outputs
            choir_outputs = [self.midi_handler.choir_out1, self.midi_handler.choir_out2, self.midi_handler.choir_out3]

            # Note velocities
            note_vel = int(self.choir_vel_sum / 4 + 30 * random() + 30)

            # Turn off previous notes
            for prev_note, port in zip(previous_notes, choir_outputs):
                if prev_note is not None:
                    self.midi_handler.send_note_off(port, prev_note)

            # Send note_on messages for new notes
            for note, port in zip(chord_notes, choir_outputs):
                self.midi_handler.send_note_on(port, note, velocity=note_vel)
                print(f"Sending note {note} ({midi_note_to_name(note)}) to port {port.name}")

            # Update previous notes
            self.choir_notes = chord_notes

            # Update velocity sum
            self.choir_vel_sum = (note_vel + self.choir_vel_sum * 3) / 4
        else:
            pass  # Hold the notes


    # def process_choir(self):
    #     """
    #     Generates and plays notes for the choir voice.
    #     """
    #     previous_note = self.choir_note  # Previous choir note

    #     # Decide to play new note on chord change or beat
    #     if self.notes_since_chord_change == 0 or self.notes_since_chord_change % self.ticks_per_beat == 0:
    #         curr_chord = self.organ_note_set.notes  # Current chord notes
    #         print(curr_chord)
    #         selected_note = int(choice(curr_chord) - 12)  # Choose note an octave lower

    #         note_vel = int(self.choir_vel_sum / 4 + 30)  # Calculate velocity

    #         if previous_note is not None:
    #             self.midi_handler.send_note_off(self.midi_handler.choir_out, previous_note)  # Turn off previous note
    #         self.midi_handler.send_note_on(self.midi_handler.choir_out, selected_note, velocity=note_vel)  # Play new note

    #         self.choir_history.push(notes=selected_note, velocities=note_vel)  # Update history
    #         self.choir_note = selected_note  # Update current note
    #         self.choir_vel_sum = (note_vel + self.choir_vel_sum * 3) / 4  # Update velocity sum
    #     else:
    #         pass  # Hold the note


    def process_brass(self):
        """
        Generates and plays notes for the brass voice.
        """
        previous_note = self.brass_note  # Previous brass note

        # Process on each tick
        # Play note on the beat
        if self.notes_since_chord_change % self.ticks_per_beat == 0:
            curr_chord = self.brass_note_set.notes  # Current chord notes
            brass_notes = [curr_chord[0], curr_chord[4 % len(curr_chord)]]  # Root and fifth
            selected_note = int(choice(brass_notes))  # Select note

            note_vel = int(self.brass_vel_sum / 4 + 40)  # Calculate velocity

            if previous_note is not None:
                self.midi_handler.send_note_off(self.midi_handler.brass_out, previous_note)  # Turn off previous note
            self.midi_handler.send_note_on(self.midi_handler.brass_out, selected_note, velocity=note_vel)  # Play new note

            self.brass_history.push(notes=selected_note, velocities=note_vel)  # Update history
            self.brass_note = selected_note  # Update current note

            self.brass_vel_sum = (note_vel + self.brass_vel_sum * 3) / 4  # Update velocity sum
        else:
            # Release the note
            if previous_note is not None:
                self.midi_handler.send_note_off(self.midi_handler.brass_out, previous_note)  # Turn off note
            self.brass_note = None  # Set current note to None

    def process_organ(self):
        """
        Generates and plays notes for the organ voice.
        """
        previous_note = self.organ_note  # Previous organ note

        # Play note on chord change
        if self.notes_since_chord_change == 0:
            bass_note = int(self.chord_sequence[self.current_chord_index][0] - 24)  # Bass note two octaves down
            note_vel = int(self.organ_vel_sum / 4 + 50)  # Calculate velocity

            if previous_note is not None:
                self.midi_handler.send_note_off(self.midi_handler.organ_out, previous_note)  # Turn off previous note
            self.midi_handler.send_note_on(self.midi_handler.organ_out, bass_note, velocity=note_vel)  # Play new note

            self.organ_history.push(notes=bass_note, velocities=note_vel)  # Update history
            self.organ_note = bass_note  # Update current note

            self.organ_vel_sum = (note_vel + self.organ_vel_sum * 3) / 4  # Update velocity sum
        else:
            pass  # Hold the note

    def print_current_state(self):
        """
        Prints the current state of the music in the specified format.
        """
        chord_root_midi = self.chord_sequence[self.current_chord_index][0]  # Current chord root note
        chord_root_name = midi_note_to_name(chord_root_midi)  # Convert to note name

        # Calculate current beat and subdivision
        current_beat = (self.notes_since_chord_change // self.ticks_per_beat) + 1  # Beat number
        current_subdivision = (self.notes_since_chord_change % self.ticks_per_beat) + 1  # Tick within beat

        melody_note_name = midi_note_to_name(self.melody_note)  # Melody note name
        choir_note_names = [midi_note_to_name(note) for note in self.choir_notes]  # Choir note names
        choir_note_names_str = ', '.join(choir_note_names)  # Combine choir note names
        brass_note_name = midi_note_to_name(self.brass_note)    # Brass note name
        organ_note_name = midi_note_to_name(self.organ_note)    # Organ note name

        # Print formatted current state
        print(f"CHORD {chord_root_name} BEAT {current_beat} TICK {current_subdivision} "
              f"VOICE: {melody_note_name} / CHOIR {choir_note_names_str} / BRASS {brass_note_name} / ORGAN {organ_note_name}")

if __name__ == "__main__":
    # Instantiate the music generator and start the main loop
    generator = ProceduralMusicGenerator()  # Create instance of the generator
    generator.run()  # Run the main loop
