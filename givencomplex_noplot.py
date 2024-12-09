# TAKEN FROM https://github.com/GarettMorrison/Procedural_MIDI
# All credit to Garett Morrison for the generated melody method

import mido                # Library for MIDI input/output
import time                # Used for timing control in the main loop
from copy import deepcopy  # For copying complex objects without reference issues
from random import random, choice # Random number generator for probabilities
import numpy as np         # Numerical operations on arrays
import math as m           # Mathematical functions

import MIDI_Funcs          # Custom module for MIDI utilities (assumed to handle MIDI cleanup)

# Constants defining the MIDI note range
NOTE_RANGE_MIN = 24  # Lowest MIDI note to use (C1)
NOTE_RANGE_MAX = 87  # Highest MIDI note to use (D#6)

# Time division constant to control tempo
TIME_DIV_CONSTANT = 2.4

# Define scale intervals in semitones
SCALE_SET = {
    'Major': [0, 2, 4, 5, 7, 9, 11],       # Major scale intervals
    'Minor': [0, 2, 3, 5, 7, 8, 10],       # Natural minor scale intervals
    'HarmMinor': [0, 2, 3, 5, 7, 8, 11],   # Harmonic minor scale intervals
    'PentMajor': [0, 2, 4, 7, 9],          # Major pentatonic scale intervals
    'PentMinor': [0, 2, 3, 7, 9],          # Minor pentatonic scale intervals
    'justFifthLol': [0, 7],                # Interval of a perfect fifth
    'No.': [0]                             # Single note (unison)
}

# Assign relative priorities to each note in the scales (importance or likelihood of being played)
SEVEN_NOTE_PRIORITY = [0.99, 0.3, 0.8, 0.7, 0.9, 0.3, 0.4]  # Priorities for seven-note scales
PENT_NOTE_PRIORITY = [0.9, 0.7, 0.8, 0.8, 0.7]              # Priorities for pentatonic scales
SCALE_ARBITRARY_PRIORITIES = {
    'Major': SEVEN_NOTE_PRIORITY,
    'Minor': SEVEN_NOTE_PRIORITY,
    'HarmMinor': SEVEN_NOTE_PRIORITY,
    'PentMajor': PENT_NOTE_PRIORITY,
    'PentMinor': PENT_NOTE_PRIORITY,
    'justFifthLol': [0.9, 0.8],
    'No.': [0.1],
}

# Define a chord sequence with root notes and corresponding scales
CHORD_SEQUENCE = [
    [50, 'PentMinor'],  # D minor pentatonic (MIDI note 50 is D3)
    [55, 'Major'],      # G major (MIDI note 55 is G3)
    [60, 'Major'],      # C major (MIDI note 60 is C4, Middle C)
    [57, 'HarmMinor'],  # A harmonic minor (MIDI note 57 is A3)
]

# Possible settings for the alto (middle) voice
ALTO_SETTING_OPTIONS = [
    'match', 'match', 'match',  # Frequently matches the lead voice
    'decay', 'decay',           # Decaying patterns
    'arpeggio', 'arpeggio',     # Arpeggiated patterns
    'pulse',                    # Pulsing rhythms
    'random',                   # Random note selection
]

# Possible settings for the bass voice
BASS_SETTING_OPTIONS = [
    'hold', 'hold', 'hold',     # Sustained notes
    'clock',                    # Plays on the beat
    'div_2', 'div_2',           # Divides the measure by 2
    'div_3',                    # Divides the measure by 3
    'div_4',                    # Divides the measure by 4
    'strike',                   # Emphasizes the note
    'none',                     # No bass note
]

# Time signature options (number of ticks per measure and subdivisions)
TIME_SPLITS = [
    (9, 2), (16, 8), (9, 3), (12, 8), (9, 4), (8, 3), (10, 4),
    (8, 4), (9, 3), (10, 2), (10, 5), (12, 3), (12, 4), (12, 6),
    (14, 7), (14, 2), (15, 3), (15, 5), (16, 4), (16, 8), (18, 3),
    (18, 6), (20, 5), (20, 4), (20, 8), (21, 3),
    (24, 3), (24, 8), (25, 5), (27, 9), (30, 6), (32, 4),
]

class NoteSet:
    """
    Represents a set of MIDI notes and their associated priorities based on a given scale and root note.
    This class handles the generation of all playable notes within the specified range for a particular scale.
    """

    def __init__(self, base_note, scale_name='Major'):
        self.base_note = base_note
        self.scale_name = scale_name
        # Generate the note set and priorities upon initialization
        self.notes, self.priorities = self.generate_note_set()

    def generate_note_set(self):
        """
        Generates all the notes and their priorities for the scale starting from the base note,
        covering the entire specified MIDI note range.
        """
        interval_set = SCALE_SET[self.scale_name]                   # Get the scale intervals
        arbitrary_interval_priority = SCALE_ARBITRARY_PRIORITIES[self.scale_name]  # Get the note priorities

        base_note = self.base_note
        # Adjust the base note to be within the playable range by moving down octaves if necessary
        while base_note >= NOTE_RANGE_MIN:
            base_note -= 12

        out_notes = []     # List to store the MIDI note numbers
        out_priority = []  # List to store the corresponding priorities
        # Generate notes across octaves until reaching the maximum note range
        while base_note < NOTE_RANGE_MAX:
            for ii in range(len(interval_set)):
                note = base_note + interval_set[ii]  # Calculate the MIDI note number
                if note < NOTE_RANGE_MIN:
                    continue  # Skip notes below the minimum range
                if note > NOTE_RANGE_MAX:
                    break     # Stop if the note exceeds the maximum range
                out_notes.append(note)
                out_priority.append(arbitrary_interval_priority[ii])
            base_note += 12  # Move to the next octave

        return np.array(out_notes), np.array(out_priority)  # Return as numpy arrays for efficiency

class MidiHandler:
    """
    Handles MIDI output ports and sending MIDI messages.
    This class abstracts the MIDI operations, making it easier to manage the outputs for different voices.
    """

    def __init__(self):
        # Open MIDI output ports for lead, bass, and alto voices
        self.arpp_out = mido.open_output('IAC Driver Bus 1')  # Lead voice output
        self.bass_out = mido.open_output('IAC Driver Bus 2')  # Bass voice output
        self.alto_out = mido.open_output('IAC Driver Bus 3')  # Alto voice output
        self.outport_sets = [self.arpp_out, self.bass_out, self.alto_out]
        # Ensure clean exit by turning off any lingering notes
        MIDI_Funcs.niceMidiExit(self.outport_sets)

    def send_note_on(self, port, note, velocity=100):
        """Sends a 'note_on' MIDI message to the specified port."""
        port.send(mido.Message('note_on', note=note, velocity=velocity))

    def send_note_off(self, port, note):
        """Sends a 'note_off' MIDI message to the specified port."""
        port.send(mido.Message('note_off', note=note))

    def close_ports(self):
        """Closes all MIDI ports and ensures all notes are turned off."""
        for outport in self.outport_sets:
            outport.send(mido.Message('note_off', note=0, velocity=0))  # Turn off all notes
            outport.close()  # Close the port

class HistoryBuffer:
    """
    Maintains a rolling history of notes, velocities, movements, and miscellaneous data.
    This is useful for referencing previous notes and patterns to influence future note generation.
    """

    def __init__(self, size):
        self.size = size
        self.notes = np.zeros(size, np.int16)              # Array to store note history
        self.moves = np.zeros(size, np.float16)            # Array to store note movement history
        self.velocities = np.full(size, -2, np.int16)      # Array to store velocity history
        self.misc = np.zeros(size, np.float16)             # Array for miscellaneous data

    def push(self, notes=None, moves=None, velocities=None, misc=None):
        """
        Adds new data to the history buffers, rolling the arrays to keep the most recent data at index 0.
        """
        if notes is not None:
            self.notes = np.roll(self.notes, 1)  # Shift notes array
            self.notes[0] = notes                # Insert new note
        if moves is not None:
            self.moves = np.roll(self.moves, 1)  # Shift moves array
            self.moves[0] = moves                # Insert new movement
        if velocities is not None:
            self.velocities = np.roll(self.velocities, 1)  # Shift velocities array
            self.velocities[0] = velocities                # Insert new velocity
        if misc is not None:
            self.misc = np.roll(self.misc, 1)    # Shift misc array
            self.misc[0] = misc                  # Insert new miscellaneous data

class ProceduralMusicGenerator:
    """
    Main class responsible for generating procedural music.
    It manages the state of the music, handles the logic for note generation,
    and interfaces with the MIDI handler to output the music.
    """

    def __init__(self):
        # Initialize MIDI handler for sending MIDI messages
        self.midi_handler = MidiHandler()
        self.chord_sequence = CHORD_SEQUENCE                  # Chord progression to follow
        self.current_chord_index = 0                          # Index of the current chord
        self.notes_since_chord_change = 0                     # Ticks since the last chord change
        self.scale_motion_velocity = 0                        # Tracks melodic movement between notes
        self.gap_count = 0                                    # Counts consecutive rests
        self.held_count = 0                                   # Counts how long a note has been held
        self.lead_vel_sum = 100                               # Sum of lead note velocities for smoothing
        self.alto_vel_sum = 100                               # Sum of alto note velocities for smoothing
        # Vibe parameters influencing note repetition and motion
        self.vibes_lead_rep_dist = 0.1
        self.vibes_lead_sequence_rep = 0.1
        self.vibes_lead_note_rep = 0.1
        self.vibes_lead_motion_rep = 0.1
        # Time signature and rhythm settings
        self.sig_div_count = 3
        self.alto_threshold = 0.01 + 0.5 * random() * random()  # Threshold for alto note activation
        self.lead_legato = 0.4 + random() * 0.4 + random() * 0.3  # Probability of lead note legato
        self.alto_setting = 'pulse'                             # Initial setting for alto voice
        self.bass_setting = 'none'                              # Initial setting for bass voice
        # Arpeggio settings for the alto voice
        self.alto_arp_up = True
        self.alto_arp_hit_end = False
        self.prev_alto_note = 0
        self.change_time_sig = False                            # Flag to change time signature
        self.time_splits = TIME_SPLITS                          # Available time signatures
        self.chord_sequence_duration = 30                       # Duration of each chord in ticks
        self.time_delay = 0.15                                  # Delay between ticks (controls tempo)
        self.current_tick = -1                                  # Current tick counter

        # Initialize history buffers for lead, alto, and bass voices
        self.history_size = self.chord_sequence_duration * 3    # Size of the history buffers
        self.lead_history = HistoryBuffer(self.history_size)
        self.alto_history = HistoryBuffer(self.history_size)
        self.bass_history = HistoryBuffer(self.history_size)

        # Initialize the first chord and note set
        self.current_note_set = self.get_current_note_set()
        self.current_note = self.current_note_set.notes[0]      # Starting note for the lead voice
        self.alto_note = self.current_note                      # Starting note for the alto voice

        # Start the bass note (two octaves below the root of the chord)
        bass_note = self.chord_sequence[self.current_chord_index][0] - 24
        self.midi_handler.send_note_on(self.midi_handler.bass_out, bass_note)
        self.bass_history.notes[0] = bass_note                  # Record the bass note in history
        self.bass_history.velocities[0] = 100                   # Record the bass velocity in history

        # Initialize the time signature and adjust settings accordingly
        self.initialize_time_signature()

    def get_current_note_set(self):
        """
        Retrieves the current set of notes and priorities based on the current chord and scale.
        """
        base_note, scale_name = self.chord_sequence[self.current_chord_index]
        return NoteSet(base_note, scale_name)  # Generate the note set for the current chord

    def initialize_time_signature(self):
        """
        Initializes the time signature by selecting a default time split and adjusting the time delay.
        """
        self.chord_sequence_duration = self.time_splits[18][0]   # Number of ticks per chord
        self.sig_div_count = self.time_splits[18][1]             # Number of subdivisions (beats) per measure
        self.time_delay = TIME_DIV_CONSTANT / self.chord_sequence_duration  # Adjust time delay to control tempo
        self.adjust_alto_velocity()                              # Adjust the alto velocity based on duration

    def adjust_alto_velocity(self):
        """
        Adjusts the alto velocity to ensure it remains within reasonable limits when the measure duration changes.
        """
        if self.chord_sequence_duration > 16:
            # Adjust the velocity scaling factor for longer durations
            alto_vel_adjust = (self.alto_threshold * 50) / (self.chord_sequence_duration - 12)
            alto_vel_adjust = max(0.5, alto_vel_adjust)
            alto_vel_adjust = min(alto_vel_adjust, 1.0)
            self.alto_vel_adjust = alto_vel_adjust
        else:
            self.alto_vel_adjust = 1.0  # No adjustment needed for shorter durations

    def print_sequence(self, full_print=True):
        """
        Prints the current sequence settings, including the time signature and part settings.
        """
        if self.chord_sequence_duration % self.sig_div_count == 0:
            time_sig_top_str = str(round(self.chord_sequence_duration / self.sig_div_count))
        else:
            time_sig_top_str = str(round(self.chord_sequence_duration / self.sig_div_count, 3))

        if full_print:
            print(f"\nSequence: Pattern: {self.chord_sequence_duration}/{self.sig_div_count} "
                  f"({time_sig_top_str} beats per measure)")
            print(f"Part Settings:   altoSetting: {self.alto_setting.ljust(10)}  bassSetting: {self.bass_setting}")

        print(f"altoThresh: {str(round(self.alto_threshold, 3)).ljust(5, ' ')}   "
              f"leadLegato: {str(round(self.lead_legato, 3)).ljust(5, ' ')}", end='')

    def run(self):
        """
        Main loop that runs the procedural music generation.
        It continuously processes chord changes, generates notes for each voice, and sends MIDI messages.
        """
        self.print_sequence()  # Print the initial sequence settings
        try:
            while True:
                start_time = time.time()  # Record the start time of the loop
                self.current_tick += 1    # Increment the tick counter

                # Handle chord changes and time signature adjustments
                self.handle_chord_changes()

                # Process each voice: bass, lead, and alto
                self.process_bass()
                self.process_lead()
                self.process_alto()

                end_time = time.time()  # Record the end time of processing

                # Delay to maintain consistent timing based on time_delay
                sleep_time = self.time_delay - (end_time - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("TIMING ERROR!!! Not enough time to process generation between ticks")

        except KeyboardInterrupt:
            # Graceful shutdown on KeyboardInterrupt
            print("\nKeyboardInterrupt detected, stopping gracefully...")
            self.midi_handler.close_ports()
            print("All MIDI ports have been closed successfully.")

    def handle_chord_changes(self):
        """
        Manages chord progression and handles changes in time signature and other settings.
        """
        self.notes_since_chord_change += 1  # Increment the count since the last chord change
        if self.notes_since_chord_change >= self.chord_sequence_duration:
            # Time to change the chord
            self.notes_since_chord_change = 0  # Reset the counter
            self.current_note_set = self.get_current_note_set()  # Update the note set for the new chord

            # Stop the previous bass note
            bass_note = self.chord_sequence[self.current_chord_index][0] - 24
            self.midi_handler.send_note_off(self.midi_handler.bass_out, bass_note)

            # Move to the next chord in the sequence
            self.current_chord_index = (self.current_chord_index + 1) % len(self.chord_sequence)

            if self.current_chord_index == 0:
                # At the end of the chord sequence, consider changing time signature
                if self.change_time_sig:
                    self.select_new_time_signature()
                    self.change_time_sig = False  # Reset the flag
                else:
                    # Adjust thresholds and legato settings slightly for variation
                    self.alto_threshold += 0.15 * (random() - 0.5)
                    self.lead_legato += 0.6 * (random() - 0.5)
                    self.print_sequence(full_print=False)
                    self.change_time_sig = True  # Set the flag to change time signature next time

                # Adjust alto velocity based on the new duration
                self.adjust_alto_velocity()
                if self.alto_vel_adjust < 1.0:
                    print(f"   altoVelAdjust: {round(self.alto_vel_adjust, 3)}")
                else:
                    print('')

    def select_new_time_signature(self):
        """
        Randomly selects a new time signature close to the current one and adjusts settings accordingly.
        """
        # Get durations of available time splits
        time_split_durs = np.array([split[0] for split in self.time_splits])
        # Find indices of time splits within +/-5 beats of the current duration
        possible_new_indices = np.where(abs(time_split_durs - self.chord_sequence_duration) < 6)[0]
        # Randomly select a new time split index
        new_sequence_index = possible_new_indices[m.floor(random() * len(possible_new_indices))]

        # Update the chord sequence duration and subdivisions
        self.chord_sequence_duration = self.time_splits[new_sequence_index][0]
        self.sig_div_count = self.time_splits[new_sequence_index][1]
        self.time_delay = TIME_DIV_CONSTANT / self.chord_sequence_duration  # Adjust time delay

        # Randomly change settings for the alto and bass voices
        self.alto_setting = random.choice(ALTO_SETTING_OPTIONS)
        self.bass_setting = random.choice(BASS_SETTING_OPTIONS[:-1])  # Exclude 'none' for variation

        # Randomly adjust thresholds and legato settings
        self.alto_threshold = 0.01 + 0.5 * random() * random()
        self.lead_legato = 0.4 + random() * 0.4 + random() * 0.3

        self.print_sequence()  # Print the updated sequence settings

    def process_bass(self):
        """
        Determines when and how the bass note should be played based on the current settings and timing.
        """
        bass_vel = -1  # Default to not playing
        notes_since_change = self.notes_since_chord_change
        sig_div = self.sig_div_count
        chord_dur = self.chord_sequence_duration

        bass_note = self.chord_sequence[self.current_chord_index][0] - 24  # Current bass note

        # Decide on bass behavior based on the bass setting
        if self.bass_setting == "hold":
            if notes_since_change == 0:
                bass_vel = 100  # Play bass note at the start of the chord

        elif self.bass_setting == "clock":
            if notes_since_change == 0 or notes_since_change % sig_div == 0:
                bass_vel = 60  # Play bass note on every beat

        elif self.bass_setting == "strike":
            if notes_since_change == 0:
                bass_vel = 120  # Emphasize bass note at the start
            elif notes_since_change % sig_div == 0:
                bass_vel = -2   # Silence bass note on beats

        elif self.bass_setting.startswith("div_"):
            division = int(self.bass_setting.split('_')[1])  # Get the division number
            if notes_since_change == 0 or notes_since_change % (sig_div * division) == 0:
                bass_vel = 70  # Play bass note at specified divisions

        elif self.bass_setting == "none":
            bass_vel = -2  # Do not play bass note

        if bass_vel >= 0:
            # Stop previous bass note before playing a new one
            self.midi_handler.send_note_off(self.midi_handler.bass_out, bass_note)

        if bass_vel > 0:
            # Play the new bass note with the specified velocity
            self.midi_handler.send_note_on(self.midi_handler.bass_out, bass_note, velocity=bass_vel)

        # Update the bass history buffer
        self.bass_history.push(notes=bass_note, velocities=bass_vel)

    def process_lead(self):
        """
        Generates and plays notes for the lead (melody) voice based on probabilities and musical considerations.
        """
        previous_note = self.current_note  # Store the previous note for reference

        # Determine the odds of playing the next note
        play_note_odds = self.calculate_play_note_odds()

        # Decide whether to play a note based on the calculated odds
        do_play_note = play_note_odds > 0.3 + (self.lead_legato - 0.4) / 4

        # Calculate the adjusted priorities for the current chord's notes
        foo_priority = self.calculate_note_priorities(previous_note)

        # Randomize velocity occasionally for dynamic variation
        if random() > 0.95:
            self.lead_vel_sum = random() * 127
            self.alto_vel_sum = self.lead_vel_sum

        if do_play_note:
            # Select a note based on the adjusted priorities
            selected_note, note_vel = self.select_lead_note(foo_priority)
            # Play the selected note
            self.play_lead_note(previous_note, selected_note, note_vel)
        else:
            # Decide whether to hold or end the current note
            self.hold_or_end_lead_note()

    def calculate_play_note_odds(self):
        """
        Calculates the probability of playing a note at the current tick based on various musical factors.
        """
        play_note_odds = random()  # Start with a random base probability
        notes_since_change = self.notes_since_chord_change
        sig_div = self.sig_div_count

        if notes_since_change == 0:
            play_note_odds *= 20  # Increase odds at the start of a chord
        elif notes_since_change % sig_div == 0:
            play_note_odds *= 5   # Increase odds on the beat

        # Adjust odds based on gap (rests) and held notes
        if self.gap_count > 10:
            play_note_odds *= self.gap_count - 8  # Encourage playing after long rest
        elif 0 < self.gap_count < 4:
            play_note_odds /= 2   # Reduce odds shortly after a rest
        elif 4 < self.gap_count < 8:
            play_note_odds /= 1.5

        if self.gap_count == 0:
            play_note_odds *= 1.0 - pow(0.95, self.held_count + 1)  # Adjust based on note hold duration

        # Increase odds if a note was played in the previous sequence
        if self.lead_history.velocities[self.chord_sequence_duration - 1] > 0:
            play_note_odds *= 20 * self.vibes_lead_sequence_rep

        return play_note_odds

    def calculate_note_priorities(self, previous_note):
        """
        Adjusts the priorities of the current chord's notes based on historical data and musical considerations.
        """
        # Initialize note range odds to 1.0 for all notes in the range
        note_range_odds = np.full(NOTE_RANGE_MAX - NOTE_RANGE_MIN, 1.0, np.float16)
        note_range_set = np.arange(NOTE_RANGE_MIN, NOTE_RANGE_MAX)
        curr_chord = self.current_note_set.notes
        curr_priority = self.current_note_set.priorities

        # Adjust odds based on the previous note and historical movements
        compare_index = self.chord_sequence_duration - 1
        if self.lead_history.velocities[compare_index] > 0:
            self.adjust_note_range_odds(note_range_odds, note_range_set, compare_index, previous_note)

        # Apply the adjusted odds to the current chord's priorities
        foo_priority = deepcopy(curr_priority)
        for ii in range(len(foo_priority)):
            match_index = np.where(curr_chord[ii] == note_range_set)[0][0]
            foo_priority[ii] *= note_range_odds[match_index]

        # Further adjust priorities based on melodic motion and position in the scale
        for ii in range(len(curr_chord)):
            if self.notes_since_chord_change != 0:
                # Prefer notes closer to the expected motion from previous note
                foo_priority[ii] *= pow(0.9, abs(curr_chord[ii] - (self.scale_motion_velocity + previous_note)))
            # Slightly discourage notes far from the center of the scale to maintain melodic cohesion
            foo_priority[ii] /= pow(abs(len(foo_priority) / 2 - ii) + len(foo_priority), 2)

        # Normalize priorities to sum to 1
        priority_sum = sum(foo_priority)
        foo_priority /= priority_sum

        return foo_priority

    def adjust_note_range_odds(self, note_range_odds, note_range_set, compare_index, previous_note):
        """
        Adjusts the odds of selecting certain notes based on previous notes and movements to encourage musical patterns.
        """
        lead_hist_notes = self.lead_history.notes
        lead_hist_moves = self.lead_history.moves

        # Boost the odds for notes that match the previous note or its octaves
        note_match = np.where((note_range_set - lead_hist_notes[compare_index]) % 12 == 0)[0]
        if len(note_match) > 0:
            note_range_odds[note_match] *= 2 * self.vibes_lead_note_rep + 2

        # Boost the odds for the exact note played in the previous sequence
        note_match_exact = np.where(note_range_set == lead_hist_notes[compare_index])[0]
        if len(note_match_exact) > 0:
            note_range_odds[note_match_exact] *= 3 * self.vibes_lead_note_rep + 2

        # Adjust odds based on the similarity of motion to previous movements
        motion_diff = abs(note_range_set - previous_note - lead_hist_moves[compare_index])
        note_range_odds *= pow(1.0 - self.vibes_lead_motion_rep / 5, motion_diff)

        # Increase odds for notes in the same direction of previous motion
        if lead_hist_moves[compare_index] > 0:
            note_range_odds[note_range_set > previous_note] *= 5
        elif lead_hist_moves[compare_index] < 0:
            note_range_odds[note_range_set < previous_note] *= 5

    def select_lead_note(self, foo_priority):
        """
        Selects a lead note based on the adjusted priorities using a probability distribution.
        """
        priority_cumsum = np.cumsum(foo_priority)  # Cumulative sum for probability distribution
        random_value = random()                    # Random number between 0 and 1
        selected_pos = np.searchsorted(priority_cumsum, random_value)  # Find the selected note index
        selected_note = self.current_note_set.notes[selected_pos]      # Get the MIDI note number

        # Calculate the velocity for the selected note
        note_vel = int((self.current_note_set.priorities[selected_pos] > 0.5) * self.lead_vel_sum / 4 +
                       30 * random() + 30 * (self.notes_since_chord_change % self.sig_div_count == 0) + 30)
        # Smooth the velocity over time for natural dynamics
        self.lead_vel_sum = (note_vel + self.lead_vel_sum * 3) / 4

        return selected_note, note_vel

    def play_lead_note(self, previous_note, current_note, note_vel):
        """
        Sends MIDI messages to play the lead note and updates the relevant histories and counters.
        """
        # Stop the previous note
        self.midi_handler.send_note_off(self.midi_handler.arpp_out, previous_note)
        # Play the current note
        self.midi_handler.send_note_on(self.midi_handler.arpp_out, current_note, velocity=note_vel)
        # Update the melodic motion velocity
        self.scale_motion_velocity = float(current_note - previous_note)
        # Reset gap and held counts as a new note is played
        self.gap_count = 0
        self.held_count = 0
        # Update the lead history buffers
        self.lead_history.push(notes=current_note, moves=self.scale_motion_velocity, velocities=note_vel)
        self.current_note = current_note  # Update the current note

    def hold_or_end_lead_note(self):
        """
        Decides whether to hold or end the current lead note based on legato settings and randomness.
        """
        previous_note = self.current_note
        # Decide whether to end the note based on legato probability and other factors
        if random() > self.lead_legato - 0.05 * self.gap_count - \
                0.5 * ((self.current_note - NOTE_RANGE_MIN) / (NOTE_RANGE_MAX - NOTE_RANGE_MIN) > 0.8):
            # End the note
            self.midi_handler.send_note_off(self.midi_handler.arpp_out, previous_note)
            self.gap_count += 1  # Increase gap count as a rest occurs
        else:
            # Hold the note
            if self.gap_count > 0:
                self.gap_count += 1
            else:
                self.held_count += 1  # Increase held count as the note continues

        # Update the lead history with appropriate velocity indicating note hold or rest
        note_vel = -2 if self.gap_count > 0 else -1
        self.lead_history.push(notes=self.current_note, velocities=note_vel)
        # Decay the motion velocity over time when holding notes
        self.scale_motion_velocity /= 1.5

    def process_alto(self):
        """
        Generates and plays notes for the alto (middle) voice based on the current setting and musical context.
        """
        # Calculate the probability of playing an alto note
        play_alto_odds = self.calculate_play_alto_odds()
        # Decide whether to play an alto note
        do_play_alto = play_alto_odds > self.alto_threshold
        alto_misc_data_pt = 0.0  # Placeholder for miscellaneous data

        if do_play_alto:
            # Play an alto note based on the current setting
            self.play_alto_note()
        else:
            note_vel_alto = -2  # Indicate that the note is not played
            # Update the alto history buffer
            self.alto_history.push(velocities=note_vel_alto, misc=alto_misc_data_pt)

    def calculate_play_alto_odds(self):
        """
        Calculates the probability of playing an alto note based on timing and historical data.
        """
        play_alto_odds = random()  # Start with a random base probability
        notes_since_change = self.notes_since_chord_change
        sig_div = self.sig_div_count

        if notes_since_change == 0:
            play_alto_odds *= 10  # Increase odds at the start of a chord
        elif notes_since_change % sig_div == 0:
            play_alto_odds *= 3   # Increase odds on the beat

        # Increase odds if an alto note was played in the previous sequence
        if self.alto_history.velocities[self.chord_sequence_duration - 1] > 0:
            play_alto_odds *= 10

        # Increase odds if an alto note was just played
        if self.alto_history.velocities[0] > 0:
            play_alto_odds *= 2

        return play_alto_odds

    def play_alto_note(self):
        """
        Determines and plays an alto note based on the current alto setting and musical context.
        """
        prev_alto_note = self.alto_note  # Store the previous alto note
        # Calculate the velocity for the alto note with some randomness
        note_vel_alto = int(self.alto_vel_adjust * ((random() / 2 + 0.5) *
                                                    (self.alto_vel_sum / 2 * (self.current_note_set.priorities[0] > 0.5) + 30) + 30))

        alto_misc_data_pt = 0.0  # Placeholder for miscellaneous data
        # Determine the alto note based on the current setting
        alto_note = self.determine_alto_note()

        # Smooth the alto velocity over time
        self.alto_vel_sum = (note_vel_alto + self.alto_vel_sum * 3) / 4

        # Stop the previous alto note
        self.midi_handler.send_note_off(self.midi_handler.alto_out, prev_alto_note)
        # Play the new alto note
        self.midi_handler.send_note_on(self.midi_handler.alto_out, alto_note, velocity=note_vel_alto)
        # Update the alto history buffers
        self.alto_history.push(notes=alto_note, velocities=note_vel_alto, misc=alto_misc_data_pt)
        self.alto_note = alto_note  # Update the current alto note

    def determine_alto_note(self):
        """
        Determines the alto note to play based on the current alto setting and musical context.
        """
        curr_chord = self.current_note_set.notes
        curr_priority = self.current_note_set.priorities
        alto_setting = self.alto_setting
        notes_since_change = self.notes_since_chord_change
        sig_div = self.sig_div_count

        if alto_setting == 'match':
            # Alto matches the lead voice's note selection
            selected_pos = self.select_position_based_on_priority(curr_priority)
            alto_note = curr_chord[selected_pos] - 12  # One octave below the lead

        elif alto_setting == 'random':
            # Alto selects a random note from high-priority notes
            potential_matches = np.where(curr_priority > 0.6)[0]
            selected_pos = m.floor(random() * len(potential_matches))
            alto_note = curr_chord[potential_matches[selected_pos]] - 12

        elif alto_setting == 'pulse':
            # Alto plays a pulsing rhythm, emphasizing certain beats
            if notes_since_change == 0:
                potential_matches = np.where(curr_priority >= 0.8)[0]
                selected_pos = m.floor(random() * len(potential_matches))
                alto_note = curr_chord[potential_matches[selected_pos]]
            else:
                alto_note = self.alto_note  # Continue holding the note

        elif alto_setting == 'decay':
            # Alto note decays downward over time
            alto_note = self.alto_decay(curr_chord, curr_priority)

        elif alto_setting == 'arpeggio':
            # Alto plays an arpeggiated pattern
            alto_note = self.alto_arpeggio(curr_chord, curr_priority)

        else:
            alto_note = self.alto_note  # Default to current alto note

        return alto_note

    def select_position_based_on_priority(self, priorities):
        """
        Selects a note index based on the cumulative priorities for probabilistic note selection.
        """
        priority_cumsum = np.cumsum(priorities)
        random_value = random()
        selected_pos = np.searchsorted(priority_cumsum, random_value)
        return selected_pos

    def alto_decay(self, curr_chord, curr_priority):
        """
        Implements the 'decay' behavior for the alto voice, moving the note downward over time.
        """
        random_value = random()
        potential_matches = np.where(curr_priority > 0.6)[0]
        potential_indices = np.where(curr_chord[potential_matches] < self.alto_note)[0]

        if self.notes_since_chord_change == 0:
            random_value *= 2
        elif self.notes_since_chord_change % self.sig_div_count == 0:
            random_value *= 6 / 5

        if self.alto_history.misc[self.chord_sequence_duration - 1] == 1.0:
            random_value *= 4 / 3

        if len(potential_indices) > 3 and random_value < 1 / 12:
            alto_note = curr_chord[potential_matches[potential_indices[-3]]]
        elif len(potential_indices) > 2 and random_value < 3 / 12:
            alto_note = curr_chord[potential_matches[potential_indices[-2]]]
        elif len(potential_indices) > 1 and random_value < 1:
            alto_note = curr_chord[potential_matches[potential_indices[-1]]]
        else:
            alto_note = self.current_note  # Match the lead note
            self.alto_history.misc[0] = 1.0

        return alto_note

    def alto_arpeggio(self, curr_chord, curr_priority):
        """
        Implements the 'arpeggio' behavior for the alto voice, creating ascending or descending patterns.
        """
        if self.notes_since_chord_change == 0:
            # At the start of a chord, possibly change arpeggio direction
            if self.alto_arp_hit_end:
                self.alto_arp_hit_end = not self.alto_arp_hit_end
            else:
                self.alto_arp_up = not self.alto_arp_up

        # Determine direction changes during arpeggio
        self.determine_arpeggio_direction()

        potential_indices = np.where(curr_priority > 0.7)[0]
        potential_notes = curr_chord[potential_indices]
        prev_alto_note = self.alto_note

        if self.alto_arp_up:
            # Arpeggio is ascending
            higher_notes = potential_notes[potential_notes > prev_alto_note]
            if higher_notes.size > 0:
                alto_note = higher_notes[0]
            else:
                # Change direction at the top
                self.alto_arp_hit_end = True
                self.alto_arp_up = False
                lower_notes = potential_notes[potential_notes < prev_alto_note]
                alto_note = lower_notes[-1]
        else:
            # Arpeggio is descending
            lower_notes = potential_notes[potential_notes < prev_alto_note]
            if lower_notes.size > 0:
                alto_note = lower_notes[-1]
            else:
                # Change direction at the bottom
                self.alto_arp_hit_end = True
                self.alto_arp_up = True
                higher_notes = potential_notes[potential_notes > prev_alto_note]
                alto_note = higher_notes[0]

        return alto_note

    def determine_arpeggio_direction(self):
        """
        Decides whether to change the direction of the arpeggio based on randomness and timing.
        """
        if self.alto_history.misc[0] == -4.0:
            if self.alto_arp_up and random() < 0.8:
                self.alto_arp_hit_end = True
                self.alto_arp_up = False
            elif random() < 0.5:
                self.alto_arp_hit_end = True
                self.alto_arp_up = True
        elif self.alto_history.misc[0] == 4.0:
            if self.alto_arp_up and random() < 0.8:
                self.alto_arp_hit_end = True
                self.alto_arp_up = True
            elif random() < 0.5:
                self.alto_arp_hit_end = True
                self.alto_arp_up = False
        elif (self.notes_since_chord_change % self.sig_div_count == 0 and random() < 0.15) or random() < 0.05:
            self.alto_arp_hit_end = True
            self.alto_arp_up = not self.alto_arp_up

if __name__ == "__main__":
    # Instantiate the music generator and start the main loop
    generator = ProceduralMusicGenerator()
    generator.run()
