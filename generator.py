import time
from random import random, choice
import numpy as np

from config import (
    BPM, TICKS_PER_BEAT, BEATS_PER_MEASURE, TIME_DELAY, MEASURE_DURATION,
    VELOCITY, DRUM_BEATS
)
from midi_utils import MidiHandler
from music_theory import (
    NoteSet, CHORD_SEQUENCE, CHORD_INTERVALS, N, NOTE_RANGE_MIN, NOTE_RANGE_MAX
)
from history import HistoryBuffer
from utils import midi_note_to_name

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
        self.ticks_per_beat = TICKS_PER_BEAT
        self.beats_per_measure = BEATS_PER_MEASURE
        self.measure_duration = MEASURE_DURATION
        self.time_delay = TIME_DELAY
        self.chord_sequence_duration = self.measure_duration

        # Initialize history buffers for each instrument
        self.history_size = self.measure_duration * 3
        self.melody_history = HistoryBuffer(self.history_size)
        self.choir_history = HistoryBuffer(self.history_size)
        self.brass_history = HistoryBuffer(self.history_size)
        self.organ_history = HistoryBuffer(self.history_size)

        # Initialize velocity sums for dynamics
        self.melody_vel_sum = 100
        self.choir_vel_sum = 100
        self.brass_vel_sum = 100
        self.organ_vel_sum = 100

        # Initialize the first chord and note sets for each voice
        self.initialize_note_sets()

        # Initialize the first note for each instrument
        self.melody_note = int(self.melody_note_set.notes[0])
        # Initialize previous choir notes with root, third, and fifth of the first chord
        self.choir_notes = [
            self.chord_sequence[0][0] + interval
            for interval in CHORD_INTERVALS[self.chord_sequence[0][1]]
        ]
        self.brass_note = None  # Start with no note
        self.organ_note = int(self.organ_note_set.notes[0])

        self.drum_beats = DRUM_BEATS

    def initialize_note_sets(self):
        """Initializes separate NoteSets for each voice with appropriate ranges based on current chord."""
        base_note, scale_name = self.chord_sequence[self.current_chord_index]

        # Melody Voice
        melody_min_note = max(base_note + 12, NOTE_RANGE_MIN)
        melody_max_note = min(base_note + 36, NOTE_RANGE_MAX)
        self.melody_note_set = NoteSet(base_note, scale_name, min_note=melody_min_note, max_note=melody_max_note)

        # Choir Voice
        choir_min_note = max(base_note + 12, NOTE_RANGE_MIN)
        choir_max_note = min(base_note + 36, NOTE_RANGE_MAX)
        self.choir_note_set = NoteSet(base_note, scale_name, min_note=choir_min_note, max_note=choir_max_note)

        # Brass Voice
        brass_min_note = max(base_note - 12, NOTE_RANGE_MIN)
        brass_max_note = min(base_note + 12, NOTE_RANGE_MAX)
        self.brass_note_set = NoteSet(base_note, scale_name, min_note=brass_min_note, max_note=brass_max_note)

        # Organ Voice
        organ_min_note = max(base_note + 12, NOTE_RANGE_MIN)
        organ_max_note = min(base_note + 24, NOTE_RANGE_MAX)
        self.organ_note_set = NoteSet(base_note, scale_name, min_note=organ_min_note, max_note=organ_max_note)

    def run(self):
        """
        Main loop that runs the procedural music generation.
        """
        try:
            while True:
                start_time = time.time()
                self.current_tick += 1

                # Handle chord changes
                self.handle_chord_changes()

                # Process each instrument
                self.process_melody()
                self.process_choir()
                self.process_brass()
                self.process_organ()
                self.process_drums()

                # Print out the current state
                self.print_current_state()

                end_time = time.time()
                sleep_time = self.time_delay - (end_time - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print("TIMING ERROR!!! Not enough time to process generation between ticks")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected, stopping gracefully...")
            self.midi_handler.close_ports()
            print("All MIDI ports have been closed successfully.")

    def handle_chord_changes(self):
        """
        Manages chord progression.
        """
        self.notes_since_chord_change += 1
        if self.notes_since_chord_change >= self.chord_sequence_duration:
            self.notes_since_chord_change = 0
            self.current_chord_index = (self.current_chord_index + 1) % len(self.chord_sequence)
            self.initialize_note_sets()

    def process_melody(self):
        """
        Generates and plays notes for the melody (solo) voice.
        """
        previous_note = self.melody_note  # Previous note for reference
        notes_since_change = self.notes_since_chord_change  # Ticks since last chord change

        play_note_odds = random()  # Base probability to play a note

        density_factor = 0.6       # Higher values increase the likelihood of playing a note
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

    def process_drums(self):
        """
        Processes and plays drum notes based on the current tick and drum configuration.
        """
        current_beat = ((self.current_tick // self.ticks_per_beat) % self.beats_per_measure) + 1
        tick_in_beat = self.current_tick % self.ticks_per_beat

        notes_to_play = []
        hit_logs = []

        if tick_in_beat == 0:
            # Downbeat
            for drum, data in self.drum_beats.items():
                if current_beat in data["beats"]:
                    notes_to_play.append(data["note"])
                    hit_logs.append(drum)
        elif tick_in_beat == self.ticks_per_beat // 2:
            # Upbeat
            for drum, data in self.drum_beats.items():
                if (current_beat + 0.5) in data["beats"]:
                    notes_to_play.append(data["note"])
                    hit_logs.append(drum)

        if notes_to_play:
            self.midi_handler.send_drum_notes(self.midi_handler.drums_out, notes_to_play, VELOCITY)
            print(f"Beat {current_beat}, Tick {tick_in_beat}: {' and '.join(hit_logs)} hit.")
