import mido
import time
import numpy as np
import threading
import json
import os
from random import random

# Load the configuration
CONFIG_FILE = "config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        # Default configuration
        config = {
            "BPM": 120,
            "CHORD_SEQUENCE": [
                [60, 'MAJOR'],  # C Major
                [62, 'MINOR'],  # D Minor
                [64, 'MINOR'],  # E Minor
                [65, 'MAJOR'],  # F Major,
            ],
            "VOICES": {
                "SOLO": {"active": True},
                "CHOIR": {"active": True},
                "BRASS": {"active": True},
                "ORGAN": {"active": True},
            }
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    else:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    return config

class MusicGenerator:
    def __init__(self):
        self.config = load_config()
        self.bpm = self.config.get("BPM", 120)
        self.chord_sequence = self.config.get("CHORD_SEQUENCE", [
            [60, 'MAJOR'],  # Default chord sequence
            [62, 'MINOR'],
            [64, 'MINOR'],
            [65, 'MAJOR'],
        ])
        self.voices = self.config.get("VOICES", {})
        self.scale_set = {
            'MAJOR': [0, 2, 4, 5, 7, 9, 11],
            'MINOR': [0, 2, 3, 5, 7, 8, 10],
            'HARMONIC_MINOR': [0, 2, 3, 5, 7, 8, 11],
            'PENTATONIC_MAJOR': [0, 2, 4, 7, 9],
            'PENTATONIC_MINOR': [0, 3, 5, 7, 10],
        }
        self.note_range_min = 48  # C3
        self.note_range_max = 84  # C6
        self.time_delay = 60 / self.bpm / 2  # Half beat delay
        self.stop_event = threading.Event()

        # Initialize MIDI outputs
        self.midi_out_solo = mido.open_output('IAC Driver Bus 1')
        self.midi_out_choir = mido.open_output('IAC Driver Bus 2')
        self.midi_out_brass = mido.open_output('IAC Driver Bus 3')
        self.midi_out_organ = mido.open_output('IAC Driver Bus 4')

        self.midi_outputs = {
            'SOLO': self.midi_out_solo,
            'CHOIR': self.midi_out_choir,
            'BRASS': self.midi_out_brass,
            'ORGAN': self.midi_out_organ,
        }

        # Variables for music generation
        self.current_chord_index = 0
        self.previous_notes = {voice: None for voice in self.voices}
        self.lock = threading.Lock()
        self.chord_sequence_duration = 16  # Number of notes per chord

    def get_note_set(self, base_note, scale_name='MAJOR'):
        interval_set = self.scale_set.get(scale_name.upper(), self.scale_set['MAJOR'])
        out_notes = []
        base_note -= 12 * ((base_note - self.note_range_min) // 12 + 1)
        while base_note < self.note_range_max:
            for interval in interval_set:
                note = base_note + interval
                if self.note_range_min <= note <= self.note_range_max:
                    out_notes.append(note)
            base_note += 12
        return np.array(out_notes)

    def play_voice(self, voice_name):
        midi_out = self.midi_outputs[voice_name]
        while not self.stop_event.is_set():
            with self.lock:
                # Reload configuration in case it has changed
                self.config = load_config()
                self.voices = self.config.get("VOICES", self.voices)
                active = self.voices[voice_name]['active']
                # Read current chord safely
                current_chord_index = self.current_chord_index
                chord_sequence = self.chord_sequence
                if chord_sequence:
                    try:
                        current_chord = chord_sequence[current_chord_index]
                    except IndexError:
                        # In case current_chord_index is out of range
                        current_chord_index = self.current_chord_index = 0
                        current_chord = chord_sequence[current_chord_index]
                    curr_chord_notes = self.get_note_set(*current_chord)
                else:
                    # No chord sequence, use default
                    curr_chord_notes = self.get_note_set(60, 'MAJOR')

            if active:
                # Generate a note
                previous_note = self.previous_notes.get(voice_name)
                if previous_note is not None and random() < 0.7:
                    possible_notes = curr_chord_notes[np.abs(curr_chord_notes - previous_note) <= 5]
                    if len(possible_notes) == 0:
                        possible_notes = curr_chord_notes
                else:
                    possible_notes = curr_chord_notes

                selected_note = np.random.choice(possible_notes)
                velocity = int(random() * 40 + 80)  # Velocity between 80 and 120

                midi_out.send(mido.Message('note_on', note=selected_note, velocity=velocity))
                self.previous_notes[voice_name] = selected_note

                # Hold the note for a random duration
                hold_time = self.time_delay * np.random.choice([1, 2, 4])
                time.sleep(hold_time)

                midi_out.send(mido.Message('note_off', note=selected_note))
            else:
                # If not active, release any previously held note
                previous_note = self.previous_notes.get(voice_name)
                if previous_note is not None:
                    midi_out.send(mido.Message('note_off', note=previous_note))
                    self.previous_notes[voice_name] = None
                time.sleep(self.time_delay)

    def update_chord_progression(self):
        while not self.stop_event.is_set():
            with self.lock:
                # Reload configuration in case it has changed
                self.config = load_config()
                self.bpm = self.config.get("BPM", 120)
                self.time_delay = 60 / self.bpm / 2
                chord_sequence = self.config.get("CHORD_SEQUENCE", self.chord_sequence)
                if chord_sequence != self.chord_sequence:
                    self.chord_sequence = chord_sequence
                    # Ensure chord_sequence is not empty
                    if not self.chord_sequence:
                        self.chord_sequence = [[60, 'MAJOR']]  # Default chord
                    # Adjust current_chord_index
                    self.current_chord_index %= len(self.chord_sequence)
                else:
                    if not self.chord_sequence:
                        self.chord_sequence = [[60, 'MAJOR']]  # Ensure chord_sequence is not empty
                # Update current chord index
                self.current_chord_index = (self.current_chord_index + 1) % len(self.chord_sequence)
            time.sleep(self.time_delay * self.chord_sequence_duration)

    def run(self):
        # Start chord progression thread
        chord_thread = threading.Thread(target=self.update_chord_progression)
        chord_thread.start()

        # Start threads for each voice
        threads = []
        for voice_name in self.voices:
            thread = threading.Thread(target=self.play_voice, args=(voice_name,))
            thread.start()
            threads.append(thread)

        try:
            while True:
                time.sleep(1)  # Main thread sleeps, threads handle music generation
        except KeyboardInterrupt:
            print("Stopping music generator...")
            self.stop_event.set()
            for thread in threads:
                thread.join()
            chord_thread.join()
            self.cleanup()

    def cleanup(self):
        # Turn off all notes and close MIDI outputs
        for _, midi_out in self.midi_outputs.items():
            for note in range(self.note_range_min, self.note_range_max + 1):
                midi_out.send(mido.Message('note_off', note=note))
            midi_out.close()
        print("MIDI outputs closed.")

if __name__ == "__main__":
    generator = MusicGenerator()
    generator.run()
