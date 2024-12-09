import mido
import time
import numpy as np
from random import random

# Open a single MIDI output port (e.g., a virtual MIDI bus)
MIDI_OUT = mido.open_output('IAC Driver Bus 1')

# Define scale intervals for different musical scales
# Each scale is represented by a set of semitone intervals from the root note
SCALE_SET = {
    'MAJOR': [0, 2, 4, 5, 7, 9, 11],       # Major scale
    'MINOR': [0, 2, 3, 5, 7, 8, 10],       # Natural minor scale
    'PENT_MAJOR': [0, 2, 4, 7, 9],         # Pentatonic major scale
    'PENT_MINOR': [0, 2, 3, 7, 9],         # Pentatonic minor scale
}

# Define the minimum and maximum note range for the MIDI output
NOTE_RANGE_MIN = 50  # MIDI note number 50 corresponds to D3
NOTE_RANGE_MAX = 80  # MIDI note number 80 corresponds to G#5
BPM = 120            # Beats per minute for the tempo

# Calculate the time delay between notes based on the BPM
# The time delay represents half notes per beat
# This determines the rhythmic feel of the generated sequence
TIME_DELAY = 60 / BPM / 2

# Function to generate a set of notes based on a root note and scale
def get_note_set(base_note, scale_name='MAJOR'):
    interval_set = SCALE_SET[scale_name]  # Get the intervals for the given scale
    # Bring the base note down within the range if it's too high
    while base_note >= NOTE_RANGE_MIN:
        base_note -= 12  # Decrease by an octave
    out_notes = []
    # Loop through and add notes in range based on intervals in the given scale
    while base_note < NOTE_RANGE_MAX:
        for interval in interval_set:
            note = base_note + interval
            if NOTE_RANGE_MIN <= note < NOTE_RANGE_MAX:
                out_notes.append(note)  # Add the note if it's within the allowed range
        base_note += 12  # Move up by an octave
    return np.array(out_notes)

# Define a chord sequence to be played (root note and scale)
# Each chord is defined by a root note and scale type
CHORD_SEQUENCE = [
    # [50, 'PENT_MINOR'],  # Dm chord (Pentatonic Minor) 
    # [55, 'MAJOR'],       # G Major chord 
    [60, 'MAJOR'],       # C Major chord
    # [57, 'MINOR'],       # A Minor chord 
]

# Define the number of notes to play before changing chords
CHORD_SEQUENCE_DURATION = 30
current_chord_index = 0  # Start with the first chord in the sequence
notes_since_chord_change = 0  # Count of notes played since last chord change
curr_chord = get_note_set(CHORD_SEQUENCE[current_chord_index][0], CHORD_SEQUENCE[current_chord_index][1])

previous_note = None  # Track the previously played note to minimize jumps

# Begin the main loop for generating MIDI notes
try:
    while True:
        start_time = time.time()  # Record the start time for each iteration
        notes_since_chord_change += 1  # Increment the note counter
        if notes_since_chord_change >= CHORD_SEQUENCE_DURATION:
            # Switch to the next chord when the duration is met
            notes_since_chord_change = 0
            current_chord_index = (current_chord_index + 1) % len(CHORD_SEQUENCE)  # Cycle through chords
            curr_chord = get_note_set(CHORD_SEQUENCE[current_chord_index][0], CHORD_SEQUENCE[current_chord_index][1])

        # Determine if a note should be played in this iteration
        play_note_odds = random()  # Random value between 0 and 1
        if play_note_odds > 0.3:  # 70% chance of playing a note
            if previous_note is not None and random() < 0.9:
                # Prefer smaller note jumps when continuing from previous note
                possible_notes = curr_chord[np.abs(curr_chord - previous_note) <= 5]  # Limit jump to within 5 semitones
                if len(possible_notes) > 0:
                    selected_note = np.random.choice(possible_notes)  # Choose from closer notes
                else:
                    selected_note = np.random.choice(curr_chord)  # Fallback to any note in the chord
            else:
                selected_note = np.random.choice(curr_chord)  # Choose any note from the current chord

            # Set the velocity (volume) of the note randomly between 30 and 100
            note_vel = int(random() * 70 + 30)
            MIDI_OUT.send(mido.Message('note_on', note=selected_note, velocity=note_vel))  # Send note on message

            # Randomly determine how long to hold the note (sometimes more than one beat)
            hold_time = TIME_DELAY * (1 if random() < 0.7 else np.random.choice([2, 3, 4]))
            time.sleep(hold_time)  # Hold the note for the determined duration

            MIDI_OUT.send(mido.Message('note_off', note=selected_note))  # Send note off message to stop the note
            previous_note = selected_note  # Update the previously played note

        end_time = time.time()  # Record the end time for the iteration

        # Ensure the total iteration time matches the desired time delay
        if TIME_DELAY - (end_time - start_time) > 0:
            time.sleep(TIME_DELAY - (end_time - start_time))
        else:
            # Print an error message if the generation took too long
            print("TIMING ERROR!!! Not enough time to process generation between ticks")
except (KeyboardInterrupt, SystemExit):
    # Gracefully stop all MIDI output when the user stops the script
    print("Stopping all MIDI output...")
    for note in range(NOTE_RANGE_MIN, NOTE_RANGE_MAX + 1):
        MIDI_OUT.send(mido.Message('note_off', note=note))  # Send note off message for all notes
    MIDI_OUT.close()  # Close the MIDI output port
