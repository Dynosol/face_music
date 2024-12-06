import mido
from mido import Message, MidiFile, MidiTrack, Backend
import time

print(mido.get_output_names())

# Set up a MIDI output port
midi_output = mido.open_output('IAC Driver Bus 1')

# Create a basic function to send MIDI notes

def send_midi_note(note=60, velocity=64, channel=0, duration=1.0):
    # Note On
    note_on = Message('note_on', note=note, velocity=velocity, channel=channel)
    midi_output.send(note_on)

    # Hold the note for the given duration
    time.sleep(duration)

    # Note Off
    note_off = Message('note_off', note=note, velocity=velocity, channel=channel)
    midi_output.send(note_off)

# Send a test note
send_midi_note(note=60, velocity=100, duration=1.0)

# Close the MIDI port
midi_output.close()