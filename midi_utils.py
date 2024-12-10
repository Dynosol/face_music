import mido
import time
import MIDI_Funcs  # Ensure this module is accessible

class MidiHandler:
    """
    Handles MIDI output ports and sending MIDI messages.
    """

    def __init__(self):
        # Open MIDI output ports for melody and choir
        self.drums_out = mido.open_output('Your Drum MIDI Port Name')  # Update with your port name
        self.melody_out = mido.open_output('IAC Driver Bus 1')      # Melody output port
        self.choir_out1 = mido.open_output('IAC Driver Bus 2')      # Choir output port 1
        self.choir_out2 = mido.open_output('IAC Driver Bus 3')      # Choir output port 2
        self.choir_out3 = mido.open_output('IAC Driver Bus 4')      # Choir output port 3
        self.brass_out = mido.open_output('IAC Driver Bus 5')       # Brass output port
        self.organ_out = mido.open_output('IAC Driver Bus 6')       # Organ output port
        self.outport_sets = [
            self.melody_out, self.choir_out1, self.choir_out2,
            self.choir_out3, self.brass_out, self.organ_out
        ]  # List of all ports
        # Ensure clean exit by turning off any lingering notes
        MIDI_Funcs.niceMidiExit(self.outport_sets)  # Call MIDI cleanup function

    def send_note_on(self, port, note, velocity=100, channel=0):
        """Sends a 'note_on' MIDI message to the specified port."""
        port.send(mido.Message('note_on', note=int(note), velocity=int(velocity), channel=channel))

    def send_note_off(self, port, note, channel=0):
        """Sends a 'note_off' MIDI message to the specified port."""
        port.send(mido.Message('note_off', note=int(note), channel=channel))

    def close_ports(self):
        """Closes all MIDI ports and ensures all notes are turned off."""
        for outport in self.outport_sets:
            outport.send(mido.Message('note_off', note=0, velocity=0))
            outport.close()

    def send_drum_notes(self, midi_out, drum_notes, velocity, duration=0.05):
        for drum_note in drum_notes:
            midi_out.send(mido.Message('note_on', note=drum_note, velocity=velocity, channel=9))
        time.sleep(duration)
        for drum_note in drum_notes:
            midi_out.send(mido.Message('note_off', note=drum_note, channel=9))
