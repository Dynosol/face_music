def midi_note_to_name(midi_note):
    if midi_note is None:
        return 'Rest'
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = midi_note % 12
    note_name = note_names[note] + str(octave)
    return note_name
