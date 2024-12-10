import mido
import time
import json

# Constants
SNARE_NOTE = 48
BASS_NOTE = 50
HI_HAT_NOTE = 45
TOM_NOTE = 47
VELOCITY = 100

# Default Drum Configuration
DRUM_BEATS = {
    "BASS": {"beats": {1, 3}, "note": BASS_NOTE},  # Hits on beats 1 and 3
    "SNARE": {"beats": {2, 4}, "note": SNARE_NOTE},  # Hits on beats 2 and 4
    "TOM": {"beats": {4}, "note": TOM_NOTE},  # Hits on beat 4
    "HI-HAT": {"beats": {1.5, 2.5, 3.5, 4.5}, "note": HI_HAT_NOTE},  # Includes upbeats as regular beats
}
BPM = 120

BEAT_DURATION = 60 / BPM
HALFBEAT_DELAY = 0.08
UPBEAT_DURATION = BEAT_DURATION / 2 - HALFBEAT_DELAY

MIDI_OUT = mido.open_output('IAC Driver Bus 10')

CONFIG_FILE = "config.json"

def load_config():
    """Load BPM and drum configuration from a JSON file."""
    global BPM, BEAT_DURATION, UPBEAT_DURATION, DRUM_BEATS
    try:
        with open(CONFIG_FILE, "r") as file:
            config = json.load(file)
            # Update BPM and calculate beat durations
            BPM = config.get("BPM", BPM)
            BEAT_DURATION = 60 / BPM
            UPBEAT_DURATION = BEAT_DURATION / 2 - HALFBEAT_DELAY
            # Update drum beats
            DRUM_BEATS = config.get("DRUM_BEATS", DRUM_BEATS)
            for _, data in DRUM_BEATS.items():
                if "beats" in data and isinstance(data["beats"], list):
                    data["beats"] = set(data["beats"])
    except FileNotFoundError:
        print("Config file not found, using default settings.")
    except json.JSONDecodeError as e:
        print(f"Error decoding config file: {e}")


def send_notes(midi_out, notes, velocity, duration=0.05):
    for note in notes:
        midi_out.send(mido.Message('note_on', note=note, velocity=velocity))
    time.sleep(duration)
    for note in notes:
        midi_out.send(mido.Message('note_off', note=note))

def play_rhythm(midi_out):
    current_beat = 1
    while True:
        load_config()  # Reload drum configuration dynamically
        beat_start = time.time()

        # Handle main beats
        notes_to_play = []
        hit_logs = []

        for drum, data in DRUM_BEATS.items():
            # Check for beats
            if current_beat in data["beats"]:
                notes_to_play.append(data["note"])
                hit_logs.append(drum)

        # Play main beats
        if notes_to_play:
            print(f"Beat {current_beat}: {', '.join(hit_logs)} hit.")
            send_notes(midi_out, notes_to_play, VELOCITY)

        # Handle upbeats
        for drum, data in DRUM_BEATS.items():
            if current_beat + 0.5 in data["beats"]:
                time.sleep(UPBEAT_DURATION)
                print(f"Upbeat after Beat {current_beat}: {drum} hit.")
                send_notes(midi_out, [data["note"]], VELOCITY)

        # Wait for the rest of the beat
        time.sleep(max(0, BEAT_DURATION - (time.time() - beat_start)))
        
        # Move to the next beat
        current_beat += 1
        if current_beat > 4:
            current_beat = 1

def main():
    try:
        play_rhythm(MIDI_OUT)
    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    finally:
        MIDI_OUT.close()
        print("MIDI port closed.")

if __name__ == "__main__":
    main()
