import json

# Constants defining the MIDI note range
NOTE_RANGE_MIN = 24  # Lowest MIDI note to use (C1, MIDI note number 24)
NOTE_RANGE_MAX = 87  # Highest MIDI note to use (D#6, MIDI note number 87)

# Define BPM and timing settings
BPM = 120           # Set the tempo of the music to 120 beats per minute
TICKS_PER_BEAT = 4   # Number of ticks per beat (representing sixteenth notes)
BEATS_PER_MEASURE = 4  # Number of beats per measure (standard 4/4 time signature)
CONFIG_FILE = "config.json"

# Calculate derived timing constants
SECONDS_PER_BEAT = 60.0 / BPM  # Duration of one beat in seconds
TIME_DELAY = SECONDS_PER_BEAT / TICKS_PER_BEAT  # Delay between ticks (duration of one tick)
MEASURE_DURATION = BEATS_PER_MEASURE * TICKS_PER_BEAT  # Total number of ticks in one measure

# Constants
VELOCITY = 100

# Default Drum Configuration
DRUM_BEATS = {
    "BASS": {"beats": {1, 3}, "note": "BASS_NOTE"},  # Will be replaced in load_config
    "SNARE": {"beats": {2, 4}, "note": "SNARE_NOTE"},  # Will be replaced in load_config
    "TOM": {"beats": {}, "note": "TOM_NOTE"},
    "HI-HAT": {"beats": {1.5, 2.5, 3.5, 4.5}, "note": "HI_HAT_NOTE"},
}

def load_config(N):
    """Load BPM and drum configuration from a JSON file."""
    global BPM, SECONDS_PER_BEAT, TIME_DELAY, MEASURE_DURATION, DRUM_BEATS
    try:
        with open(CONFIG_FILE, "r") as file:
            config = json.load(file)
            # Update BPM and calculate beat durations
            BPM = config.get("BPM", BPM)
            SECONDS_PER_BEAT = 60.0 / BPM
            TIME_DELAY = SECONDS_PER_BEAT / TICKS_PER_BEAT
            MEASURE_DURATION = BEATS_PER_MEASURE * TICKS_PER_BEAT

            # Update drum beats
            DRUM_BEATS = config.get("DRUM_BEATS", DRUM_BEATS)
            for _, data in DRUM_BEATS.items():
                if "beats" in data and isinstance(data["beats"], list):
                    data["beats"] = set(data["beats"])
                if isinstance(data["note"], str):
                    data["note"] = N.get(data["note"], 36)  # Default to Bass Drum (note 36)
    except FileNotFoundError:
        print("Config file not found, using default settings.")
    except json.JSONDecodeError as e:
        print(f"Error decoding config file: {e}")
