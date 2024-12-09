import json
import threading

CONFIG_FILE = "config.json"
config_lock = threading.Lock()  # Thread lock for synchronizing access to config.json

# Initialize variables for drum beat patterns
DEFAULT_DRUM_BEATS = {
    "BASS": [1, 2, 3, 4],
    "SNARE": [1, 2, 3, 4],
    "TOM": [1, 2, 3, 4],
    "HI-HAT": [1, 2, 3, 4]
}

# Patterns for specific drum combinations
DRUM_COMBO_PATTERNS = {
    frozenset([]): {
        "BASS": [],
        "SNARE": [],
        "TOM": [],
        "HI-HAT": []
    },
    frozenset(["SNARE", "BASS"]): {
        "BASS": [1, 3],
        "SNARE": [2, 4],
        "TOM": [],
        "HI-HAT": []
    },
    frozenset(["SNARE", "TOM"]): {
        "BASS": [],
        "SNARE": [2, 4],
        "TOM": [1, 3],
        "HI-HAT": []
    },
    frozenset(["BASS", "TOM"]): {
        "BASS": [1, 3],
        "SNARE": [],
        "TOM": [2, 4],
        "HI-HAT": []
    },
    frozenset(["BASS", "HI-HAT"]): {
        "BASS": [1, 3],
        "SNARE": [],
        "TOM": [],
        "HI-HAT": [2, 4]
    },
    frozenset(["SNARE", "HI-HAT"]): {
        "BASS": [],
        "SNARE": [1,2, 3,4],
        "TOM": [],
        "HI-HAT": [1.5, 2.5, 3.5, 4.5],
    },
    frozenset(["TOM", "HI-HAT"]): {
        "BASS": [],
        "SNARE": [],
        "TOM": [1, 3],
        "HI-HAT": [2, 4]
    },
    frozenset(["BASS", "SNARE", "TOM"]): {
        "BASS": [1, 3],
        "SNARE": [1.5, 2.5, 3.5, 4.5],
        "TOM": [2, 4],
        "HI-HAT": []
    },
    frozenset(["BASS", "SNARE", "HI-HAT"]): {
        "BASS": [1, 3],
        "SNARE": [2, 4],
        "TOM": [],
        "HI-HAT": [1.5, 2.5, 3.5, 4.5],
    },
    frozenset(["BASS", "TOM", "HI-HAT"]): {
        "BASS": [1, 3],
        "SNARE": [],
        "TOM": [2, 4],
        "HI-HAT": [1.5, 2.5, 3.5, 4.5],
    },
    frozenset(["SNARE", "TOM", "HI-HAT"]): {
        "BASS": [],
        "SNARE": [2, 4],
        "TOM": [1,3],
        "HI-HAT": [1.5, 2.5, 3.5, 4.5],
    },
    frozenset(["BASS", "SNARE", "TOM", "HI-HAT"]): {
        "BASS": [1, 3],
        "SNARE": [2, 4],
        "TOM": [2],
        "HI-HAT": [1.5, 2.5, 3.5, 4.5],
    },
}

def modify_config(bpm=None, drums=None, voices=None, reset=False):
    """
    Modify the config.json file.

    Args:
        bpm (int, optional): New BPM to set. If None, BPM remains unchanged.
        drums (str or list/set of str, optional): Drums to activate.
        voices (str or list/set of str, optional): Voices to activate.
        reset (bool, optional): If True, reset the config to an empty state.
    """
    with config_lock:
        try:
            # If reset is requested, clear the configuration
            if reset:
                default_config = {
                    "BPM": 60,
                    "DRUM_BEATS": {
                        "BASS": {"beats": [], "note": 50},
                        "SNARE": {"beats": [], "note": 48},
                        "TOM": {"beats": [], "note": 47},
                        "HI-HAT": {"beats": [], "note": 45}
                    },
                    "VOICES": {
                        "SOLO": {"active": False},
                        "CHOIR": {"active": False},
                        "BRASS": {"active": False},
                        "ORGAN": {"active": False},
                    }
                }
                with open(CONFIG_FILE, "w") as file:
                    json.dump(default_config, file, indent=4)
                print(f"Configuration reset to default state.")
                return

            # Load the current configuration
            with open(CONFIG_FILE, "r") as file:
                config = json.load(file)

            # Update BPM if provided
            if bpm is not None:
                config["BPM"] = bpm
                print(f"BPM updated to {bpm}.")

            # Update drum beats if drums is not None
            if drums is not None:
                # Convert drums to set for easier handling
                if isinstance(drums, str):
                    drums_set = set([drums.upper()])
                else:
                    drums_set = set([drum.upper() for drum in drums])

                # Make sure all drums are valid
                valid_drums = set(config["DRUM_BEATS"].keys())
                invalid_drums = drums_set - valid_drums
                if invalid_drums:
                    print(f"Invalid drum(s) specified: {', '.join(invalid_drums)}")
                    drums_set = drums_set & valid_drums

                # Attempt to find a predefined pattern for the combination
                combo_pattern = DRUM_COMBO_PATTERNS.get(frozenset(drums_set))

                if combo_pattern:
                    # Use the predefined combination pattern
                    for drum in config["DRUM_BEATS"]:
                        if drum in combo_pattern:
                            beats = combo_pattern[drum]
                            config["DRUM_BEATS"][drum]["beats"] = beats
                            print(f"Set {drum} beats to {beats} for combination {', '.join(drums_set)}.")
                        else:
                            config["DRUM_BEATS"][drum]["beats"] = []
                            print(f"Cleared beats for {drum}.")
                else:
                    # Use default beats for specified drums, clear others
                    for drum in config["DRUM_BEATS"]:
                        if drum in drums_set:
                            # Set beats to default pattern
                            beats = DEFAULT_DRUM_BEATS.get(drum, [])
                            config["DRUM_BEATS"][drum]["beats"] = beats
                            print(f"Set {drum} beats to {beats}.")
                        else:
                            # Clear beats
                            config["DRUM_BEATS"][drum]["beats"] = []
                            print(f"Cleared beats for {drum}.")

            # Update voices if voices is not None
            if voices is not None:
                # Convert voices to set for easier handling
                if isinstance(voices, str):
                    voices_set = set([voices.upper()])
                else:
                    voices_set = set([voice.upper() for voice in voices])

                # Make sure all voices are valid
                valid_voices = set(config["VOICES"].keys())
                invalid_voices = voices_set - valid_voices
                if invalid_voices:
                    print(f"Invalid voice(s) specified: {', '.join(invalid_voices)}")
                    voices_set = voices_set & valid_voices

                # Update the active state of voices
                for voice in config["VOICES"]:
                    if voice in voices_set:
                        config["VOICES"][voice]["active"] = True
                        print(f"Activated voice: {voice}.")
                    else:
                        config["VOICES"][voice]["active"] = False
                        print(f"Deactivated voice: {voice}.")

            # Save the updated configuration
            with open(CONFIG_FILE, "w") as file:
                json.dump(config, file, indent=4)
                print("Config updated successfully.")

        except FileNotFoundError:
            print(f"Configuration file {CONFIG_FILE} not found.")
        except json.JSONDecodeError as e:
            print(f"Error reading {CONFIG_FILE}: {e}")
