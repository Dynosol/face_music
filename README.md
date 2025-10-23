# Face Music - Interactive Face and Hand Procedural Music Generation

Generate music dynamically based on your face and hands using computer vision and real-time audio synthesis.

[![Usage example!](https://img.youtube.com/vi/0CHTdzLosq8/0.jpg)](https://www.youtube.com/watch?v=0CHTdzLosq8)

## Features

- Play music based on your face and hands
- Facial/hand detection via mediapipe routed from Python to MIDI output then to VCV Rack
- Adjust BPM with mouth width (mouth openness proved to be too variable)
- Adjust configurable chord sequence (any chord, ~almost any scale) by permutating which eyes are closed (held open/closed for two seconds, to prevent rapid changes)
- Play and adjust configurable drum combinations by pinching corresponding fingers to thumb on left hand
- Play procedurally generated reversible uniform markov-chain-based melody with normalized unique scale-pitch preferences and rhythm preferences
- Play configurable chord-sequence (on dominant, third, fifth)
- Play procedurally generated arpeggiated voice line
- Play pedal tones
- Adjust configurable voice ranges

## Project Structure

- `main.py` - Runs the program (`detect.py` and `drums_and_melody.py`)
- `detect.py` - Handles facial feature and hand feature detection and adjusts `config.json` accordingly
- `drums_and_melody.py` - Generates drum patterns and melodic sequences
- `MIDI_Funcs.py` - MIDI utility functions for musical data handling (only used for shutting down midi outputs when program closes)
- `config_modifier.py` - Allows for real-time modification of the config so `detect.py` and `drums_and_melody.py` can both access it
- `config.json` - Dynamic config that changes and is accessed in runtime

## Prerequisites

- Python 3.8 or higher
- Webcam access
- VCV Rack 2 Free (several downloadable modules that give the option to download when the file is loaded)

## Usage

1. Set output in VCV Rack to the output source you want
2. Set the correct midi inputs for each MIDI>CV module, the top center should be IAC Driver Bus 1, top left IAC Driver Bus 10, everything below 1 is 2,3,4, top left is 6, and left center is 5 (sorry)
3. Start the application:
```bash
pip intall requirements.txt
python main.py
```
4. Position yourself in front of the webcam
5. Pinching (and holding) fingers on the left hand against the thumb plays, configurably, looping lines for the permutation of fingers pinched on the drum set
6. Pinching (and holding) fingers on the right hand plays different voices
7. Mouth width adjusts BPM
8. Eyes open or closed adjusts configurable chord sequence on configurable bass notes (hold for two seconds to change)

## Configuration (everything labeled MODIFY THIS)

- `main.py` - Adjust max and min BPM
- `drums_and_melody.py` - Adjust or add scales, note priorities, chord intervals, and other voice-specific variables (density factor/neighborhood factor for melody, for example), voice note ranges
- `detect.py` - Adjust chord sequences (using Note dictionary, one chord in the array could be [N["C2"], 'Major'], for example), adjust clap threshold, text color and other visuals, max/min bpm
- `config_modifier.py` - Adjust drum configurations (1,2,3,4 for downbeats, 1.5,2.5,3.5,4.5 for upbeats)

## Some custom algorithms explained
- Things like max mouth width or openness or eye openness, etc. are determined dynamically, so it should adjust per person's face, adjusts max/min on a timer, sae with eyes closed or open
- Everything is ratio-based, so detecting finger pinching is a ratio of the height of the palm
- The melody line is a pseudo Markov chain that only depends on the last note, has exponential preference for nearbly notes and has normalized preference for certain scale tone notes, increases playing odds based on if the beat is on the downbeat, less so on upbeat, other rhythmic factors included
- The arpeggiating line semi-randomly decides to go up or down (rarely switches direction, but resets when chord changes and also avoids going above or below the range), prefers exponentially to skip less notes than more

# Credit to:
- Harvard GENED1080 team, Professor Robert Wood, PHD candidate Lauryn Whiteside, PHD candidate Rachel Rosenman
- Garett/Will Morisson for proof of concept and inspiration for the procedural generation algorithm (his is much more intense at https://github.com/GarettMorrison/Procedural_MIDI)


