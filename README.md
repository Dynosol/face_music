# Face Music - Interactive Facial Expression Music Generator

Generate music dynamically based on your face and hands using computer vision and real-time audio synthesis.

## Features

- Real-time facial expression detection
- Dynamic music generation based on emotional state
- Drum pattern generation
- Melody synthesis
- Configurable musical parameters
- Cross-platform compatibility

## Project Structure

- `main.py` - Entry point of the application, orchestrates the music generation and facial detection processes
- `detect.py` - Handles facial detection and expression analysis
- `drums_and_melody.py` - Generates drum patterns and melodic sequences
- `vcv_givencomplex.py` - Complex VCV rack integration for advanced sound synthesis
- `MIDI_Funcs.py` - MIDI utility functions for musical data handling
- `config_modifier.py` - Configuration management system
- `config.json` - Application configuration settings

## Prerequisites

- Python 3.8 or higher
- Webcam access
- Audio output device

### Project Architecture

- Facial Detection Module: Uses OpenCV and deep learning models for expression analysis
- Music Generation Module: Handles real-time audio synthesis and MIDI control
- Configuration System: Manages application settings and runtime parameters

See `requirements.txt` for a complete list of dependencies and versions.
