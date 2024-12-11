# Face Music - Interactive Facial Expression Music Generator

Generate music dynamically based on your facial expressions using computer vision and real-time audio synthesis.

## Overview

Face Music is an innovative Python application that creates a unique musical experience by analyzing facial expressions in real-time and generating corresponding musical patterns. The system uses computer vision to detect facial expressions and emotions, which are then translated into musical parameters that control a dynamic music generation system.

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

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face_music.git
cd face_music
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your webcam is properly connected and accessible

## Configuration

The application can be configured through the `config.json` file. Key parameters include:

- BPM range (60-240)
- Audio device settings
- Facial detection parameters
- Musical pattern configurations

Use `config_modifier.py` to programmatically adjust settings during runtime.

## Usage

1. Start the application:
```bash
python main.py
```

2. Position yourself in front of the webcam
3. The application will begin detecting your facial expressions and generating music
4. Different expressions will influence various aspects of the music:
   - Tempo
   - Melody patterns
   - Drum rhythms
   - Sound characteristics

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

### Project Architecture

- Facial Detection Module: Uses OpenCV and deep learning models for expression analysis
- Music Generation Module: Handles real-time audio synthesis and MIDI control
- Configuration System: Manages application settings and runtime parameters

## Troubleshooting

Common issues and solutions:

1. Webcam not detected:
   - Check webcam connections
   - Verify webcam permissions
   - Ensure no other applications are using the webcam

2. Audio issues:
   - Check audio device settings
   - Verify audio output permissions
   - Ensure correct audio device is selected in system settings

## Dependencies

Major dependencies include:
- OpenCV
- TensorFlow
- Mediapipe
- NumPy
- SoundDevice
- Flask
- Various ML and computer vision libraries

See `requirements.txt` for a complete list of dependencies and versions.