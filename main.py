import subprocess
import sys

MAX_BPM = 240
MIN_BPM = 60

def main():
    try:
        # Launch basic_drums_vcv.py
        music_process = subprocess.Popen([sys.executable, "music_main.py"])
        print("Running music_main.py...")

        # Launch detect.py
        detect_process = subprocess.Popen([sys.executable, "detect.py"])
        print("Running detect.py...")

        # Keep the main process alive until the scripts are terminated
        music_process.wait()
        detect_process.wait()

    except KeyboardInterrupt:
        print("\nTerminating processes...")
        music_process.terminate()
        detect_process.terminate()
        print("Processes terminated.")

if __name__ == "__main__":
    main()
