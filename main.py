import subprocess
import sys

MAX_BPM = 240
MIN_BPM = 60

def main():
    try:
        # Launch basic_drums_vcv.py
        drums_process = subprocess.Popen([sys.executable, "basic_drums_vcv.py"])
        print("Running basic_drums_vcv.py...")

        # Launch detect.py
        detect_process = subprocess.Popen([sys.executable, "detect.py"])
        print("Running detect.py...")

        # Keep the main process alive until the scripts are terminated
        drums_process.wait()
        detect_process.wait()

    except KeyboardInterrupt:
        print("\nTerminating processes...")
        drums_process.terminate()
        detect_process.terminate()
        print("Processes terminated.")

if __name__ == "__main__":
    main()
