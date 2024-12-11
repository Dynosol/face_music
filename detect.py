from turtle import left, right
import cv2
import mediapipe as mp
import math
import time
from deepface import DeepFace
import config_modifier
from main import MIN_BPM, MAX_BPM
import threading

N = {}# shorthand for notes

# Generate a dictionary mapping note names to MIDI note numbers
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

for midi_number in range(128):
    note_in_octave = midi_number % 12
    octave = (midi_number // 12) - 1
    note_name = note_names[note_in_octave] + str(octave)
    N[note_name] = midi_number

CHORD_SEQUENCE_I_V_vi_IV = [
    [N["C2"], 'Major'],
    [N["G2"], 'Major'],
    [N["A2"], 'Minor'],
    [N["F2"], 'Major'],
], "CHORD_SEQUENCE_(C)_I_V_vi_IV"

CHORD_SEQUENCE_vi_IV_I_V = [
    [N["A2"], 'Minor'],
    [N["F2"], 'Major'],
    [N["C2"], 'Major'],
    [N["G2"], 'Major'],
], "CHORD_SEQUENCE_(C)_vi_IV_I_V"

CHORD_SEQUENCE_I_vi_IV_V = [
    [N["C2"], 'Major'],
    [N["A2"], 'Minor'],
    [N["F2"], 'Major'],
    [N["G2"], 'Major'],
], "CHORD_SEQUENCE_(C)_I_vi_IV_V"

# both closed
CHORD_SEQUENCE_D_I_V_vi_IV = [
    [N["D2"], 'Major'],
    [N["A2"], 'Major'],
    [N["B2"], 'Minor'],
    [N["G2"], 'Major'],
], "CHORD_SEQUENCE_(D)_I_V_vi_IV"

CHORD_SEQUENCE_IV_V__vi_I = [
    [N["F2"], 'Major'],
    [N["G2"], 'Major'],
    [N["A2"], 'Minor'],
    [N["C2"], 'Major'],
], "CHORD_SEQUENCE_(C)_IV_V__vi_I"

CHORD_SEQUENCE_I_ii_vi_V = [
    [N["C2"], 'Major'],
    [N["D2"], 'Minor'],
    [N["A2"], 'Minor'],
    [N["G2"], 'Major'],
], "CHORD_SEQUENCE_(C)_I_ii_vi_V"

# Initialize mediapipe Face Mesh for detecting facial landmarks
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# Landmark indices for facial features
NOSE_TIP_INDEX = 1
MOUTH_TOP_INDEX = 13  # Upper lip
MOUTH_BOTTOM_INDEX = 14  # Lower lip
MOUTH_LEFT_CORNER_INDEX = 61
MOUTH_RIGHT_CORNER_INDEX = 291
LEFT_EYE_TOP_INDEX = 159
LEFT_EYE_BOTTOM_INDEX = 145
RIGHT_EYE_TOP_INDEX = 386
RIGHT_EYE_BOTTOM_INDEX = 374
FOREHEAD_TOP_INDEX = 10
CHIN_BOTTOM_INDEX = 152
LEFT_CHEEK_INDEX = 234
RIGHT_CHEEK_INDEX = 454

CLAP_THRESHOLD = -4
CLAPTEXT_FADE = 0.5

clap_distance = None
palm_height_measure = 0

max_mouth_openness = 0
max_mouth_wideness = 0
min_mouth_wideness = 0.5

fgbg = cv2.createBackgroundSubtractorMOG2()


# EYES CLOSED FOR CHORD SQUENCE
last_chord_state = None
state_start_time = None
left_eye_text = "Closed?"
right_eye_text = "Closed?"

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

finger_names = {
    mp_hands.HandLandmark.INDEX_FINGER_TIP: "Index Finger",
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP: "Middle Finger",
    mp_hands.HandLandmark.RING_FINGER_TIP: "Ring Finger",
    mp_hands.HandLandmark.PINKY_TIP: "Pinky Finger"
}

finger_labels = {
    'Left': ["SNARE", "BASS", "HI-HAT", "TOM"],
    'Right': ["SOLO", "CHOIR", "BRASS", "ORGAN"]
}

finger_states = {label: None for labels in finger_labels.values() for label in labels}
finger_colors = {label: WHITE for labels in finger_labels.values() for label in labels}

# For changing active drums and voices
active_drums = set()
previous_active_drums = set()
active_voices = set()
previous_active_voices = set()

prev_hand_distance = None
prev_time = time.time()  # for clapping!
start_time = time.time()
clap_start_time = None

frame_lock = threading.Lock()

def process_face(face_results, frame):
    global lengths, emotion_text, start_time, max_mouth_openness, max_mouth_wideness, min_mouth_wideness, frame_lock, clap_distance, last_chord_state, state_start_time, left_eye_text, right_eye_text

    current_time = time.time()
    if current_time - start_time >= 5:
        # Reset values
        max_mouth_openness = 0
        max_mouth_wideness = 0
        min_mouth_wideness = 0.5
        # Reset the start time for the next interval
        start_time = current_time

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            h, w, _ = frame.shape  # frame dimensions for pixel conversion

            # Get coordinates for the mouth and calculate mouth openness
            mouth_top = face_landmarks.landmark[MOUTH_TOP_INDEX]
            mouth_bottom = face_landmarks.landmark[MOUTH_BOTTOM_INDEX]
            mouth_left = face_landmarks.landmark[MOUTH_LEFT_CORNER_INDEX]
            mouth_right = face_landmarks.landmark[MOUTH_RIGHT_CORNER_INDEX]
            mouth_top_y = mouth_top.y * h
            mouth_bottom_y = mouth_bottom.y * h
            mouth_left_x = mouth_left.x * w
            mouth_right_x = mouth_right.x * w
            mouth_openness = abs(mouth_bottom_y - mouth_top_y)
            mouth_wideness = abs(mouth_left_x - mouth_right_x)
            head_height = abs(face_landmarks.landmark[FOREHEAD_TOP_INDEX].y - face_landmarks.landmark[CHIN_BOTTOM_INDEX].y) * h
            head_width = abs(face_landmarks.landmark[RIGHT_CHEEK_INDEX].x - face_landmarks.landmark[LEFT_CHEEK_INDEX].x) * w
            lengths.append(f"Head Height: {head_height:.2f}")
            lengths.append(f"Head Width: {head_width:.2f}")
            max_mouth_openness = max(head_height / 4, mouth_wideness)
            max_mouth_wideness = max(head_width / 2, mouth_wideness)
            mouth_openness_normalized = min(mouth_openness / max_mouth_openness, 1.0)

            mouth_wideness_normalized = min(mouth_wideness / max_mouth_wideness, 1.0)

            # normalize mouth wideness for differing head widths
            min_mouth_wideness = min(mouth_wideness_normalized + 0.05, min_mouth_wideness)

            scaled_mouth_wideness = ((mouth_wideness_normalized - min_mouth_wideness) / (1.0 - min_mouth_wideness))
            scaled_mouth_wideness = max(0, min(scaled_mouth_wideness, 1))  # Clamp between 0 and 1

            bpm = int(MIN_BPM + (scaled_mouth_wideness * (MAX_BPM - MIN_BPM)))

            clap_distance = head_width * 3

            # ==================================================================================================================================
            config_modifier.modify_config(bpm=bpm)
            # ==================================================================================================================================

            lengths.append(f"Mouth Openness Normalized: {mouth_openness_normalized:.2f}")
            lengths.append(f"Mouth Wideness Normalized: {scaled_mouth_wideness:.2f}")

            # Get coordinates for the top and bottom of the left and right eyes
            left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP_INDEX]
            left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM_INDEX]
            right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP_INDEX]
            right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM_INDEX]
            left_eye_top_x, left_eye_top_y = int(left_eye_top.x * w), int(left_eye_top.y * h)
            left_eye_bottom_x, left_eye_bottom_y = int(left_eye_bottom.x * w), int(left_eye_bottom.y * h)
            right_eye_top_x, right_eye_top_y = int(right_eye_top.x * w), int(right_eye_top.y * h)
            right_eye_bottom_x, right_eye_bottom_y = int(right_eye_bottom.x * w), int(right_eye_bottom.y * h)
            mouth_left_x, mouth_left_y = int(mouth_left.x * w), int(mouth_left.y * h)
            mouth_right_x, mouth_right_y = int(mouth_right.x * w), int(mouth_right.y * h)

            # Calculate eye openness
            left_eye_openness = abs(left_eye_bottom_y - left_eye_top_y)
            right_eye_openness = abs(right_eye_bottom_y - right_eye_top_y)
            eye_max_open = head_height / 70 # some weird phenotypical stuff... but it works
            lengths.append(f"Eye Max Openness: {eye_max_open:.2f}")

            left_closed = left_eye_openness < eye_max_open
            right_closed = right_eye_openness < eye_max_open

            # Update eye text instantly
            if left_closed and right_closed:
                left_eye_text = "Closed"
                right_eye_text = f"Closed, {CHORD_SEQUENCE_D_I_V_vi_IV[1]}"
                current_state = "both_closed"
            elif not left_closed and not right_closed:
                left_eye_text = "Open"
                right_eye_text = f"Open, {CHORD_SEQUENCE_I_ii_vi_V[1]}" 
                current_state = "both_open"
            elif not left_closed and right_closed:
                left_eye_text = "Open"
                right_eye_text = f"Closed, {CHORD_SEQUENCE_vi_IV_I_V[1]}"
                current_state = "left_open_right_closed"
            elif left_closed and not right_closed:
                left_eye_text = "Closed"
                right_eye_text = f"Open, {CHORD_SEQUENCE_IV_V__vi_I[1]}"
                current_state = "left_closed_right_open"
            else:
                current_state = None

            # Handle chord modification
            if current_state != last_chord_state:
                # State changed, reset timer
                last_chord_state = current_state
                state_start_time = time.time()
            else:
                # State is consistent
                elapsed_time = time.time() - state_start_time
                if elapsed_time >= 2:
                    # State has been consistent for 2 seconds, modify chord
                    if current_state == "both_closed":
                        config_modifier.modify_config(chord_sequence=CHORD_SEQUENCE_D_I_V_vi_IV[0])
                    elif current_state == "both_open":
                        config_modifier.modify_config(chord_sequence=CHORD_SEQUENCE_I_ii_vi_V[0])
                    elif current_state == "left_open_right_closed":
                        config_modifier.modify_config(chord_sequence=CHORD_SEQUENCE_vi_IV_I_V[0])
                    elif current_state == "left_closed_right_open":
                        config_modifier.modify_config(chord_sequence=CHORD_SEQUENCE_IV_V__vi_I[0])

            lengths.append(f"Left Eye Openness: {left_eye_openness:.2f}")
            lengths.append(f"Right Eye Openness: {right_eye_openness:.2f}")

            # Draw circles on the detected facial features for visualization
            with frame_lock:
                cv2.circle(frame, (left_eye_top_x, left_eye_top_y), 5, RED, -1)
                cv2.circle(frame, (left_eye_bottom_x, left_eye_bottom_y), 5, RED, -1)
                cv2.circle(frame, (right_eye_top_x, right_eye_top_y), 5, RED, -1)
                cv2.circle(frame, (right_eye_bottom_x, right_eye_bottom_y), 5, RED, -1)
                cv2.circle(frame, (int(mouth_top.x * w), int(mouth_top.y * h)), 5, RED, -1)
                cv2.circle(frame, (int(mouth_bottom.x * w), int(mouth_bottom.y * h)), 5, RED, -1)
                cv2.circle(frame, (mouth_left_x, mouth_left_y), 5, BLUE, -1)
                cv2.circle(frame, (mouth_right_x, mouth_right_y), 5, BLUE, -1)

                # Draw lines between facial features
                cv2.line(frame, (left_eye_top_x, left_eye_top_y), (left_eye_bottom_x, left_eye_bottom_y), RED, 2)
                cv2.line(frame, (right_eye_top_x, right_eye_top_y), (right_eye_bottom_x, right_eye_bottom_y), RED, 2)
                cv2.line(frame, (int(mouth_top.x * w), int(mouth_top.y * h)), (int(mouth_bottom.x * w), int(mouth_bottom.y * h)), RED, 2)
                cv2.line(frame, (mouth_left_x, mouth_left_y), (mouth_right_x, mouth_right_y), BLUE, 2)

                cv2.putText(frame, f"Openness: {round(mouth_openness_normalized * 100, 0)}%", (int(mouth_top.x * w), int(mouth_top.y * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Wideness (BPM): {round(scaled_mouth_wideness * 100, 0)}%, {bpm} BPM", (mouth_left_x, mouth_left_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{left_eye_text}", (left_eye_top_x, left_eye_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{right_eye_text}", (right_eye_top_x, right_eye_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

            # Extract the face ROI for emotion analysis
            x_min = min(int(mouth_left.x * w), left_eye_top_x, right_eye_top_x)
            y_min = min(int(mouth_top.y * h), left_eye_top_y, right_eye_top_y)
            x_max = max(int(mouth_right.x * w), left_eye_bottom_x, right_eye_bottom_x)
            y_max = max(int(mouth_bottom.y * h), left_eye_bottom_y, right_eye_bottom_y)

            face_roi = frame[y_min:y_max, x_min:x_max]

            # Perform emotion analysis
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion_text = result[0]['dominant_emotion']
            except:
                emotion_text = ""

def process_hands(hand_results, frame):
    global lengths, velocity_text, clap_detected, finger_length
    global active_drums, previous_active_drums
    global active_voices, previous_active_voices
    global finger_states, finger_colors, prev_hand_distance, prev_time, clap_start_time, original_hand_distance
    global palm_height_measure
    global frame_lock

    pinch_time = time.time()
    if pinch_time - start_time >= 10:
        # Reset values
        palm_height_measure = 0

    if hand_results.multi_hand_landmarks:
        hand_midpoints = []
        for hand_landmarks, hand_label in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            h, w, _ = frame.shape  # frame dimensions for pixel conversion
            # Get the label of the hand ('Left' or 'Right')
            label = hand_label.classification[0].label
            # labels for this hand
            labels_for_hand = finger_labels[label]

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            # Calculate palm height
            wrist_y = wrist.y * h
            pinky_mcp_y = pinky_mcp.y * h
            palm_height_measure = max(abs(pinky_mcp_y - wrist_y), palm_height_measure)
            pinch_threshold = palm_height_measure / 3 # I know it's magic but it works

            # Calculate midpoint between thumb and index finger for each hand
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            midpoint_x = (thumb_x + index_x) // 2
            midpoint_y = (thumb_y + index_y) // 2
            hand_midpoints.append((midpoint_x, midpoint_y))

            # Draw midpoint
            with frame_lock:
                cv2.circle(frame, (midpoint_x, midpoint_y), 4, BLUE, -1)

            finger_indices = [
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]

            for idx, finger_idx in enumerate(finger_indices):
                landmark = hand_landmarks.landmark[finger_idx]
                hand_x, hand_y = int(landmark.x * w), int(landmark.y * h)

                # Draw circles on the detected hand landmarks for visualization
                with frame_lock:
                    cv2.circle(frame, (hand_x, hand_y), 2, GREEN, -1)

                # Process pinch detection between thumb and this finger
                cv2.line(frame, (thumb_x, thumb_y), (hand_x, hand_y), GREEN, 1)
                finger_length = math.sqrt((thumb_x - hand_x) ** 2 + (thumb_y - hand_y) ** 2)
                lengths.append(f"Thumb to {finger_names.get(finger_idx, 'Unknown')}: {finger_length:.2f}")

                # Assign finger label for this finger
                finger_label = labels_for_hand[idx]

                # Determine if the finger is pinched or unpinched
                current_state = "pinched" if finger_length < pinch_threshold else "unpinched"

                # Compare with the previous state
                if finger_states[finger_label] != current_state:
                    finger_states[finger_label] = current_state

                    # Update active_drums or active_voices based on the hand label
                    if label == 'Left':
                        # Left hand controls drums
                        if current_state == "pinched":
                            finger_colors[finger_label] = RED
                            active_drums.add(finger_label)
                        else:
                            finger_colors[finger_label] = WHITE
                            active_drums.discard(finger_label)
                    elif label == 'Right':
                        # Right hand controls voices
                        if current_state == "pinched":
                            finger_colors[finger_label] = RED
                            active_voices.add(finger_label)
                        else:
                            finger_colors[finger_label] = WHITE
                            active_voices.discard(finger_label)

                    # Only update config if active_drums or active_voices has changed
                    if active_drums != previous_active_drums or active_voices != previous_active_voices:
                        # Update config with the current sets of active drums and voices
                        config_modifier.modify_config(drums=active_drums, voices=active_voices)
                        previous_active_drums = active_drums.copy()
                        previous_active_voices = active_voices.copy()

                # Try to display the label
                try:
                    with frame_lock:
                        cv2.putText(
                            frame, finger_label, (hand_x, hand_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, finger_colors[finger_label], 2, cv2.LINE_AA
                        )
                except Exception as e:
                    print(f"Error while putting text on hand landmark: {e}")

        # Draw line between the midpoints of both hands (if both hands are detected)
        if len(hand_midpoints) == 2:
            hand_distance = math.sqrt((hand_midpoints[0][0] - hand_midpoints[1][0]) ** 2 + (hand_midpoints[0][1] - hand_midpoints[1][1]) ** 2)
            current_time = time.time()
            if prev_hand_distance is not None:
                time_diff = current_time - prev_time
                distance_ratio = hand_distance / clap_distance
                velocity = (distance_ratio - (prev_hand_distance / clap_distance)) / time_diff
                velocity_text = f"Velocity of Hand Distance: {velocity:.2f}x/sec"

                # Detect clap based on velocity ratio
                if velocity < CLAP_THRESHOLD:
                    clap_detected = True

            prev_hand_distance = hand_distance
            prev_time = current_time

            # Draw line between hand midpoints, change color if clap is detected
            line_color = RED if clap_detected else BLUE
            with frame_lock:
                cv2.line(frame, hand_midpoints[0], hand_midpoints[1], line_color, 2)

            # Display "CLAP DETECTED" if a clap is detected
            midpoint_x = (hand_midpoints[0][0] + hand_midpoints[1][0]) // 2
            midpoint_y = (hand_midpoints[0][1] + hand_midpoints[1][1]) // 2 - 20
            clap_text = "CLAP NOT DETECTED"
            clap_text_color = WHITE
            if clap_detected:
                clap_text = "CLAP DETECTED"
                clap_text_color = RED
                clap_start_time = time.time()
            with frame_lock:
                cv2.putText(frame, clap_text, (midpoint_x - 50, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 1, clap_text_color, 2, cv2.LINE_AA)

            if clap_start_time is not None:
                elapsed_time = time.time() - clap_start_time
                if elapsed_time < CLAPTEXT_FADE:  # Display text for 0.5 seconds
                    # Calculate the fade effect by reducing the opacity over time (fades in 0.5 seconds)
                    fade_factor = max(0, 1 - (elapsed_time / CLAPTEXT_FADE))

                    # Create a transparent overlay
                    overlay = frame.copy()

                    # Set the text on the overlay
                    with frame_lock:
                        cv2.putText(overlay, "CLAP DETECTED", (int(w * 0.37), int(h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 5, cv2.LINE_AA)

                        # Blend the overlay with the original frame
                        cv2.addWeighted(overlay, fade_factor, frame, 1 - fade_factor, 0, frame)
                else:
                    clap_start_time = None  # Reset clap_start_time after 0.5 seconds

    else:
        # Reset the active drums and voices if no hands are detected
        if active_drums or active_voices:
            active_drums.clear()
            active_voices.clear()
            config_modifier.modify_config(drums=active_drums, voices=active_voices)

def display_results(frame, lengths, emotion_text, velocity_text):
    y_offset = 20
    for length in lengths:
        cv2.putText(frame, length, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        y_offset += 20

    # Display the detected emotion on the top right corner
    if emotion_text:
        cv2.putText(frame, f"Emotion: {emotion_text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        y_offset += 30

    # Display the velocity of hand distance
    if velocity_text:
        cv2.putText(frame, velocity_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)

def main():
    global lengths, emotion_text, velocity_text, clap_detected, finger_length
    global prev_hand_distance, prev_time, clap_start_time, original_hand_distance, start_time

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (1920, 1080))  # Increase resolution for better visualization
        frame = cv2.flip(frame, 1)  # Mirror the image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe requires RGB format

        # Process the frame for facial landmarks
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        lengths = []  # List to store the lengths of the features
        emotion_text = ""
        velocity_text = ""
        clap_detected = False
        finger_length = float('inf')  # for the thumb default

        # Create threads
        face_thread = threading.Thread(target=process_face, args=(face_results, frame))
        hand_thread = threading.Thread(target=process_hands, args=(hand_results, frame))

        face_thread.start()
        hand_thread.start()

        face_thread.join()
        hand_thread.join()

        display_results(frame, lengths, emotion_text, velocity_text)

        # Exit if Backspace or 'q' is pressed
        key = cv2.waitKey(1)
        if key == 8 or key == ord('q'):  # ASCII code for Backspace is 8, fallback to 'q'
            break

    # Release the webcam and close the window
    config_modifier.modify_config(reset=True)  # all playing music should stop
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
