from turtle import left, right
import cv2
import mediapipe as mp # facial landmarks
import math
import time
from deepface import DeepFace # just for the emotions only
import config_modifier
from main import MAX_BPM, MIN_BPM
import threading

class FaceMusicDetector:
    def __init__(self):
        # Initialize note dictionary
        self.N = {}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for midi_number in range(128):
            note_in_octave = midi_number % 12
            octave = (midi_number // 12) - 1
            note_name = note_names[note_in_octave] + str(octave)
            self.N[note_name] = midi_number

        # Initialize chord sequences
        self.CHORD_SEQUENCE_I_V_vi_IV = [
            [self.N["C2"], 'Major'],
            [self.N["G2"], 'Major'],
            [self.N["A2"], 'Minor'],
            [self.N["F2"], 'Major'],
        ], "CHORD_SEQUENCE_(C)_I_V_vi_IV"

        self.CHORD_SEQUENCE_vi_IV_I_V = [
            [self.N["A2"], 'Minor'],
            [self.N["F2"], 'Major'],
            [self.N["C2"], 'Major'],
            [self.N["G2"], 'Major'],
        ], "CHORD_SEQUENCE_(C)_vi_IV_I_V"

        self.CHORD_SEQUENCE_I_vi_IV_V = [
            [self.N["C2"], 'Major'],
            [self.N["A2"], 'Minor'],
            [self.N["F2"], 'Major'],
            [self.N["G2"], 'Major'],
        ], "CHORD_SEQUENCE_(C)_I_vi_IV_V"

        self.CHORD_SEQUENCE_D_I_V_vi_IV = [
            [self.N["D2"], 'Major'],
            [self.N["A2"], 'Major'],
            [self.N["B2"], 'Minor'],
            [self.N["G2"], 'Major'],
        ], "CHORD_SEQUENCE_(D)_I_V_vi_IV"

        self.CHORD_SEQUENCE_IV_V__vi_I = [
            [self.N["F2"], 'Major'],
            [self.N["G2"], 'Major'],
            [self.N["A2"], 'Minor'],
            [self.N["C2"], 'Major'],
        ], "CHORD_SEQUENCE_(C)_IV_V__vi_I"

        self.CHORD_SEQUENCE_I_ii_vi_V = [
            [self.N["C2"], 'Major'],
            [self.N["D2"], 'Minor'],
            [self.N["A2"], 'Minor'],
            [self.N["G2"], 'Major'],
        ], "CHORD_SEQUENCE_(C)_I_ii_vi_V"

        # Initialize mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)

        # Colors
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        self.WHITE = (255, 255, 255)

        # Landmark indices
        self.NOSE_TIP_INDEX = 1
        self.MOUTH_TOP_INDEX = 13
        self.MOUTH_BOTTOM_INDEX = 14
        self.MOUTH_LEFT_CORNER_INDEX = 61
        self.MOUTH_RIGHT_CORNER_INDEX = 291
        self.LEFT_EYE_TOP_INDEX = 159
        self.LEFT_EYE_BOTTOM_INDEX = 145
        self.RIGHT_EYE_TOP_INDEX = 386
        self.RIGHT_EYE_BOTTOM_INDEX = 374
        self.FOREHEAD_TOP_INDEX = 10
        self.CHIN_BOTTOM_INDEX = 152
        self.LEFT_CHEEK_INDEX = 234
        self.RIGHT_CHEEK_INDEX = 454

        # Constants
        self.CLAP_THRESHOLD = -4
        self.CLAPTEXT_FADE = 0.5

        # State variables
        self.clap_distance = None
        self.palm_height_measure = 0
        self.max_mouth_openness = 0
        self.max_mouth_wideness = 0
        self.min_mouth_wideness = 0.5
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.last_chord_state = None
        self.state_start_time = None
        self.left_eye_text = "Closed?"
        self.right_eye_text = "Closed?"
        self.video_capture = cv2.VideoCapture(0)
        self.frame_lock = threading.Lock()
        self.lengths = []
        self.emotion_text = ""
        self.velocity_text = ""

        # Finger tracking
        self.finger_names = {
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP: "Index Finger",
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP: "Middle Finger",
            self.mp_hands.HandLandmark.RING_FINGER_TIP: "Ring Finger",
            self.mp_hands.HandLandmark.PINKY_TIP: "Pinky Finger"
        }

        self.finger_labels = {
            'Left': ["SNARE", "BASS", "HI-HAT", "TOM"],
            'Right': ["SOLO", "CHOIR", "BRASS", "ORGAN"]
        }

        self.finger_states = {label: None for labels in self.finger_labels.values() for label in labels}
        self.finger_colors = {label: self.WHITE for labels in self.finger_labels.values() for label in labels}

        # Active instruments
        self.active_drums = set()
        self.previous_active_drums = set()
        self.active_voices = set()
        self.previous_active_voices = set()

        # Timing variables
        self.prev_hand_distance = None
        self.prev_time = time.time()
        self.start_time = time.time()
        self.clap_start_time = None

    def process_face(self, face_results, frame):
        current_time = time.time()
        if current_time - self.start_time >= 5:
            # Reset values
            self.max_mouth_openness = 0
            self.max_mouth_wideness = 0
            self.min_mouth_wideness = 0.5
            self.start_time = current_time

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Get coordinates for the mouth and calculate mouth openness
                mouth_top = face_landmarks.landmark[self.MOUTH_TOP_INDEX]
                mouth_bottom = face_landmarks.landmark[self.MOUTH_BOTTOM_INDEX]
                mouth_left = face_landmarks.landmark[self.MOUTH_LEFT_CORNER_INDEX]
                mouth_right = face_landmarks.landmark[self.MOUTH_RIGHT_CORNER_INDEX]
                mouth_top_y = mouth_top.y * h
                mouth_bottom_y = mouth_bottom.y * h
                mouth_left_x = mouth_left.x * w
                mouth_right_x = mouth_right.x * w
                mouth_openness = abs(mouth_bottom_y - mouth_top_y)
                mouth_wideness = abs(mouth_left_x - mouth_right_x)
                head_height = abs(face_landmarks.landmark[self.FOREHEAD_TOP_INDEX].y - face_landmarks.landmark[self.CHIN_BOTTOM_INDEX].y) * h
                head_width = abs(face_landmarks.landmark[self.RIGHT_CHEEK_INDEX].x - face_landmarks.landmark[self.LEFT_CHEEK_INDEX].x) * w
                self.lengths.append(f"Head Height: {head_height:.2f}")
                self.lengths.append(f"Head Width: {head_width:.2f}")
                self.max_mouth_openness = max(head_height / 4, mouth_wideness)
                self.max_mouth_wideness = max(head_width / 2, mouth_wideness)
                mouth_openness_normalized = min(mouth_openness / self.max_mouth_openness, 1.0)

                mouth_wideness_normalized = min(mouth_wideness / self.max_mouth_wideness, 1.0)
                self.min_mouth_wideness = min(mouth_wideness_normalized + 0.05, self.min_mouth_wideness)

                scaled_mouth_wideness = ((mouth_wideness_normalized - self.min_mouth_wideness) / (1.0 - self.min_mouth_wideness))
                scaled_mouth_wideness = max(0, min(scaled_mouth_wideness, 1))

                bpm = int(MIN_BPM + (scaled_mouth_wideness * (MAX_BPM - MIN_BPM)))
                self.clap_distance = head_width * 3

                config_modifier.modify_config(bpm=bpm)

                self.lengths.append(f"Mouth Openness Normalized: {mouth_openness_normalized:.2f}")
                self.lengths.append(f"Mouth Wideness Normalized: {scaled_mouth_wideness:.2f}")

                # Get coordinates for eyes
                left_eye_top = face_landmarks.landmark[self.LEFT_EYE_TOP_INDEX]
                left_eye_bottom = face_landmarks.landmark[self.LEFT_EYE_BOTTOM_INDEX]
                right_eye_top = face_landmarks.landmark[self.RIGHT_EYE_TOP_INDEX]
                right_eye_bottom = face_landmarks.landmark[self.RIGHT_EYE_BOTTOM_INDEX]
                left_eye_top_x, left_eye_top_y = int(left_eye_top.x * w), int(left_eye_top.y * h)
                left_eye_bottom_x, left_eye_bottom_y = int(left_eye_bottom.x * w), int(left_eye_bottom.y * h)
                right_eye_top_x, right_eye_top_y = int(right_eye_top.x * w), int(right_eye_top.y * h)
                right_eye_bottom_x, right_eye_bottom_y = int(right_eye_bottom.x * w), int(right_eye_bottom.y * h)
                mouth_left_x, mouth_left_y = int(mouth_left.x * w), int(mouth_left.y * h)
                mouth_right_x, mouth_right_y = int(mouth_right.x * w), int(mouth_right.y * h)

                # Calculate eye openness
                left_eye_openness = abs(left_eye_bottom_y - left_eye_top_y)
                right_eye_openness = abs(right_eye_bottom_y - right_eye_top_y)
                eye_max_open = head_height / 70
                self.lengths.append(f"Eye Max Openness: {eye_max_open:.2f}")

                left_closed = left_eye_openness < eye_max_open
                right_closed = right_eye_openness < eye_max_open

                # Update eye text and handle chord states
                if left_closed and right_closed:
                    self.left_eye_text = "Closed"
                    self.right_eye_text = f"Closed, {self.CHORD_SEQUENCE_D_I_V_vi_IV[1]}"
                    current_state = "both_closed"
                elif not left_closed and not right_closed:
                    self.left_eye_text = "Open"
                    self.right_eye_text = f"Open, {self.CHORD_SEQUENCE_I_ii_vi_V[1]}"
                    current_state = "both_open"
                elif not left_closed and right_closed:
                    self.left_eye_text = "Open"
                    self.right_eye_text = f"Closed, {self.CHORD_SEQUENCE_vi_IV_I_V[1]}"
                    current_state = "left_open_right_closed"
                elif left_closed and not right_closed:
                    self.left_eye_text = "Closed"
                    self.right_eye_text = f"Open, {self.CHORD_SEQUENCE_IV_V__vi_I[1]}"
                    current_state = "left_closed_right_open"
                else:
                    current_state = None

                # Handle chord modification
                if current_state != self.last_chord_state:
                    self.last_chord_state = current_state
                    self.state_start_time = time.time()
                else:
                    elapsed_time = time.time() - self.state_start_time
                    if elapsed_time >= 2:
                        if current_state == "both_closed":
                            config_modifier.modify_config(chord_sequence=self.CHORD_SEQUENCE_D_I_V_vi_IV[0])
                        elif current_state == "both_open":
                            config_modifier.modify_config(chord_sequence=self.CHORD_SEQUENCE_I_ii_vi_V[0])
                        elif current_state == "left_open_right_closed":
                            config_modifier.modify_config(chord_sequence=self.CHORD_SEQUENCE_vi_IV_I_V[0])
                        elif current_state == "left_closed_right_open":
                            config_modifier.modify_config(chord_sequence=self.CHORD_SEQUENCE_IV_V__vi_I[0])

                # Draw visualization
                with self.frame_lock:
                    cv2.circle(frame, (left_eye_top_x, left_eye_top_y), 5, self.RED, -1)
                    cv2.circle(frame, (left_eye_bottom_x, left_eye_bottom_y), 5, self.RED, -1)
                    cv2.circle(frame, (right_eye_top_x, right_eye_top_y), 5, self.RED, -1)
                    cv2.circle(frame, (right_eye_bottom_x, right_eye_bottom_y), 5, self.RED, -1)
                    cv2.circle(frame, (int(mouth_top.x * w), int(mouth_top.y * h)), 5, self.RED, -1)
                    cv2.circle(frame, (int(mouth_bottom.x * w), int(mouth_bottom.y * h)), 5, self.RED, -1)
                    cv2.circle(frame, (mouth_left_x, mouth_left_y), 5, self.BLUE, -1)
                    cv2.circle(frame, (mouth_right_x, mouth_right_y), 5, self.BLUE, -1)

                    cv2.line(frame, (left_eye_top_x, left_eye_top_y), (left_eye_bottom_x, left_eye_bottom_y), self.RED, 2)
                    cv2.line(frame, (right_eye_top_x, right_eye_top_y), (right_eye_bottom_x, right_eye_bottom_y), self.RED, 2)
                    cv2.line(frame, (int(mouth_top.x * w), int(mouth_top.y * h)), (int(mouth_bottom.x * w), int(mouth_bottom.y * h)), self.RED, 2)
                    cv2.line(frame, (mouth_left_x, mouth_left_y), (mouth_right_x, mouth_right_y), self.BLUE, 2)

                    cv2.putText(frame, f"Openness: {round(mouth_openness_normalized * 100, 0)}%", (int(mouth_top.x * w), int(mouth_top.y * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Wideness (BPM): {round(scaled_mouth_wideness * 100, 0)}%, {bpm} BPM", (mouth_left_x, mouth_left_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"{self.left_eye_text}", (left_eye_top_x, left_eye_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"{self.right_eye_text}", (right_eye_top_x, right_eye_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2, cv2.LINE_AA)

                # Extract face ROI for emotion analysis
                x_min = min(int(mouth_left.x * w), left_eye_top_x, right_eye_top_x)
                y_min = min(int(mouth_top.y * h), left_eye_top_y, right_eye_top_y)
                x_max = max(int(mouth_right.x * w), left_eye_bottom_x, right_eye_bottom_x)
                y_max = max(int(mouth_bottom.y * h), left_eye_bottom_y, right_eye_bottom_y)

                face_roi = frame[y_min:y_max, x_min:x_max]

                # Perform emotion analysis
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    self.emotion_text = result[0]['dominant_emotion']
                except:
                    self.emotion_text = ""

    def process_hands(self, hand_results, frame):
        pinch_time = time.time()
        if pinch_time - self.start_time >= 10:
            # Reset values
            self.palm_height_measure = 0

        if hand_results.multi_hand_landmarks:
            hand_midpoints = []
            for hand_landmarks, hand_label in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                h, w, _ = frame.shape
                # Get the label of the hand ('Left' or 'Right')
                label = hand_label.classification[0].label
                # labels for this hand
                labels_for_hand = self.finger_labels[label]

                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

                # Calculate palm height
                wrist_y = wrist.y * h
                pinky_mcp_y = pinky_mcp.y * h
                self.palm_height_measure = max(abs(pinky_mcp_y - wrist_y), self.palm_height_measure)
                pinch_threshold = min(self.palm_height_measure / 4, 100)

                # Calculate midpoint between thumb and index finger for each hand
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                midpoint_x = (thumb_x + index_x) // 2
                midpoint_y = (thumb_y + index_y) // 2
                hand_midpoints.append((midpoint_x, midpoint_y))

                # Draw midpoint
                with self.frame_lock:
                    cv2.circle(frame, (midpoint_x, midpoint_y), 4, self.BLUE, -1)

                finger_indices = [
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    self.mp_hands.HandLandmark.RING_FINGER_TIP,
                    self.mp_hands.HandLandmark.PINKY_TIP
                ]

                for idx, finger_idx in enumerate(finger_indices):
                    landmark = hand_landmarks.landmark[finger_idx]
                    hand_x, hand_y = int(landmark.x * w), int(landmark.y * h)

                    # Draw circles on the detected hand landmarks for visualization
                    with self.frame_lock:
                        cv2.circle(frame, (hand_x, hand_y), 2, self.GREEN, -1)

                    # Process pinch detection between thumb and this finger
                    cv2.line(frame, (thumb_x, thumb_y), (hand_x, hand_y), self.GREEN, 1)
                    finger_length = math.sqrt((thumb_x - hand_x) ** 2 + (thumb_y - hand_y) ** 2)
                    self.lengths.append(f"Thumb to {self.finger_names.get(finger_idx, 'Unknown')}: {finger_length:.2f}")

                    # Assign finger label for this finger
                    finger_label = labels_for_hand[idx]

                    # Determine if the finger is pinched or unpinched
                    current_state = "pinched" if finger_length < pinch_threshold else "unpinched"

                    # Compare with the previous state
                    if self.finger_states[finger_label] != current_state:
                        self.finger_states[finger_label] = current_state

                        # Update active_drums or active_voices based on the hand label
                        if label == 'Left':
                            # Left hand controls drums
                            if current_state == "pinched":
                                self.finger_colors[finger_label] = self.RED
                                self.active_drums.add(finger_label)
                            else:
                                self.finger_colors[finger_label] = self.WHITE
                                self.active_drums.discard(finger_label)
                        elif label == 'Right':
                            # Right hand controls voices
                            if current_state == "pinched":
                                self.finger_colors[finger_label] = self.RED
                                self.active_voices.add(finger_label)
                            else:
                                self.finger_colors[finger_label] = self.WHITE
                                self.active_voices.discard(finger_label)

                        # Only update config if active_drums or active_voices has changed
                        if self.active_drums != self.previous_active_drums or self.active_voices != self.previous_active_voices:
                            # Update config with the current sets of active drums and voices
                            config_modifier.modify_config(drums=self.active_drums, voices=self.active_voices)
                            self.previous_active_drums = self.active_drums.copy()
                            self.previous_active_voices = self.active_voices.copy()

                    # Try to display the label
                    try:
                        with self.frame_lock:
                            cv2.putText(
                                frame, finger_label, (hand_x, hand_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.finger_colors[finger_label], 2, cv2.LINE_AA
                            )
                    except Exception as e:
                        print(f"Error while putting text on hand landmark: {e}")

            # Draw line between the midpoints of both hands (if both hands are detected)
            if len(hand_midpoints) == 2:
                hand_distance = math.sqrt((hand_midpoints[0][0] - hand_midpoints[1][0]) ** 2 + (hand_midpoints[0][1] - hand_midpoints[1][1]) ** 2)
                current_time = time.time()
                if self.prev_hand_distance is not None:
                    time_diff = current_time - self.prev_time
                    distance_ratio = hand_distance / self.clap_distance
                    velocity = (distance_ratio - (self.prev_hand_distance / self.clap_distance)) / time_diff
                    self.velocity_text = f"Velocity of Hand Distance: {velocity:.2f}x/sec"

                    # Detect clap based on velocity ratio
                    clap_detected = velocity < self.CLAP_THRESHOLD

                self.prev_hand_distance = hand_distance
                self.prev_time = current_time

                # Draw line between hand midpoints, change color if clap is detected
                line_color = self.RED if clap_detected else self.BLUE
                with self.frame_lock:
                    cv2.line(frame, hand_midpoints[0], hand_midpoints[1], line_color, 2)

                # Display "CLAP DETECTED" if a clap is detected
                midpoint_x = (hand_midpoints[0][0] + hand_midpoints[1][0]) // 2
                midpoint_y = (hand_midpoints[0][1] + hand_midpoints[1][1]) // 2 - 20
                clap_text = "CLAP NOT DETECTED"
                clap_text_color = self.WHITE
                if clap_detected:
                    clap_text = "CLAP DETECTED"
                    clap_text_color = self.RED
                    self.clap_start_time = time.time()
                with self.frame_lock:
                    cv2.putText(frame, clap_text, (midpoint_x - 50, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 1, clap_text_color, 2, cv2.LINE_AA)

                if self.clap_start_time is not None:
                    elapsed_time = time.time() - self.clap_start_time
                    if elapsed_time < self.CLAPTEXT_FADE:  # Display text for 0.5 seconds
                        # Calculate the fade effect by reducing the opacity over time (fades in 0.5 seconds)
                        fade_factor = max(0, 1 - (elapsed_time / self.CLAPTEXT_FADE))

                        # Create a transparent overlay
                        overlay = frame.copy()

                        # Set the text on the overlay
                        with self.frame_lock:
                            cv2.putText(overlay, "CLAP DETECTED", (int(w * 0.37), int(h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, self.RED, 5, cv2.LINE_AA)

                            # Blend the overlay with the original frame
                            cv2.addWeighted(overlay, fade_factor, frame, 1 - fade_factor, 0, frame)
                    else:
                        self.clap_start_time = None  # Reset clap_start_time after 0.5 seconds

        else:
            # Reset the active drums and voices if no hands are detected
            if self.active_drums or self.active_voices:
                self.active_drums.clear()
                self.active_voices.clear()
                config_modifier.modify_config(drums=self.active_drums, voices=self.active_voices)
            cv2.putText(frame, "HANDS NOT DETECTED", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, self.RED, 3, cv2.LINE_AA)

    def display_results(self, frame):
        y_offset = 20
        for length in self.lengths:
            cv2.putText(frame, length, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)
            y_offset += 20

        # Display the detected emotion on the top right corner
        if self.emotion_text:
            cv2.putText(frame, f"Emotion: {self.emotion_text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.WHITE, 2, cv2.LINE_AA)
            y_offset += 30

        # Display the velocity of hand distance
        if self.velocity_text:
            cv2.putText(frame, self.velocity_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1, cv2.LINE_AA)

        # Display the resulting frame
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.imshow('Video', frame)

    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            frame = cv2.resize(frame, (1920, 1080))  # Increase resolution for better visualization
            frame = cv2.flip(frame, 1)  # Mirror the image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe requires RGB format

            # Process the frame for facial landmarks
            face_results = self.face_mesh.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)

            self.lengths = []  # Reset lengths list
            self.emotion_text = ""
            self.velocity_text = ""
            clap_detected = False

            # Create threads
            face_thread = threading.Thread(target=self.process_face, args=(face_results, frame))
            hand_thread = threading.Thread(target=self.process_hands, args=(hand_results, frame))

            face_thread.start()
            hand_thread.start()

            face_thread.join()
            hand_thread.join()

            self.display_results(frame)

            # Exit if Backspace or 'q' is pressed
            key = cv2.waitKey(1)
            if key == 8 or key == ord('q'):  # ASCII code for Backspace is 8, fallback to 'q'
                break

        # Release the webcam and close the window
        config_modifier.modify_config(reset=True)  # all playing music should stop
        self.video_capture.release()
        cv2.destroyAllWindows()

def main():
    detector = FaceMusicDetector()
    detector.run()

if __name__ == '__main__':
    main()
