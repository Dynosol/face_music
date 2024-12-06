from cProfile import label
from turtle import right
import cv2
import mediapipe as mp
import math
import time
from deepface import DeepFace
import numpy as np
from rembg import remove
from PIL import Image

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

original_hand_distance = None

max_mouth_openness = 0
max_mouth_wideness = 0
min_mouth_wideness = 0.5

fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

finger_names = {
    mp_hands.HandLandmark.INDEX_FINGER_TIP: "Index Finger",
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP: "Middle Finger",
    mp_hands.HandLandmark.RING_FINGER_TIP: "Ring Finger",
    mp_hands.HandLandmark.PINKY_TIP: "Pinky Finger"
}

# A and F are thumbs!!!!
finger_labels = ["A", "SNARE", "BASS", "HI-HAT", "RIDE", "F", "SOLO", "CHOIR", "BRASS", "ORGAN"]

prev_hand_distance = None
prev_time = time.time() # for clapping!
start_time = time.time()
clap_start_time = None

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
    finger_length = float('inf') # for the thumb default

    # RESETING FACE RATIOS
    current_time = time.time()
    if current_time - start_time >= 5:
        # Reset values
        max_mouth_openness = 0
        max_mouth_wideness = 0
        min_mouth_wideness = 0.5
        # Reset the start time for the next interval
        start_time = current_time

    # Check if landmarks are detected for face
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
            max_mouth_wideness = max(head_width/2, mouth_wideness)
            max_wideness = head_width / 2
            mouth_openness_normalized = min(mouth_openness / max_mouth_openness, 1.0)
            mouth_wideness_normalized = min(mouth_wideness / max_mouth_wideness, 1.0)

            # normalize mouth widness for differing head widths
            min_mouth_wideness = min(mouth_wideness_normalized+0.05, min_mouth_wideness)

            scaled_mouth_wideness = ((mouth_wideness_normalized - min_mouth_wideness) / (1.0 - min_mouth_wideness))
            scaled_mouth_wideness = max(0, min(scaled_mouth_wideness, 1))  # Clamp between 0 and 100

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
            eye_max_open = head_height / 40
            lengths.append(f"Eye Max Openness: {eye_max_open:.2f}")
            left_eye_text = "Closed" if left_eye_openness < eye_max_open else "Open"
            right_eye_text = "Closed" if right_eye_openness < eye_max_open else "Open"
            lengths.append(f"Left Eye Openness: {left_eye_openness:.2f}")
            lengths.append(f"Right Eye Openness: {right_eye_openness:.2f}")

            # Draw circles on the detected facial features for visualization
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
            cv2.putText(frame, f"Wideness: {round(scaled_mouth_wideness * 100, 0)}%", (mouth_left_x, mouth_left_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{left_eye_text}", (left_eye_top_x, left_eye_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{right_eye_text}", (right_eye_top_x, right_eye_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

            # Extract the face ROI for emotion analysis
            x_min = min(int(mouth_left.x * w), left_eye_top_x, right_eye_top_x)
            y_min = min(int(mouth_top.y * h), left_eye_top_y, right_eye_top_y)
            x_max = max(int(mouth_right.x * w), left_eye_bottom_x, right_eye_bottom_x)
            y_max = max(int(mouth_bottom.y * h), left_eye_bottom_y, right_eye_bottom_y)

            face_roi = rgb_frame[y_min:y_max, x_min:x_max]

            # Perform emotion analysis
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion_text = result[0]['dominant_emotion']
            except:
                emotion_text = ""

    # Check if landmarks are detected for hands
    if hand_results.multi_hand_landmarks:
        hand_midpoints = []
        label_index = 0
        for hand_landmarks in hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape  # frame dimensions for pixel conversion
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

            # Calculate palm height
            wrist_y = wrist.y * h
            pinky_mcp_y = pinky_mcp.y * h
            palm_height = abs(pinky_mcp_y - wrist_y)
            pinch_threshold = palm_height / 2

            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Calculate midpoint between thumb and index finger
            midpoint_x = (thumb_x + index_x) // 2
            midpoint_y = (thumb_y + index_y) // 2
            hand_midpoints.append((midpoint_x, midpoint_y))

            # Draw midpoint
            cv2.circle(frame, (midpoint_x, midpoint_y), 4, BLUE, -1)

            for idx, landmark in enumerate(hand_landmarks.landmark):
                hand_x, hand_y = int(landmark.x * w), int(landmark.y * h)

                # Draw circles on the detected hand landmarks for visualization
                cv2.circle(frame, (hand_x, hand_y), 2, GREEN, -1)

                # Draw label on each finger landmark
                if idx in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                           mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                           mp_hands.HandLandmark.PINKY_TIP]:
                    # Draw lines between each finger tip and thumb tip
                    if idx in finger_names:
                        cv2.line(frame, (thumb_x, thumb_y), (hand_x, hand_y), GREEN, 1)
                        finger_length = math.sqrt((thumb_x - hand_x) ** 2 + (thumb_y - hand_y) ** 2)
                        lengths.append(f"Thumb to {finger_names[idx]}: {finger_length:.2f}")

                    # FINGER PINCH DETECTOR
                    if finger_length < pinch_threshold:
                        label_color = RED
                    else:
                        label_color = WHITE

                    # Try to put text, if fails continue without crashing
                    try:
                        cv2.putText(frame, finger_labels[label_index], (hand_x, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error while putting text on hand landmark: {e}")

                    label_index += 1

        # Draw line between the midpoints of both hands (if both hands are detected)
        if len(hand_midpoints) == 2:
            hand_distance = math.sqrt((hand_midpoints[0][0] - hand_midpoints[1][0]) ** 2 + (hand_midpoints[0][1] - hand_midpoints[1][1]) ** 2)
            current_time = time.time()
            if prev_hand_distance is not None:
                time_diff = current_time - prev_time
                if original_hand_distance is None:
                    original_hand_distance = hand_distance
                distance_ratio = hand_distance / original_hand_distance
                velocity = (distance_ratio - (prev_hand_distance / original_hand_distance)) / time_diff
                velocity_text = f"Velocity of Hand Distance: {velocity:.2f}x/sec"

                # Detect clap based on velocity ratio
                if velocity < CLAP_THRESHOLD:
                    clap_detected = True

            prev_hand_distance = hand_distance
            prev_time = current_time

            # Draw line between hand midpoints, change color if clap is detected
            line_color = RED if clap_detected else BLUE
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
            cv2.putText(frame, clap_text, (midpoint_x - 50, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 1, clap_text_color, 2, cv2.LINE_AA)

            if clap_start_time is not None:
                elapsed_time = time.time() - clap_start_time
                if elapsed_time < CLAPTEXT_FADE:  # Display text for 0.5 seconds
                    # Calculate the fade effect by reducing the opacity over time (fades in 0.5 seconds)
                    fade_factor = max(0, 1 - (elapsed_time / CLAPTEXT_FADE))

                    # Create a transparent overlay
                    overlay = frame.copy()
                    
                    # Set the text on the overlay
                    cv2.putText(overlay, "CLAP DETECTED", (int(w * 0.37), int(h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, RED, 5, cv2.LINE_AA)

                    # Blend the overlay with the original frame
                    cv2.addWeighted(overlay, fade_factor, frame, 1 - fade_factor, 0, frame)
                else:
                    clap_start_time = None  # Reset clap_start_time after 0.5 seconds

    # Display the lengths on the top right corner of the frame
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

    # Exit if Backspace or 'q' is pressed
    key = cv2.waitKey(1)
    if key == 8 or key == ord('q'):  # ASCII code for Backspace is 8, fallback to 'q'
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()