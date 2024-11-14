import cv2
import mediapipe as mp

# Initialize mediapipe Face Mesh for detecting facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# Landmark indices for facial features
NOSE_TIP_INDEX = 1
MOUTH_TOP_INDEX = 13  # Upper lip
MOUTH_BOTTOM_INDEX = 14  # Lower lip
LEFT_EYE_INDEX = 33
RIGHT_EYE_INDEX = 263

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (800, 450))  # reduce resolution for faster processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe requires RGB format

    # Process the frame for facial landmarks
    results = face_mesh.process(rgb_frame)

    # Check if landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape  # frame dimensions for pixel conversion

            # Get coordinates for the nose tip
            nose_tip = face_landmarks.landmark[NOSE_TIP_INDEX]
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
            print(f"Nose Tip: x={nose_x}, y={nose_y}")

            # Get coordinates for the mouth and calculate mouth openness
            mouth_top = face_landmarks.landmark[MOUTH_TOP_INDEX]
            mouth_bottom = face_landmarks.landmark[MOUTH_BOTTOM_INDEX]
            mouth_top_y = mouth_top.y * h
            mouth_bottom_y = mouth_bottom.y * h
            mouth_openness = abs(mouth_bottom_y - mouth_top_y)
            print(f"Mouth Openness: {mouth_openness:.2f}")

            # Get coordinates for the left and right eyes
            left_eye = face_landmarks.landmark[LEFT_EYE_INDEX]
            right_eye = face_landmarks.landmark[RIGHT_EYE_INDEX]
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
            print(f"Left Eye: x={left_eye_x}, y={left_eye_y}")
            print(f"Right Eye: x={right_eye_x}, y={right_eye_y}")

            for idx, landmark in enumerate(face_landmarks.landmark):
              print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}")


            # Draw circles on the detected facial features for visualization
            cv2.circle(frame, (nose_x, nose_y), 2, GREEN, -1)
            cv2.circle(frame, (left_eye_x, left_eye_y), 2, BLUE, -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 2, BLUE, -1)
            cv2.circle(frame, (int(mouth_top.x * w), int(mouth_top.y * h)), 2, RED, -1)
            cv2.circle(frame, (int(mouth_bottom.x * w), int(mouth_bottom.y * h)), 2, RED, -1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if Backspace or 'q' is pressed
    key = cv2.waitKey(1)
    if key == 8 or key == ord('q'):  # ASCII code for Backspace is 8, fallback to 'q'
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
