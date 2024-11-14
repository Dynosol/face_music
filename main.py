import cv2
import mediapipe as mp

# Initialize mediapipe Face Mesh for detecting facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

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
            # Print each landmark's x, y, z coordinates
            for idx, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                print(f"Landmark {idx}: x={x}, y={y}, z={z}")

                # Convert normalized coordinates to pixel coordinates for visualization
                h, w, _ = frame.shape
                x_px, y_px = int(x * w), int(y * h)
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if Backspace or 'q' is pressed
    key = cv2.waitKey(1)
    if key == 8 or key == ord('q'):  # ASCII code for Backspace is 8, fallback to 'q'
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()
