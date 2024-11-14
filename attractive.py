import cv2
import mediapipe as mp
import math

# Initialize mediapipe Face Mesh for detecting facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

def get_landmark_coordinates(landmarks, index, image_width, image_height):
    landmark = landmarks[index]
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    return (x, y)

def euclidean_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )

def calculate_attractiveness(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    attractiveness_score = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get coordinates for the required landmarks
            landmarks = face_landmarks.landmark

            # Define landmark indices
            indices = {
                'left_face': 234,
                'right_face': 454,
                'chin': 152,
                'forehead': 10,
                'left_eye_outer': 33,
                'right_eye_outer': 263,
                'left_eye_inner': 133,
                'right_eye_inner': 362,
                'mouth_left': 61,
                'mouth_right': 291,
                'nose_left': 168,
                'nose_right': 197
            }

            coords = {}
            for name, idx in indices.items():
                coords[name] = get_landmark_coordinates(
                    landmarks, idx, w, h)

            # Compute distances
            face_width = euclidean_distance(
                coords['left_face'], coords['right_face'])
            face_height = euclidean_distance(
                coords['chin'], coords['forehead'])
            eye_width = euclidean_distance(
                coords['left_eye_outer'], coords['right_eye_outer'])
            eye_inner_distance = euclidean_distance(
                coords['left_eye_inner'], coords['right_eye_inner'])
            mouth_width = euclidean_distance(
                coords['mouth_left'], coords['mouth_right'])
            nose_width = euclidean_distance(
                coords['nose_left'], coords['nose_right'])

            # Compute ratios
            eye_width_ratio = eye_width / face_width
            eye_inner_ratio = eye_inner_distance / face_width
            mouth_width_ratio = mouth_width / face_width
            nose_width_ratio = nose_width / face_width
            face_height_ratio = face_height / face_width

            # Ideal ratios (approximate values from studies)
            ideal_ratios = {
                'eye_width_ratio': 0.46,
                'eye_inner_ratio': 0.28,
                'mouth_width_ratio': 0.38,
                'nose_width_ratio': 0.34,
                'face_height_ratio': 1.618  # Golden ratio
            }

            # Compute percentage deviations
            deviations = {
                'eye_width_deviation': abs(eye_width_ratio - ideal_ratios['eye_width_ratio']) / ideal_ratios['eye_width_ratio'],
                'eye_inner_deviation': abs(eye_inner_ratio - ideal_ratios['eye_inner_ratio']) / ideal_ratios['eye_inner_ratio'],
                'mouth_width_deviation': abs(mouth_width_ratio - ideal_ratios['mouth_width_ratio']) / ideal_ratios['mouth_width_ratio'],
                'nose_width_deviation': abs(nose_width_ratio - ideal_ratios['nose_width_ratio']) / ideal_ratios['nose_width_ratio'],
                'face_height_deviation': abs(face_height_ratio - ideal_ratios['face_height_ratio']) / ideal_ratios['face_height_ratio'],
            }

            """
            THERE MUST BE A BETTER WAY OF CALCULATING ATTRACTIVENESS
            """

            # Total deviation
            # total_deviation = sum(deviations.values())

            # # Compute attractiveness score using exponential decay
            # k = 2  # scaling factor, adjust this value as needed
            # attractiveness_score = 100 * math.exp(-k * total_deviation)
            # attractiveness_score = max(0, min(attractiveness_score, 100))  # Clamp between 0 and 100

            # Output the attractiveness score
            print(f"Attractiveness Score: {attractiveness_score:.2f}")

            # Draw circles on the detected facial features for visualization
            for point in coords.values():
                cv2.circle(frame, point, 2, GREEN, -1)

            # Display the attractiveness score on the frame
            cv2.putText(frame, f"Attractiveness Score: {attractiveness_score:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)
    else:
        print("No face detected.")

    return frame, attractiveness_score

def process_image(image_path):
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Image not found or unable to read.")
        return

    frame = cv2.resize(frame, (800, 450))
    processed_frame, score = calculate_attractiveness(frame)

    # Display the image with the attractiveness score
    cv2.imshow('Image', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    choice = input("Enter 'i' to process an image or 'v' for video feed: ").lower()

    if choice == 'i':
        image_path = input("Enter the path to the image file: ")
        process_image(image_path)
    elif choice == 'v':
        # Initialize the webcam
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            frame = cv2.resize(frame, (800, 450))
            processed_frame, score = calculate_attractiveness(frame)

            # Display the resulting frame
            cv2.imshow('Video', processed_frame)

            # Exit if Backspace or 'q' is pressed
            key = cv2.waitKey(1)
            if key == 8 or key == ord('q'):  # ASCII code for Backspace is 8, fallback to 'q'
                break

        # Release the webcam and close the window
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Please enter 'i' or 'v'.")
