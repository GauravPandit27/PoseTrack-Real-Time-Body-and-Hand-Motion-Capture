import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import atexit

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Webcam not detected.")

# Flags and data for recording and replaying
recording = False
replaying = False
recorded_data = []
replay_index = 0

def save_recording(data):
    """Save recorded data to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Recording saved as {filename}")

def draw_landmarks_on_blank(image, body_landmarks, hand_landmarks):
    """Draw landmarks on a blank image."""
    height, width, _ = image.shape

    # Draw body landmarks
    if body_landmarks:
        for lm in body_landmarks:
            x, y = int(lm[0] * width), int(lm[1] * height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # Define body connections
        body_connections = [
            (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),  # Right arm
            (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),  # Left arm
            (23, 25), (25, 27), (27, 29), (29, 31),  # Right leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Left leg
            (11, 23), (12, 24), (23, 24)  # Shoulders to hips
        ]
        for start, end in body_connections:
            if start < len(body_landmarks) and end < len(body_landmarks):
                x1, y1 = int(body_landmarks[start][0] * width), int(body_landmarks[start][1] * height)
                x2, y2 = int(body_landmarks[end][0] * width), int(body_landmarks[end][1] * height)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Draw hand landmarks
    if hand_landmarks:
        for hand in hand_landmarks:
            for lm in hand:
                x, y = int(lm[0] * width), int(lm[1] * height)
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

            # Define hand connections
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (5, 6), (6, 7), (7, 8),  # Index
                (9, 10), (10, 11), (11, 12),  # Middle
                (13, 14), (14, 15), (15, 16),  # Ring
                (17, 18), (18, 19), (19, 20)  # Pinky
            ]
            for start, end in hand_connections:
                if start < len(hand) and end < len(hand):
                    x1, y1 = int(hand[start][0] * width), int(hand[start][1] * height)
                    x2, y2 = int(hand[end][0] * width), int(hand[end][1] * height)
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image

def mirror_landmarks(landmarks, flip_x_only=True):
    """Mirror the landmarks along the X-axis (horizontally)."""
    mirrored_landmarks = []
    for lm in landmarks:
        if flip_x_only:
            mirrored_x = 1 - lm[0]  # Flip the x-coordinate
            mirrored_landmarks.append((mirrored_x, lm[1]))  # Keep the y-coordinate the same
        else:
            mirrored_x = 1 - lm[0]  # Flip the x-coordinate
            mirrored_y = lm[1]  # Flip the y-coordinate too, if needed
            mirrored_landmarks.append((mirrored_x, mirrored_y))
    return mirrored_landmarks

def cleanup():
    """Ensure resources are cleaned up on exit."""
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")

atexit.register(cleanup)

def main():
    global recording, replaying, replay_index

    print("Starting webcam... Press 'q' to quit, 'w' to record, 'e' to stop recording, 'a' to replay.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame capture failed.")
                break

            # Flip the frame by 1 (horizontal flip) for the webcam
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape

            # Create a blank image for visualization
            blank_image = np.zeros_like(frame)

            if replaying:
                if replay_index < len(recorded_data):
                    body_landmarks = recorded_data[replay_index].get("body", [])
                    hand_landmarks = recorded_data[replay_index].get("hands", [])
                    replay_index += 1
                else:
                    replaying = False
                    replay_index = 0
                    print("Replay finished.")
                    continue
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)
                hand_results = hands.process(rgb_frame)

                # Extract body and hand landmarks
                body_landmarks = []
                if pose_results.pose_landmarks:
                    body_landmarks = [(lm.x, 1 - lm.y) for lm in pose_results.pose_landmarks.landmark]

                hand_landmarks = []
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks_data in hand_results.multi_hand_landmarks:
                        hand_landmarks.append([(lm.x, 1 - lm.y) for lm in hand_landmarks_data.landmark])

                # Mirror the hand and body landmarks
                hand_landmarks = [mirror_landmarks(hand) for hand in hand_landmarks]
                body_landmarks = mirror_landmarks(body_landmarks, flip_x_only=True)

                if recording:
                    recorded_data.append({"body": body_landmarks, "hands": hand_landmarks})

            # Draw landmarks on blank image
            blank_image = draw_landmarks_on_blank(blank_image, body_landmarks, hand_landmarks)

            # Rotate the blank image by 180 degrees (this rotates the visualization, not the webcam feed)
            blank_image = cv2.rotate(blank_image, cv2.ROTATE_180)

            # Combine both images for side-by-side display
            combined_image = np.hstack((frame, blank_image))

            # Display instructions and combined image
            cv2.putText(frame, "Press 'w' to Record, 'e' to Stop, 'a' to Replay, 'q' to Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if recording:
                cv2.putText(frame, "Recording...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif replaying:
                cv2.putText(frame, "Replaying...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Body and Hand Landmarks", combined_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Quit
                break
            elif key == ord("w"):  # Start recording
                recording = True
                recorded_data = []
                print("Recording started.")
            elif key == ord("e"):  # Stop recording
                recording = False
                save_recording(recorded_data)
                print("Recording stopped.")
            elif key == ord("a"):  # Replay recording
                if recorded_data:
                    replaying = True
                    replay_index = 0
                    print("Replaying recording.")
                else:
                    print("No recording to replay.")

    except KeyboardInterrupt:
        print("Program interrupted.")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
