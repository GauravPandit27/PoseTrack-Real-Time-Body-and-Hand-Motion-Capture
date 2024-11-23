import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from datetime import datetime

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

# Create a 2D Plot for stickman animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Set the window to fullscreen
cv2.namedWindow("Body and Hand Landmarks", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Body and Hand Landmarks", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Starting webcam... Displaying body and hand landmarks.")

# Flags and data for recording and replaying
recording = False
replaying = False
recorded_data = []  # Store recorded frames with landmarks
replay_index = 0    # Track the replay frame index

# Helper function to save recorded data
def save_recording(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Recording saved as {filename}")

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Handle replay mode
        if replaying:
            if replay_index < len(recorded_data):
                # Retrieve landmarks from the recorded data
                body_landmarks_2d = recorded_data[replay_index].get("body", [])
                hand_landmarks_2d = recorded_data[replay_index].get("hands", [])
                replay_index += 1
            else:
                # End replay mode and reset
                replaying = False
                replay_index = 0
                print("Replay finished. Resuming normal detection.")
                continue
        else:
            # Process the frame for body and hand landmarks
            pose_results = pose.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            # Extract 2D body landmarks
            body_landmarks_2d = []
            if pose_results.pose_landmarks:
                for lm in pose_results.pose_landmarks.landmark:
                    body_landmarks_2d.append((lm.x, 1 - lm.y))  # Invert Y for correct orientation
            
            # Extract 2D hand landmarks
            hand_landmarks_2d = []
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    hand_landmarks_2d.append([(lm.x, 1 - lm.y) for lm in hand_landmarks.landmark])

            # Record data if in recording mode
            if recording:
                recorded_data.append({"body": body_landmarks_2d, "hands": hand_landmarks_2d})
        
        # Clear previous 2D plot
        ax.cla()

        # Plot body landmarks
        if body_landmarks_2d:
            body_x = [lm[0] for lm in body_landmarks_2d]
            body_y = [lm[1] for lm in body_landmarks_2d]
            ax.scatter(body_x, body_y, color='r', marker='o', label='Body Landmarks')

            # Define and draw connections
            body_connections = [
                # Face connections
                (8, 6), (6, 5), (5, 4), (4, 8),  # Left Eye
                (1, 2), (2, 3), (3, 7), (7, 1),  # Right Eye
                (0, 1), (0, 2), (0, 3), (0, 7), (0, 6), (0, 5), (0, 4),  # Nose to eyes
                (9, 10),  # Mouth

                # Shoulders and arms
                (12, 11),  # Shoulders
                (12, 14),  # Left shoulder to left elbow
                (14, 18), (18, 20), (20, 22),  # Left arm
                (11, 13),  # Right shoulder to right elbow
                (13, 15), (15, 19), (19, 17),  # Right arm

                # Waist and torso
                (24, 23),  # Waist
                (11, 23), (12, 24),  # Shoulders to waist

                # Legs
                (24, 26), (26, 28), (28, 30),  # Left leg
                (23, 25), (25, 27), (27, 29),  # Right leg
                (29, 31), (31, 33),  # Right foot additional connections
                (30, 32), (32, 34),  # Left foot additional connections
            ]
            for start, end in body_connections:
                if start < len(body_landmarks_2d) and end < len(body_landmarks_2d):
                    ax.plot([body_landmarks_2d[start][0], body_landmarks_2d[end][0]],
                            [body_landmarks_2d[start][1], body_landmarks_2d[end][1]], 'k-', lw=2)

        # Plot hand landmarks
        if hand_landmarks_2d:
            for hand in hand_landmarks_2d:
                hand_x = [lm[0] for lm in hand]
                hand_y = [lm[1] for lm in hand]
                ax.scatter(hand_x, hand_y, color='b', marker='o', label='Hand Landmarks')
                
                hand_connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (5, 6), (6, 7), (7, 8),  # Index
                    (9, 10), (10, 11), (11, 12),  # Middle
                    (13, 14), (14, 15), (15, 16),  # Ring
                    (17, 18), (18, 19), (19, 20)  # Pinky
                ]
                for start, end in hand_connections:
                    if start < len(hand) and end < len(hand):
                        ax.plot([hand[start][0], hand[end][0]], [hand[start][1], hand[end][1]], 'b-', lw=2)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.draw()
        plt.pause(0.01)
        
        # Display landmarks on frame
        if not replaying and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if not replaying and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display webcam feed
        cv2.imshow("Body and Hand Landmarks", frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('w'):  # Start recording
            recording = True
            recorded_data = []  # Reset recorded data
            print("Recording started.")
        elif key == ord('e'):  # Stop recording
            recording = False
            save_recording(recorded_data)
            print("Recording stopped.")
        elif key == ord('a'):  # Replay recording
            if recorded_data:
                replaying = True
                replay_index = 0
                print("Replaying recording.")
            else:
                print("No recording to replay.")
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

