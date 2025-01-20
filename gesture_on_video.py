import cv2
import torch
import mediapipe as mp
import numpy as np
import atexit

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Video file path
VIDEO_FILE_PATH = "/home/gpandit/learn/test_gesture.mp4"

# Load YOLOv5 model (use a small model like YOLOv5n for faster inference)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Ensure resources are released upon exit
cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Error: Could not open video file '{VIDEO_FILE_PATH}'.")

def cleanup():
    cap.release()
    cv2.destroyAllWindows()

atexit.register(cleanup)

# Function to draw landmarks and connections dynamically
def draw_landmarks_on_canvas(landmarks, connections, canvas, color=(255, 255, 255)):
    if landmarks:
        h, w, _ = canvas.shape
        # Draw landmarks
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(canvas, (x, y), 2, color, -1)
        # Draw connections
        if connections:
            for connection in connections:
                start = landmarks.landmark[connection[0]]
                end = landmarks.landmark[connection[1]]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(canvas, start_point, end_point, color, 1)

# Initialize the Holistic model
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True
) as holistic:
    print("Processing video... Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame.")
            break

        # Use YOLOv5 for person detection
        results = yolo_model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Get detection results

        # Create a blank canvas
        height, width, _ = frame.shape
        blank_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Process each detected person
        person_counter = 0
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) != 0:  # Class 0 is "person" in COCO
                continue

            person_counter += 1
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Extract person ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Convert ROI to RGB for MediaPipe
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            holistic_results = holistic.process(rgb_roi)

            # Draw landmarks on the blank canvas
            if holistic_results.pose_landmarks:
                draw_landmarks_on_canvas(holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, blank_canvas[y1:y2, x1:x2], color=(0, 255, 0))
            if holistic_results.face_landmarks:
                draw_landmarks_on_canvas(holistic_results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, blank_canvas[y1:y2, x1:x2], color=(255, 0, 0))
            if holistic_results.left_hand_landmarks:
                draw_landmarks_on_canvas(holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, blank_canvas[y1:y2, x1:x2], color=(0, 0, 255))
            if holistic_results.right_hand_landmarks:
                draw_landmarks_on_canvas(holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, blank_canvas[y1:y2, x1:x2], color=(255, 255, 0))

            # Add label above the person's head
            label = f"Person {person_counter}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Combine the original frame and the blank canvas side by side
        combined_frame = np.hstack((frame, blank_canvas))

        # Display the combined frame
        cv2.namedWindow('Original Frame + Landmarks on Canvas', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Frame + Landmarks on Canvas', combined_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Cleanup resources
cleanup()

