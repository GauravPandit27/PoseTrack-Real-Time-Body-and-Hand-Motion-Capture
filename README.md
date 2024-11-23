# PoseTrack-Real-Time-Body-and-Hand-Motion-Capture
This project leverages MediaPipe and OpenCV to detect body and hand landmarks in real-time from a webcam feed. The application visualizes detected landmarks on both a 2D plot and the video feed, with the added functionality of recording and replaying landmark data.

Features
Real-Time Detection: Detects body and hand landmarks using MediaPipe's Pose and Hands solutions.
Landmark Visualization: Visualizes landmarks on a 2D plot and overlays them on the video feed.
Recording Mode: Records landmark data for body and hands to a JSON file.
Replay Mode: Replays previously recorded landmark movements.
User-Friendly Controls:
W: Start recording.
E: Stop recording and save data to a JSON file.
A: Replay recorded data.
Q: Quit the application.


Requirements
Python 3.7+
OpenCV
MediaPipe
Matplotlib
NumPy

Use the key controls to interact with the application:
Press W to start recording landmarks.
Press E to stop recording and save to a JSON file.
Press A to replay a recording.
Press Q to quit.

File Structure
body_hand_landmarks.py: Main script for detection, recording, and replay.
Recorded Data: Saved as recording_YYYYMMDD_HHMMSS.json.
Future Improvements
Add support for multiple camera inputs.
Include 3D visualization for landmarks.
Improve replay controls (e.g., pause, rewind, fast-forward).
Export recorded data to other formats (e.g., CSV).


Acknowledgments
MediaPipe for the robust pose and hand detection solutions.
OpenCV for video processing.
Matplotlib for 2D plotting.
