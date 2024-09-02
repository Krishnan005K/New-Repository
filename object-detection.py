import cv2
import torch
import numpy as np
import time
import pytesseract
from translate import Translator
import pyttsx3

# Load YOLOv5 model weights (ensure the path is correct)
model_path = 'yolov5s.pt'  # You can choose yolov5s.pt, yolov5m.pt, or yolov5l.pt
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(2)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create()

# Set to keep track of detected objects and their last detection time
detected_objects = {}
display_interval = 5  # seconds between re-displaying the same object
prev_keypoints = None
angle_change_threshold = 20  # Threshold for angle change detection, tune this as needed

def extract_text_from_frame(frame):
    """Extract text from a video frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def translate_text(text, target_lang='en'):
    """Translate text to the target language using translate library."""
    translator = Translator(to_lang=target_lang)
    translation = translator.translate(text)
    return translation

def speak_text(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert frame to grayscale for ORB processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

    if prev_keypoints is not None and len(keypoints) > 0 and len(prev_keypoints) > 0:
        # Calculate the optical flow between the previous and current keypoints
        prev_pts = np.float32([kp.pt for kp in prev_keypoints]).reshape(-1, 1, 2)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, prev_pts, None)

        if curr_pts is not None and len(curr_pts) > 0:
            # Calculate the average movement of keypoints
            movement = np.abs(curr_pts - prev_pts).mean()

            if movement > angle_change_threshold:
                # Perform object detection
                results = model(frame)

                # Extract detection details
                for *box, conf, cls in results.pred[0]:
                    # Filter detections by confidence level
                    if conf >= 0.6:
                        obj_name = results.names[int(cls)]
                        current_time = time.time()

                        # If the object is new or the timer has expired, display it
                        if obj_name not in detected_objects or (current_time - detected_objects[obj_name] > display_interval):
                            print(f"Detected: {obj_name} with confidence: {conf:.2f}")
                            detected_objects[obj_name] = current_time

                            

                # Render results on the frame
                annotated_frame = results.render()[0]

                # Display the resulting frame
                cv2.imshow('Object Detection', annotated_frame)

    # Update the previous frame and keypoints
    prev_frame = gray_frame
    prev_keypoints = keypoints

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()