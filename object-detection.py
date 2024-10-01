import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')  # Use a more accurate model if possible

# Initialize webcam with lower resolution
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create()

# Set to keep track of detected objects and their last detection time
detected_objects = {}
display_interval = 2  # seconds between re-displaying the same object
prev_keypoints = None
angle_change_threshold = 20  # Threshold for angle change detection
iou_threshold = 0.5  # Threshold for deciding if an object is a new instance

def preprocess_frame(frame):
    """Preprocess the frame to enhance visibility in various lighting conditions."""
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for better contrast in low-light conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_frame = clahe.apply(gray_frame)
    # Convert back to BGR for object detection
    processed_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)
    return processed_frame

def extract_text_from_frame(frame):
    """Extract text from a video frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def speak_text(text):
    """Convert text to speech."""
    if text.strip():  # Check if text is not empty
        print(f"Speaking: {text}")

def iou(box1, box2):
    """Calculate Intersection Over Union (IOU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    # Calculate the intersection area
    xi1 = max(x1, x1p)
    yi1 = max(y1, y1p)
    xi2 = min(x2, x2p)
    yi2 = min(y2, y2p)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate the union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2p - x1p) * (y2p - y1p)
    union_area = box1_area + box2_area - inter_area

    # Compute the IOU
    iou_value = inter_area / union_area if union_area > 0 else 0
    return iou_value

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess frame for better visibility
    processed_frame = preprocess_frame(frame)

    # Convert frame to grayscale for ORB processing
    gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

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
                results = model(processed_frame)

                # The results are returned as a list, so we need to access the first result
                result = results[0]  # Accessing the first result from the list

                # Extract detection details
                boxes = result.boxes  # Bounding boxes
                scores = result.boxes.conf  # Confidence scores for each box
                names = result.names  # Class names for each detected object

                for i, box in enumerate(boxes):
                    score = scores[i]
                    obj_name = names[int(box.cls)]
                    if score >= 0.7:  # Confidence threshold for detection
                        current_time = time.time()

                        # Extract box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        new_instance = True
                        for prev_box in detected_objects.get(obj_name, []):
                            if iou((x1, y1, x2, y2), prev_box) > iou_threshold:
                                new_instance = False
                                break

                        # If the object is new or the timer has expired, display it
                        if new_instance:
                            print(f"Detected: {obj_name} with confidence: {score:.2f}")
                            speak_text(f"Detected {obj_name}")

                            # Store the bounding box of the detected object
                            if obj_name not in detected_objects:
                                detected_objects[obj_name] = []
                            detected_objects[obj_name].append((x1, y1, x2, y2))

                # Render results on the frame
                annotated_frame = result.plot()

                # Display the resulting frames
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