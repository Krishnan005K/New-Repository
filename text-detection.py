import cv2
import easyocr
import keyboard
import time

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def extract_text_from_frame(frame):
    """Extract text from a video frame using EasyOCR."""
    # Convert the frame to RGB (EasyOCR expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use EasyOCR to extract text
    results = reader.readtext(rgb_frame)
    
    # Extract text from results
    text = ' '.join([result[1] for result in results])
    return text

def main():
    cap = cv2.VideoCapture(1)  # Change to the correct camera index if needed
    key_pressed = False  # Track if the key was pressed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the live video feed
        cv2.imshow('Webcam Feed', frame)

        # Check for key presses using the keyboard package
        if keyboard.is_pressed('s'):
            if not key_pressed:  # Process only if the key was not previously pressed
                key_pressed = True
                print("Key 's' pressed.")
                # Extract text from the current frame
                text = extract_text_from_frame(frame)
                if text.strip():  # Check if text is not empty
                    print(f"Extracted Text: {text}")
                else:
                    print("No text detected.")
            # Small delay to prevent rapid repeated detections
            time.sleep(0.1)
        else:
            key_pressed = False  # Reset the key press status when the key is not pressed

        # Break loop on 'q' key press
        if keyboard.is_pressed('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
