import cv2
import easyocr
import keyboard
from gtts import gTTS
from io import BytesIO
import tempfile
import os
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

def speak_text(text):
    """Convert text to speech using gTTS."""
    if text.strip():  # Check if text is not empty
        try:
            # Create a BytesIO object to hold the audio data
            audio_data = BytesIO()

            # Convert text to speech and write it into the BytesIO object
            tts = gTTS(text=text, lang='en', slow=False)
            tts.write_to_fp(audio_data)

            # Rewind the BytesIO object to the beginning
            audio_data.seek(0)

            # Use a temporary file to play the audio without saving permanently
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(audio_data.read())
                temp_file_name = temp_audio_file.name

            # Ensure the file is closed before trying to access it
            os.startfile(temp_file_name)

            # Give some time for the audio to play before deleting the file
            time.sleep(5)

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
    else:
        print("No text detected.")
        # Notify if no text was detected
        speak_text("No text detected.")

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
                    # Convert text to speech
                    speak_text(text)
                else:
                    # Notify if no text was detected
                    speak_text("No text detected.")
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
