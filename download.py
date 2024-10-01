from gtts import gTTS
from io import BytesIO
import tempfile
import subprocess

# The text that you want to convert to audio
mytext = 'Welcome to GeeksforGeeks Joe!'

# Language in which you want to convert
language = 'en'

# Create a BytesIO object to hold the audio data
audio_data = BytesIO()

# Convert text to speech and write it into the BytesIO object
tts = gTTS(text=mytext, lang=language, slow=False)
tts.write_to_fp(audio_data)

# Rewind the BytesIO object to the beginning
audio_data.seek(0)

# Use a temporary file to play the audio without saving permanently
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
    temp_audio_file.write(audio_data.read())
    temp_file_name = temp_audio_file.name

# Close the file first, then play the audio
subprocess.run(["start", temp_file_name], shell=True)

# Clean up the temporary file
os.remove(temp_file_name)
