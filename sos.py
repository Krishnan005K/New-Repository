from twilio.rest import Client
import time
import keyboard

# Twilio credentials
TWILIO_ACCOUNT_SID = 'ACfa50b8bb5c57bb2d276c70265d596d20'
TWILIO_AUTH_TOKEN = 'd0d180c587f433f15a88d5acbc3a3dc3'
TWILIO_PHONE_NUMBER = '+15717652333'  # Your Twilio phone number
SOS_PHONE_NUMBER = '+919361250297'  # Recipient's phone number (e.g., '+919876543210')

# Default location
DEFAULT_LATITUDE = '10.926089137086475'
DEFAULT_LONGITUDE = '76.92545950330751'

# Configure the SOS key
SOS_KEY = 's'

def generate_maps_link(lat, lon):
    return f"http://maps.google.com/maps?q={lat},{lon}"

def send_sms(body, to):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=body,
            from_=TWILIO_PHONE_NUMBER,
            to=to
        )
        print(f"SOS SMS sent successfully! Message SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

def main():
    print(f"Press '{SOS_KEY}' to send an SOS SMS with location.")
    try:
        while True:
            if keyboard.is_pressed(SOS_KEY):
                print("SOS triggered!")
                # Use default location
                latitude = DEFAULT_LATITUDE
                longitude = DEFAULT_LONGITUDE
                # Create the message with location
                maps_link = generate_maps_link(latitude, longitude)
                sos_message = f"This is an SOS alert. Please respond immediately. Location: {maps_link}"
                send_sms(
                    body=sos_message,
                    to=SOS_PHONE_NUMBER
                )
                # Wait a bit to avoid multiple triggers
                time.sleep(5)
    except KeyboardInterrupt:
        print("Program terminated")

if __name__ == "__main__":
    main()
