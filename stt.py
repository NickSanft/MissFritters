import speech_recognition as sr


class StuffHearer:
    def __init__(self):
        # Initialize the recognizer in the constructor for better flexibility
        self.recognizer = sr.Recognizer()

    def hear_stuff(self):
        """
        Listens to the microphone and recognizes speech.
        Returns the recognized text if successful, otherwise None.
        """
        with sr.Microphone() as source:
            self._adjust_for_noise(source)
            print("Speak your truth!")
            audio = self._listen_to_audio(source)

        if audio:
            result = self._recognize_audio(audio)
            return result
        return None

    def _adjust_for_noise(self, source):
        """
        Adjust the recognizer for ambient noise in the environment.
        """
        self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def _listen_to_audio(self, source):
        """
        Listens to audio from the microphone and returns the audio data.
        """
        try:
            return self.recognizer.listen(source)
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return None
        except Exception as e:
            print(f"Error listening to audio: {e}")
            return None

    def _recognize_audio(self, audio):
        """
        Recognizes speech from the audio and returns the result as text.
        """
        try:
            result = self.recognizer.recognize_google(audio)
            print(f"Here's what you said: {result}")
            return result
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio.")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
        return None
