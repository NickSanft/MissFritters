import pyttsx3


class SimpleStuffSayer:
    engine = pyttsx3.init()

    def __init__(self):
        self.engine.setProperty('voice',
                                "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0")

    def print_staff_saying_config(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            # to get the info. about various voices in our PC
            print("Voice:")
            print("ID: %s" % voice.id)
            print("Name: %s" % voice.name)
            print("Age: %s" % voice.age)
            print("Gender: %s" % voice.gender)
            print("Languages Known: %s" % voice.languages)

    def say_stuff(self, message: str):
        print("Message to say: {}".format(message))
        self.engine.say(message)
        self.engine.runAndWait()
