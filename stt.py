import speech_recognition as sr


class StuffHearer:
    r = sr.Recognizer()

    def hear_stuff(self):
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source, duration=1)
            print("Speak your truth!")
            audio = self.r.listen(source)
            try:
                result = self.r.recognize_google(audio)
                print("Here's what you said: {}".format(result))
            except:
                print("Couldn't hear crap, captain")
        return result
