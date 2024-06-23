import speech_recognition as sr


class StuffHearer:
    r = sr.Recognizer()

    def hear_stuff(self, microphone_device_no: int):
        mic_name = sr.Microphone.list_microphone_names()[microphone_device_no]
        print("Initializing using microphone device {}: {}".format(microphone_device_no, mic_name))
        with sr.Microphone(device_index=microphone_device_no) as source:
            self.r.adjust_for_ambient_noise(source, duration=1)
            print("Speak your truth!")
            audio = self.r.listen(source)
            try:
                result = self.r.recognize_google(audio)
                print("Here's what you said: {}".format(result))
            except:
                print("Couldn't hear crap, captain")
        return result

    def get_mic_device(self):
        mics = sr.Microphone.list_microphone_names()
        num_mics = len(mics)
        if num_mics == 0:
            raise TypeError("No microphones, bro!")
        if num_mics == 1:
            print("Only 1 mic, setting your mic to: {}".format(mics[0]))
            return 1

        index = 1
        for mic in mics:
            print("{}: {}".format(index, mic))
            index = index + 1
        num = int(input("Whoa whoa whoa, which of these device numbers is your microphone?"))

        if num < 1 or num > num_mics:
            raise TypeError("Non-existent mic!")

        return num
