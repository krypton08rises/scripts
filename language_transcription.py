import speech_recognition as sr

from pydub import AudioSegment


def language_transcriptor(fileName):
    ## TODO: convert all possible filetypes to wav
    # convert mp3 file to wav
    sound = AudioSegment.from_mp3(fileName)  # r"D:\Projects\Datasets\Indian_Languages_Audio_Dataset\Hindi\42.mp3")
    sound.export("transcript.wav", format="wav")

    # transcribe audio file
    AUDIO_FILE = "transcript.wav"
    # use the audio file as the audio source
    r = sr.Recognizer()

    txt = ""
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

        print("Transcription: " + r.recognize_google(audio))
        txt += r.recognize_google(audio) + "\n"
    return txt

# from openai import OpenAI
#
# def language_transcriptor(client, fileName):
#
#   # add must be mp3 functionality
#   audio_file = open(fileName, "rb")
#   transcript = client.audio.transcriptions.create(
#     file=audio_file,
#     model="whisper-1",
#     response_format="verbose_json",
#     timestamp_granularities=["word"]
#     )
#
#   return transcript.words

## tests for OPENAI
# client = OpenAI()
# language_transcriptor(client, r"D:\Projects\Datasets\Indian_Languages_Audio_Dataset\Hindi\42.mp3")
