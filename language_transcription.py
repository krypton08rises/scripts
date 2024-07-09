from openai import OpenAI
client = OpenAI()

def language_transcriptor(fileName):

  audio_file = open("speech.mp3", "rb")
  transcript = client.audio.transcriptions.create(
    file=audio_file,
    model="whisper-1",
    response_format="verbose_json",
    timestamp_granularities=["word"]
  )

  return transcript.words