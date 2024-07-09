import os
import random
import whisper

from pydub import AudioSegment


def language_detector(model, fileName):

    # if file[-4:] != '.mp3':
    #     wav_audio = AudioSegment.from_file("audio.wav", format="wav")
    #     raw_audio = AudioSegment.from_file("audio.wav", format="raw",
    #                                        frame_rate=44100, channels=2, sample_width=2)
    #
    #     wav_audio.export("audio.mp3", format="mp3")
    #     raw_audio.export("audio1.mp3", format="mp3")
    if not fileName.endswith('.mp3'):
        raise ValueError("File is not of type .mp3, Please ")
    awesome = AudioSegment.from_file(fileName, "")
    awesome.export("D:\Projects\Datasets\manjot_voice.mp3", format="mp3")

    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # print(f"Detected language: {max(probs, key=probs.get)}")
    return max(probs, key=probs.get)


## TESTS:

# tests=20
# audio_file = "D:\Projects\Datasets\manjot_voice.opus"
# test_dir = r"D:\Projects\Datasets\Indian_Languages_Audio_Dataset"

# model = whisper.load_model("small")

# for i in range(tests):
#     lang_dir = os.path.join(test_dir, random.choice(os.listdir(test_dir)))
#     file = random.choice(os.listdir(lang_dir))
#     audio_file = os.path.join(lang_dir, file)

#     convert file type

#     print(lang_dir.split('\\')[-1])
#     language_detector(model, audio_file)