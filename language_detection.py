import os
import random
import whisper

model = whisper.load_model("small")

test_dir = r"D:\Projects\Datasets\Indian_Languages_Audio_Dataset"



for i in range(20):
    lang_dir = os.path.join(test_dir, random.choice(os.listdir(test_dir)))
    audio_file = os.path.join(lang_dir, random.choice(os.listdir(lang_dir)))
    print(lang_dir.split('\\')[-1])

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

#D:\Projects\Datasets\Indian_Languages_Audio_Dataset\Punjabi
#D:\Projects\Datasets\Indian_Languages_Audio_Dataset\Punjabi\9715.mp3
