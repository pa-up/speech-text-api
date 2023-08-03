from fastapi import FastAPI
import soundfile as sf
import speech_recognition as sr
from speech_recognition import UnknownValueError
import numpy as np
from pydantic import BaseModel

class RequestData(BaseModel):
  audio_list: list
  wav_file_path: str
  samplerate: int

app = FastAPI()

def speech_to_text(wav_filename):
    audio_recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_filename) as source:
            audio = audio_recognizer.record(source)  # 音声ファイルを読み込む
            recognized_text = audio_recognizer.recognize_google(audio, language="ja-JP")
    except (UnknownValueError , ValueError):
        recognized_text = ""
    return recognized_text

@app.post("/")
def speech_text(data: RequestData):
    audio_list = data.audio_list
    wav_file_path = data.wav_file_path
    samplerate = data.samplerate
    audio_np = np.array(audio_list).astype(np.int16)
    sf.write(wav_file_path, audio_np, samplerate=samplerate)
    recognized_text = speech_to_text(wav_file_path)
    return {"recognized_text": recognized_text}