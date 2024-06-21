import os
import sys
import queue
import sounddevice as sd
import vosk
import threading
import logging
import json
from datetime import datetime
import pyaudio
import jieba

# 創建log資料夾
if not os.path.exists('log'):
    os.makedirs('log')

# 使用標準時間格式命名日誌文件
log_filename = datetime.utcnow().strftime('log/transcription_%Y-%m-%dT%H-%M-%SSZ.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

q = queue.Queue()

def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    microphones = []

    for i in range(0, num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            microphones.append((i, device_info.get('name')))

    p.terminate()
    return microphones

def select_microphone():
    microphones = list_microphones()
    for idx, (device_index, device_name) in enumerate(microphones):
        print(f"{idx}: {device_name} (Device Index: {device_index})")
    choice = int(input("Select a microphone by number: "))
    return microphones[choice][0]

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def auto_segment_transcription(text):
    punctuation = '。！？'
    segments = []
    segment = ''
    for char in text:
        segment += char
        if char in punctuation:
            segments.append(segment)
            segment = ''
    if segment:
        segments.append(segment)
    return segments

def audio_record():
    model_path = "model/vosk-model-small-en-us-0.15"  # 替換為解壓後的模型路徑
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return

    model = vosk.Model(model_path)
    device_index = select_microphone()

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback, device=device_index):
        print('#' * 80)
        print('Press Ctrl+C to stop the recording')
        print('#' * 80)

        rec = vosk.KaldiRecognizer(model, 16000)
        buffer = ""
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                text = json.loads(result).get('text', '')
                print(f"Transcription: {text}")
                if text.strip():
                    buffer += text
                    segmented_transcription = auto_segment_transcription(buffer)
                    for segment in segmented_transcription:
                        if segment.endswith(('。', '！', '？')):
                            print(f"Segmented Transcription: {segment}")
                            logging.info(segment)
                            buffer = buffer.replace(segment, '')
            else:
                partial_result = rec.PartialResult()
                print(f"Partial: {json.loads(partial_result).get('partial', '')}")

try:
    audio_thread = threading.Thread(target=audio_record)
    audio_thread.start()
    audio_thread.join()
except KeyboardInterrupt:
    print('Stopping recording')
