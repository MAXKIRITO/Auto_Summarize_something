import os
import queue
import sys
import sounddevice as sd
import torch
import torchaudio
import numpy as np
import logging
import pyaudio
from datetime import datetime
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize logging and directories
if not os.path.exists('log'):
    os.makedirs('log')

log_filename = datetime.utcnow().strftime('log/transcription_%Y-%m-%dT%H-%M-%SSZ.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize queues and other shared resources
q = queue.Queue()
transcriptions = []
summary_interval = 60  # summary interval in seconds
last_summary_time = datetime.utcnow()

# Initialize the speech recognition model
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
labels = bundle.get_labels()
if torch.cuda.is_available():
    model = model.to('cuda')

# Initialize the T5 model and tokenizer for summarization
tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_model.to(device)

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

def greedy_decoder(emissions, labels):
    """Greedy decoder for CTC."""
    _, indices = torch.max(emissions, dim=-1)
    indices = indices.squeeze(0)
    result = []
    for i in range(indices.size(0)):
        if i > 0 and indices[i] == indices[i - 1]:
            continue
        result.append(labels[indices[i]])
    return "".join(result).replace("|", " ").replace("-", "").strip()

def transcribe_audio(waveform):
    if torch.cuda.is_available():
        waveform = waveform.to('cuda')
    with torch.no_grad():
        emissions, _ = model(waveform)
    transcription = greedy_decoder(emissions.cpu(), labels)
    return transcription

def summarize_text_t5(text):
    # 構建輸入文本
    input_text = "summarize: " + text
    # 編碼文本
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    
    attention_mask = torch.ones(input_ids.shape, device=device)
    
    # 設置生成配置
    generate_config = {
        "max_length": 150,
        "num_beams": 2,
        "length_penalty": 2.0,
        "early_stopping": True,
        "attention_mask": attention_mask
    }
    
    # 生成摘要
    with torch.no_grad():
        summary_ids = t5_model.generate(input_ids, **generate_config)
    
    # 解碼摘要
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def audio_record():
    global last_summary_time

    device_index = select_microphone()

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback, device=device_index):
        print('#' * 80)
        print('Press Ctrl+C to stop the recording')
        print('#' * 80)

        buffer = b""
        while True:
            data = q.get()
            buffer += data
            if len(buffer) > 16000 * 2:  # Collect 1 second of audio
                audio_data = torch.FloatTensor(np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0).unsqueeze(0)
                if torch.cuda.is_available():
                    audio_data = audio_data.to('cuda')
                transcription = transcribe_audio(audio_data)
                print(f"Transcription: {transcription}")
                timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S.%fZ')
                transcriptions.append((timestamp_str, transcription))
                buffer = b""

            # Check if it's time to summarize
            if (datetime.utcnow() - last_summary_time).total_seconds() >= summary_interval:
                recent_texts = ' '.join([t[1] for t in transcriptions[-10:]])  # use last 10 transcriptions
                summary_text = summarize_text_t5(recent_texts)
                summary_timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S.%fZ')
                print(f"Summary at {summary_timestamp}: {summary_text}")
                last_summary_time = datetime.utcnow()

def main():
    # Start the audio recording thread
    audio_record()

if __name__ == "__main__":
    main()
