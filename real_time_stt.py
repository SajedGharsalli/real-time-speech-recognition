import whisper
import pyaudio
import numpy as np
import asyncio
import threading
import warnings
import scipy.signal
from collections import deque

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load the Whisper model
model = whisper.load_model("base")

# Audio stream settings
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
BUFFER_SIZE = RATE * 5  # Buffer size for 5 seconds of audio
OVERLAP = CHUNK // 2    # 50% overlap

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

# Circular buffer to accumulate audio data
audio_buffer = deque(maxlen=BUFFER_SIZE)

def preprocess_audio(audio_data):
    nyquist = 0.5 * RATE
    low = 300 / nyquist
    b, a = scipy.signal.butter(1, low, btype='high')
    filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
    return filtered_audio

def transcribe_audio(audio_to_transcribe):
    try:
        result = model.transcribe(audio_to_transcribe, language="en")
        print(result['text'])
    except Exception as e:
        print(f"Error during transcription: {e}")

async def process_audio():
    global audio_buffer
    while True:
        if len(audio_buffer) >= BUFFER_SIZE:
            audio_to_transcribe = np.array(audio_buffer)
            audio_buffer.clear()  # Clear the buffer after extracting audio for transcription
            audio_to_transcribe = audio_to_transcribe.astype(np.float32)
            # Transcribe in a separate thread to avoid blocking
            threading.Thread(target=transcribe_audio, args=(audio_to_transcribe,)).start()
        await asyncio.sleep(0.01)  # Short sleep to avoid busy waiting

def audio_stream():
    global audio_buffer
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_data = preprocess_audio(audio_data)
            audio_buffer.extend(audio_data)  # Add new audio data to the circular buffer

            # Maintain overlap
            if len(audio_buffer) > BUFFER_SIZE:
                overlap_data = np.array(audio_buffer)[-OVERLAP:]
                audio_buffer = deque(overlap_data, maxlen=BUFFER_SIZE)
        except IOError as e:
            print(f"Audio stream error: {e}")

# Run audio stream and processing
loop = asyncio.get_event_loop()
loop.run_in_executor(None, audio_stream)
loop.run_until_complete(process_audio())