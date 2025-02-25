import asyncio
import requests
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket
from io import BytesIO

app = FastAPI()

# Hugging Face API Details
HF_API_TOKEN = "your_huggingface_api_token"
MODEL_ID = "openai/whisper-large-v2"  # You can try "facebook/wav2vec2-large-960h" too
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

async def transcribe_audio(audio_data):
    """Send audio to Hugging Face API and get the transcript."""
    response = requests.post(API_URL, headers=HEADERS, files={"file": ("audio.wav", audio_data, "audio/wav")})
    return response.json().get("text", "")

@app.websocket("/TranscribeStreaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = BytesIO()

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio_buffer.write(message["bytes"])

                if audio_buffer.tell() > 16000 * 2 * 5:  # 5 seconds of audio
                    audio_buffer.seek(0)
                    transcript = await transcribe_audio(audio_buffer.read())
                    await websocket.send_text(transcript)
                    audio_buffer = BytesIO()  # Reset buffer
            elif "text" in message and message["text"] == "submit_response":
                break
    finally:
        await websocket.close()
