import asyncio
import logging
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Optional

app = FastAPI()

# Load Whisper model (medium or small model is recommended)
model = WhisperModel("small", device="cpu", compute_type="int8")

@app.websocket("/TranscribeStreaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_queue: asyncio.Queue = asyncio.Queue()
    audio_chunks = []
    is_connected = True

    async def mic_stream():
        while is_connected:
            try:
                indata = await audio_queue.get()
                if indata is None:
                    break
                yield np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
            except Exception as e:
                logging.error(f"Error in mic_stream: {e}")
                break

    async def process_audio():
        try:
            async for chunk in mic_stream():
                if not is_connected:
                    break
                audio_chunks.append(chunk)
                if len(audio_chunks) >= 5:  # Process every 5 chunks
                    audio_data = np.concatenate(audio_chunks)
                    sf.write("temp_audio.wav", audio_data, 16000)
                    segments, _ = model.transcribe("temp_audio.wav")
                    transcript = " ".join(segment.text for segment in segments)
                    if is_connected:
                        await websocket.send_text(transcript)
                    audio_chunks.clear()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logging.error(f"Error in process_audio: {e}")

    audio_task = asyncio.create_task(process_audio())

    try:
        while True:
            try:
                message = await websocket.receive()
                if "bytes" in message:
                    await audio_queue.put(message["bytes"])
                elif "text" in message and message["text"] == "submit_response":
                    # Process any remaining chunks
                    if audio_chunks:
                        audio_data = np.concatenate(audio_chunks)
                        sf.write("temp_audio.wav", audio_data, 16000)
                        segments, _ = model.transcribe("temp_audio.wav")
                        transcript = " ".join(segment.text for segment in segments)
                        await websocket.send_text(transcript)
                    break
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    break
                raise
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    finally:
        is_connected = False
        await audio_queue.put(None)  # Signal mic_stream to stop
        try:
            await audio_task
        except Exception as e:
            logging.error(f"Error while cleaning up audio task: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=None)