import asyncio
import websockets
import soundfile as sf

async def send_audio_file(ws, file_path):
    """Send audio file data to WebSocket server."""
    with sf.SoundFile(file_path, 'r') as audio_file:
        while True:
            data = audio_file.buffer_read(1024, dtype='int16')  # Read in chunks
            if not data:
                break  # Stop when the file is fully read
            await ws.send(data)  # Send chunk to WebSocket server

    await ws.send("submit_response")  # Indicate end of transmission

async def receive_transcriptions(ws):
    """Receive transcriptions from WebSocket server."""
    async for message in ws:
        print(f"Received: {message}")

async def test_websocket(file_path):
    uri = "ws://localhost:8000/TranscribeStreaming"
    async with websockets.connect(uri) as ws:
        send_task = asyncio.create_task(send_audio_file(ws, file_path))
        print("task sent")
        receive_task = asyncio.create_task(receive_transcriptions(ws))
        print("task received")

        await asyncio.gather(send_task, receive_task, return_exceptions=True)
        await receive_task  # Ensure all messages are received
        await ws.close()

if __name__ == "__main__":
    audio_file_path = "/home/sriram/sriram_repo/speech_to_text/speech_to_text_fastapi/harvard.wav"  # Change this to your audio file path
    asyncio.run(test_websocket(audio_file_path))
