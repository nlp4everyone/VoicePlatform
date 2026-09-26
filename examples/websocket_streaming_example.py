import asyncio, base64, json, os, time
import numpy as np
import librosa
import websockets
from dotenv import load_dotenv

# This example targets vLLM's Realtime API (a WebSocket endpoint distinct from the
# `/v1/audio/transcriptions` REST endpoint used by the other examples in this folder).
# Requires extra client-side dependencies not needed by the other examples:
#   pip install websockets librosa numpy python-dotenv

def audio_to_pcm16_bytes(audio_path: str, sample_rate: int = 16000) -> bytes:
    """
    Load an audio file and convert it to raw PCM16 mono audio at the given sample rate.

    Args:
        audio_path (str): Path to the audio file to convert
        sample_rate (int): Target sample rate expected by the realtime API (default: 16000)

    Returns:
        bytes: Raw little-endian PCM16 audio bytes (not yet base64-encoded)
    """
    # Load and resample the audio to mono float32 samples at the target sample rate
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    # Scale floating point samples ([-1.0, 1.0]) into the signed 16-bit integer range
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


async def stream_websocket_response(ws_url: str,
                                    audio_path: str,
                                    model: str,
                                    chunk_size: int = 4096):
    """
    Perform realtime streaming transcription using the vLLM WebSocket Realtime API.

    Args:
        ws_url (str): WebSocket URL of the vLLM realtime endpoint (e.g. ws://localhost:8001/v1/realtime)
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        chunk_size (int): Size in bytes of each PCM16 audio chunk sent per message (default: 4096)

    Returns:
        None: Prints transcription results to stdout as they stream in
    """
    async with websockets.connect(ws_url) as ws:
        # Wait for the server to acknowledge the new session
        response = json.loads(await ws.recv())
        if response.get("type") != "session.created":
            print(f"Unexpected response: {response}")
            return
        print(f"Session created: {response['id']}")

        # Pin the session to the model currently served by vLLM
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # Signal that the client is ready to start streaming audio
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Convert the audio file to raw PCM16 @ 16kHz as expected by the API
        print(f"Loading audio from: {audio_path}")
        pcm16_bytes = audio_to_pcm16_bytes(audio_path)

        # Stream the audio to the server in fixed-size chunks
        total_chunks = (len(pcm16_bytes) + chunk_size - 1) // chunk_size
        print(f"Sending {total_chunks} audio chunks...")
        stream_start = time.perf_counter()
        for offset in range(0, len(pcm16_bytes), chunk_size):
            chunk = pcm16_bytes[offset : offset + chunk_size]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }))

        # Signal that no more audio will be sent for this turn
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        print("Audio sent. Waiting for transcription...\n")

        # Consume the streamed transcription deltas as they arrive
        print("transcription result [stream]:", end=" ")
        while True:
            response = json.loads(await ws.recv())
            response_type = response.get("type")
            if response_type == "transcription.delta":
                # Print the transcribed text immediately without newline
                print(response["delta"], end="", flush=True)
            elif response_type == "transcription.done":
                elapsed = time.perf_counter() - stream_start
                print(f"\n[Stream finished] Final transcription: {response['text']}")
                usage = response.get("usage")
                if usage:
                    print(f"Usage: {usage}")
                # Wall-clock estimate: covers sending the audio plus generation, so it is a
                # lower bound on the server's own decode speed rather than a pure benchmark.
                output_tokens = (usage or {}).get("completion_tokens")
                if output_tokens:
                    print(f"Estimated throughput: {output_tokens / elapsed:.1f} tokens/s "
                          f"({output_tokens} output tokens in {elapsed:.2f}s)")
                break
            elif response_type == "error":
                print(f"\n[Stream error]: {response['error']}")
                break


def main():
    """
    Main function to demonstrate realtime streaming audio transcription over WebSocket.
    Connects to the vLLM Realtime API endpoint and streams the sample audio for transcription.
    """
    # Read MODEL_NAME from the project's .env (searched upwards from this file)
    load_dotenv()

    # Default vLLM realtime WebSocket endpoint (adjust if your server runs on different port/host)
    ws_url = "ws://localhost:8001/v1/realtime"
    # Path to the audio file for transcription
    audio_path = "resources/sample_vi.mp3"
    # Model served by vLLM, matching MODEL_NAME in .env
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")

    print(f"Using model: {model_name}")
    # Run the websocket streaming function
    asyncio.run(stream_websocket_response(ws_url = ws_url,
                                          audio_path = audio_path,
                                          model = model_name))

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()
