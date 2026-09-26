import asyncio, base64, json, os, time
import numpy as np
import librosa
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Alternative to `websocket_streaming_example.py`: instead of talking to vLLM's Realtime
# WebSocket API through the raw `websockets` library, this uses the OpenAI Python SDK's
# `realtime.connect()` helper for the connection/auth handshake. vLLM's realtime events
# (transcription.delta / transcription.done / a top-level "model" on session.update) are
# not part of OpenAI's own Realtime API schema, so messages are still sent/received as raw
# JSON via `send_raw` / `recv_bytes` rather than the SDK's typed event helpers.
# Requires extra client-side dependencies not needed by the other examples:
#   pip install "openai[realtime]" librosa numpy python-dotenv

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


async def stream_realtime_via_openai_sdk(client: AsyncOpenAI,
                                         audio_path: str,
                                         model: str,
                                         chunk_size: int = 4096):
    """
    Perform realtime streaming transcription against vLLM's WebSocket Realtime API,
    using the OpenAI SDK's `realtime.connect()` context manager for the connection
    instead of the raw `websockets` library.

    Args:
        client (AsyncOpenAI): Async OpenAI client instance pointed at the vLLM server
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        chunk_size (int): Size in bytes of each PCM16 audio chunk sent per message (default: 4096)

    Returns:
        None: Prints transcription results to stdout as they stream in
    """
    # `realtime.connect()` derives the WebSocket URL from the client's base_url
    # (http(s):// -> ws(s)://) and appends "/realtime", matching vLLM's `/v1/realtime` route
    async with client.realtime.connect() as connection:
        # Wait for the server to acknowledge the new session
        response = json.loads(await connection.recv_bytes())
        if response.get("type") != "session.created":
            print(f"Unexpected response: {response}")
            return
        print(f"Session created: {response['id']}")

        # Pin the session to the model currently served by vLLM
        await connection.send_raw(json.dumps({"type": "session.update", "model": model}))

        # Signal that the client is ready to start streaming audio
        await connection.send_raw(json.dumps({"type": "input_audio_buffer.commit"}))

        # Convert the audio file to raw PCM16 @ 16kHz as expected by the API
        print(f"Loading audio from: {audio_path}")
        pcm16_bytes = audio_to_pcm16_bytes(audio_path)

        # Stream the audio to the server in fixed-size chunks
        total_chunks = (len(pcm16_bytes) + chunk_size - 1) // chunk_size
        print(f"Sending {total_chunks} audio chunks...")
        stream_start = time.perf_counter()
        for offset in range(0, len(pcm16_bytes), chunk_size):
            chunk = pcm16_bytes[offset : offset + chunk_size]
            await connection.send_raw(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }))

        # Signal that no more audio will be sent for this turn
        await connection.send_raw(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        print("Audio sent. Waiting for transcription...\n")

        # Consume the streamed transcription deltas as they arrive. `recv_bytes()` (raw
        # JSON) is used instead of the SDK's typed `recv()`, since vLLM's transcription.*
        # events would otherwise fail to parse against OpenAI's own Realtime event schema.
        print("transcription result [stream]:", end=" ")
        while True:
            response = json.loads(await connection.recv_bytes())
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
    Main function to demonstrate realtime streaming audio transcription over WebSocket
    using the OpenAI SDK's realtime client instead of the raw `websockets` library.
    """
    # Read MODEL_NAME from the project's .env (searched upwards from this file)
    load_dotenv()

    # vLLM uses "EMPTY" as a placeholder API key when not requiring authentication
    openai_api_key = "EMPTY"
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8001/v1"
    # Path to the audio file for transcription
    audio_path = "resources/sample_vi.mp3"
    # Model served by vLLM, matching MODEL_NAME in .env
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")

    # Initialize asynchronous OpenAI client with custom base URL and empty API key
    client = AsyncOpenAI(api_key = openai_api_key,
                         base_url = openai_api_base)
    print(f"Using model: {model_name}")
    # Run the websocket streaming function
    asyncio.run(stream_realtime_via_openai_sdk(client = client,
                                               audio_path = audio_path,
                                               model = model_name))

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()
