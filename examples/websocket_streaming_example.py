import asyncio, json, os, time, wave
import websockets

# This example targets vLLM-Omni's streaming text-input TTS WebSocket endpoint
# (`/v1/audio/speech/stream`), distinct from the `/v1/audio/speech` REST endpoint used by the
# other examples in this folder. The client sends text incrementally (`input.text`), flushes
# it with `input.done`, and receives the generated audio back on the same connection.
# `input.done` ends one utterance but keeps the connection open, so several utterances can
# reuse a single WebSocket handshake.
# Requires: pip install websockets

# Directory where generated audio files are written
RESULTS_DIR = "results"

# VoxCPM2 emits 16-bit mono PCM at 48 kHz
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2

async def stream_websocket_speech(ws_url: str,
                                  utterances: list[list[str]],
                                  model: str,
                                  output_dir: str,
                                  timeout: float = 120.0):
    """
    Perform streaming speech synthesis over the vLLM-Omni WebSocket endpoint and write
    the PCM audio of every utterance to its own WAV file as it arrives.

    Args:
        ws_url (str): WebSocket URL of the streaming speech endpoint (e.g. ws://localhost:8002/v1/audio/speech/stream)
        utterances (list[list[str]]): Utterances to synthesize; each one is a list of text chunks sent one by one
        model (str): Name of the TTS model to use
        output_dir (str): Directory that receives one WAV file per utterance
        timeout (float): Maximum time in seconds to wait for each server message (default: 120.0)

    Returns:
        None: Prints streaming progress and per-utterance metrics to stdout
    """

    # The session config must be the first message; it stays in effect for every utterance
    session_config = {
        "type": "session.config",
        "model": model,
        "voice": "default",             # Placeholder, ignored by VoxCPM2
        "response_format": "pcm",       # stream_audio requires raw PCM
        "stream_audio": True,           # Send audio progressively instead of once per utterance
    }

    # max_size=None: audio frames can be larger than the library's default message limit
    async with websockets.connect(ws_url, max_size = None) as ws:
        await ws.send(json.dumps(session_config))

        for index, chunks in enumerate(utterances):
            output_path = os.path.join(output_dir, f"output_websocket_{index}.wav")
            start_time = time.perf_counter()
            first_chunk_time = None
            total_bytes = 0

            # Send the text piece by piece, as it would arrive from an LLM, then flush it
            for chunk in chunks:
                await ws.send(json.dumps({"type": "input.text", "text": chunk}))
            await ws.send(json.dumps({"type": "input.done"}))

            print(f"\n[Utterance {index}] {''.join(chunks)}")

            # Open the output WAV; the `wave` module writes a correct header on close
            with wave.open(output_path, "wb") as wav:
                wav.setnchannels(NUM_CHANNELS)
                wav.setsampwidth(SAMPLE_WIDTH)
                wav.setframerate(SAMPLE_RATE)

                # Receive until the server reports the end of this utterance
                while True:
                    message = await asyncio.wait_for(ws.recv(), timeout = timeout)

                    # Binary frames carry raw PCM audio
                    if isinstance(message, bytes):
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - start_time
                            print(f"First audio chunk after {first_chunk_time:.2f} seconds")
                        wav.writeframes(message)
                        total_bytes += len(message)
                        # Show progress without newline
                        print(".", end="", flush=True)
                        continue

                    # Text frames carry JSON control events
                    event = json.loads(message)
                    event_type = event.get("type")
                    if event_type == "session.done":
                        # Ends the flushed utterance, not the connection
                        break
                    if event_type == "error":
                        raise RuntimeError(f"Server error: {event.get('message')}")

            # Print per-utterance results and processing metrics
            processing_time = time.perf_counter() - start_time
            duration = total_bytes / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH)
            print(f"\n=== Websocket Speech Synthesis Results [utterance {index}] ===")
            print(f"Output File: {output_path} ({total_bytes:,} bytes, {duration:.2f} s of audio)")
            print(f"Processing Time: {processing_time:.2f} seconds (RTF {processing_time / duration:.3f})")
            print("=" * 35)

        # Tell the server we are done with this connection
        await ws.send(json.dumps({"type": "session.close"}))

async def main():
    """
    Main coroutine to demonstrate streaming speech synthesis over WebSocket, sending two
    utterances through a single connection.
    """
    # Default vLLM-Omni WebSocket endpoint (adjust if your server runs on different port/host)
    ws_url = "ws://localhost:8002/v1/audio/speech/stream"
    # Each utterance is a list of text chunks, sent one by one before the flush
    utterances = [
        ["Xin chào, đây là VoicePlatform. ",
         "Âm thanh được phát trực tiếp trong lúc mô hình đang tạo."],
        ["Chúc bạn một ngày tốt lành."],
    ]
    # Model name for speech synthesis
    model_name = "openbmb/VoxCPM2"
    # Make sure the results directory exists before writing into it
    os.makedirs(RESULTS_DIR, exist_ok = True)

    print(f"Using model: {model_name}")

    # Start streaming speech synthesis
    await stream_websocket_speech(ws_url = ws_url,
                                  utterances = utterances,
                                  model = model_name,
                                  output_dir = RESULTS_DIR)

# Entry point: Run the main coroutine when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())