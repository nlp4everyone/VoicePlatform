import base64, json, os, time, wave
import requests

# Directory where generated audio files are written
RESULTS_DIR = "results"

# VoxCPM2 emits 16-bit mono PCM at 48 kHz
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2

def stream_api_response(text: str,
                        model: str,
                        openai_api_base: str,
                        output_path: str,
                        chunk_size: int = 8192):
    """
    Perform streaming speech synthesis using raw HTTP requests to the vLLM API server
    and consume OpenAI-style `speech.audio.*` Server-Sent Events.

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated WAV file
        chunk_size (int): Size of chunks to read from streaming response (default: 8192)

    Returns:
        None: Prints streaming progress to stdout
    """

    # Construct the API endpoint URL for speech synthesis
    api_url = f"{openai_api_base}/audio/speech"
    # Set headers for the HTTP request
    headers = {"User-Agent": "Speech-Client"}
    # Set up the request payload for SSE streaming synthesis
    payload = {
        "stream": True,                 # Enable streaming (SSE by default)
        "model": model,                 # Specify the model to use
        "input": text,                  # Text to synthesize
        "voice": "default",             # Placeholder, ignored by VoxCPM2
        "response_format": "pcm",       # Streaming supports pcm/wav only
    }

    # Record start time so we can measure time-to-first-audio
    start_time = time.perf_counter()
    first_chunk_time = None
    total_bytes = 0

    # Open the output WAV; the `wave` module writes a correct header on close
    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(NUM_CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)

        # Make POST request with streaming enabled
        response = requests.post(api_url,
                                 headers = headers,
                                 json = payload,
                                 stream = True)
        response.raise_for_status()

        # Process the SSE stream line by line; only `data:` lines carry payloads
        for line in response.iter_lines(chunk_size = chunk_size,
                                        decode_unicode = False,
                                        delimiter = b"\n"):
            if not line.startswith(b"data: "):
                continue
            # Remove the "data: " prefix and parse the JSON event
            event = json.loads(line[len("data: "):].decode("utf-8"))
            event_type = event.get("type")

            if event_type == "speech.audio.delta":
                # Decode the base64 PCM chunk and append it to the WAV file
                chunk = base64.b64decode(event["audio"])
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - start_time
                    print(f"First audio chunk after {first_chunk_time:.2f} seconds")
                wav.writeframes(chunk)
                total_bytes += len(chunk)
                # Show progress without newline
                print(".", end="", flush=True)

            elif event_type == "speech.audio.done":
                # Final event carries token usage; stop reading the stream
                print(f"\n[Stream finished, usage: {event.get('usage')}]")
                break

            elif event_type == "speech.audio.error":
                # Surface server-side failures and stop
                raise RuntimeError(event["error"]["message"])

    # Print streaming results and processing metrics
    processing_time = time.perf_counter() - start_time
    duration = total_bytes / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH)
    print(f"=== Streaming Speech Synthesis Results [sync/SSE] ===")
    print(f"Output File: {output_path} ({total_bytes:,} bytes, {duration:.2f} s of audio)")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print("=" * 35)

def main():
    """
    Main function to demonstrate streaming speech synthesis over Server-Sent Events.
    """
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8002/v1"
    # Text to synthesize
    text = "Xin chào, đây là VoicePlatform. Âm thanh được phát trực tiếp trong lúc mô hình đang tạo."
    # Model name for speech synthesis
    model_name = "openbmb/VoxCPM2"
    # Where to write the generated audio
    output_path = os.path.join(RESULTS_DIR, "output_sync_stream.wav")
    # Make sure the results directory exists before writing into it
    os.makedirs(RESULTS_DIR, exist_ok = True)

    print(f"Using model: {model_name}")

    # Start streaming speech synthesis
    stream_api_response(text = text,
                        model = model_name,
                        openai_api_base = openai_api_base,
                        output_path = output_path)

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()
