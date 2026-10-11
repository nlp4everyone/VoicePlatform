from openai import AsyncOpenAI
import asyncio, os, time, wave

# Directory where generated audio files are written
RESULTS_DIR = "results"

# VieNeu-TTS v3 Turbo emits 16-bit mono PCM at 48 kHz
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2

async def stream_speech(text: str,
                        model: str,
                        voice: str,
                        openai_api_base: str,
                        output_path: str,
                        api_key: str = "EMPTY",
                        chunk_size: int = 8192):
    """
    Perform streaming speech synthesis using the async OpenAI client and write
    raw PCM chunks to a WAV file as soon as they arrive.

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        voice (str): Preset voice name (see GET /v1/voices)
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated WAV file
        api_key (str): API key for the server, i.e. VIENEU_API_KEY (default: "EMPTY")
        chunk_size (int): Size of chunks to read from the streaming response (default: 8192)

    Returns:
        None: Prints streaming progress to stdout
    """

    # Initialize the async OpenAI client with local server configuration
    async with AsyncOpenAI(base_url = openai_api_base,
                           api_key = api_key) as client:
        # Record start time so we can measure time-to-first-audio
        start_time = time.perf_counter()
        first_chunk_time = None
        total_bytes = 0

        # Open the output WAV; the `wave` module writes a correct header on close
        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(NUM_CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)

            # Ask the server for raw PCM bytes streamed as they are generated
            # (the default stream_format is a raw chunked body)
            async with client.audio.speech.with_streaming_response.create(
                model = model,
                input = text,
                voice = voice,
                response_format = "pcm",        # Streaming supports pcm/wav only
            ) as response:
                # Consume the response body chunk by chunk
                async for chunk in response.iter_bytes(chunk_size):
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter() - start_time
                        print(f"First audio chunk after {first_chunk_time:.2f} seconds")
                    # Append the PCM samples to the WAV file immediately
                    wav.writeframes(chunk)
                    total_bytes += len(chunk)
                    # Show progress without newline
                    print(".", end="", flush=True)

        # Print streaming results and processing metrics
        processing_time = time.perf_counter() - start_time
        duration = total_bytes / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH)
        rtf = processing_time / duration if duration else float("nan")
        print(f"\n=== Streaming Speech Synthesis Results [async] ===")
        print(f"Output File: {output_path} ({total_bytes:,} bytes, {duration:.2f} s of audio)")
        print(f"Processing Time: {processing_time:.2f} seconds (RTF {rtf:.3f})")
        print("=" * 35)

async def main():
    """
    Main coroutine to demonstrate asynchronous streaming speech synthesis.
    Sets up the AsyncOpenAI client and initiates raw audio streaming.
    """
    # Default server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8000/v1"
    # Text to synthesize
    text = "Xin chào, đây là VoicePlatform. Âm thanh được phát trực tiếp trong lúc mô hình đang tạo."
    # Model name for speech synthesis (reported back as-is by the server)
    model_name = "vieneu-v3-turbo"
    # Preset voice; list the available ones with GET /v1/voices
    voice = "Mai Anh"
    # Where to write the generated audio
    output_path = os.path.join(RESULTS_DIR, "output_async_stream.wav")
    # Make sure the results directory exists before writing into it
    os.makedirs(RESULTS_DIR, exist_ok = True)

    print(f"Using model: {model_name}")

    # Start streaming speech synthesis
    await stream_speech(text = text,
                        model = model_name,
                        voice = voice,
                        openai_api_base = openai_api_base,
                        output_path = output_path)

# Entry point: Run the main coroutine when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
