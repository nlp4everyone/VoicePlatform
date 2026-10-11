from openai import AsyncOpenAI
import asyncio, os, time, wave
import httpx

# Directory where generated audio files are written
RESULTS_DIR = "results"

# VieNeu-TTS v3 Turbo emits 16-bit mono PCM at 48 kHz
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2

def write_wav(pcm: bytes, output_path: str) -> float:
    """
    Wrap raw PCM bytes in a WAV container with a correct header.

    The server streams audio as it is generated, so a `wav` response carries a header with
    an unknown length. Requesting `pcm` and writing the header here gives a regular file
    that every tool can seek in.

    Args:
        pcm (bytes): 16-bit mono PCM samples at 48 kHz
        output_path (str): Where to write the WAV file

    Returns:
        float: Duration of the audio in seconds
    """

    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(NUM_CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return len(pcm) / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH)

async def synthesize_with_openai(text: str,
                                 model: str,
                                 voice: str,
                                 openai_api_base: str,
                                 output_path: str,
                                 api_key: str = "EMPTY") -> float:
    """
    Perform speech synthesis using the official OpenAI client interface and save the
    complete clip.

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        voice (str): Preset voice name (see GET /v1/voices)
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated audio file
        api_key (str): API key for the server, i.e. VIENEU_API_KEY (default: "EMPTY")

    Returns:
        float: Duration of the generated audio in seconds
    """

    # Initialize the async OpenAI client with local server configuration
    async with AsyncOpenAI(base_url = openai_api_base,
                           api_key = api_key) as client:
        # Request the whole audio clip in a single response (raw PCM, see write_wav)
        response = await client.audio.speech.create(
            model = model,
            input = text,
            voice = voice,
            response_format = "pcm",
        )
        # Read the full audio payload into memory
        pcm = response.content

    # Persist the generated audio to disk
    return write_wav(pcm, output_path)

async def synthesize_with_httpx(text: str,
                                model: str,
                                voice: str,
                                openai_api_base: str,
                                output_path: str,
                                api_key: str = "EMPTY",
                                timeout: float = 300.0) -> float:
    """
    Perform speech synthesis using raw HTTP requests via the `httpx` library.

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        voice (str): Preset voice name (see GET /v1/voices)
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated audio file
        api_key (str): API key for the server, i.e. VIENEU_API_KEY (default: "EMPTY")
        timeout (float): Maximum time in seconds to wait for the response (default: 300.0)

    Returns:
        float: Duration of the generated audio in seconds
    """

    # Construct the API endpoint URL for speech synthesis
    api_url = f"{openai_api_base}/audio/speech"
    # Set headers for the HTTP request
    headers = {"User-Agent": "Speech-Client",
               "Authorization": f"Bearer {api_key}"}
    # Set up the request payload for non-streaming synthesis
    payload = {
        "model": model,                       # Specify the model to use
        "input": text,                        # Text to synthesize
        "voice": voice,                       # Preset voice name
        "response_format": "pcm",             # Raw PCM, wrapped into a WAV below
    }

    # Reuse a single async client so the connection is closed deterministically
    async with httpx.AsyncClient(timeout = timeout) as client:
        response = await client.post(api_url,
                                     headers = headers,
                                     json = payload)

    # Raise an informative error when the server rejects the request
    response.raise_for_status()
    # Persist the generated audio to disk
    return write_wav(response.content, output_path)

async def run_synthesizer(label: str,
                          synthesize,
                          text: str,
                          model: str,
                          voice: str,
                          openai_api_base: str,
                          output_path: str):
    """
    Await a single async synthesis client and report its result, timing and RTF.

    Args:
        label (str): Human readable name of the client being exercised
        synthesize (Callable): Async synthesis function to await
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        voice (str): Preset voice name (see GET /v1/voices)
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated audio file

    Returns:
        None: Prints synthesis results to stdout
    """

    try:
        # Record start time for processing measurement
        start_time = time.perf_counter()

        # Synthesize the text with the current client
        duration = await synthesize(text = text,
                                    model = model,
                                    voice = voice,
                                    openai_api_base = openai_api_base,
                                    output_path = output_path)

        # Calculate processing time and real-time factor (generation time / audio duration)
        processing_time = time.perf_counter() - start_time
        rtf = processing_time / duration if duration else float("nan")

        # Print synthesis results and processing metrics
        print(f"=== Speech Synthesis Results [{label}] ===")
        print(f"Output File: {output_path} ({duration:.2f} s of audio)")
        print(f"Processing Time: {processing_time:.2f} seconds (RTF {rtf:.3f})")
        print("=" * 35)

    except Exception as e:
        # Report the failure but let the other clients finish
        print(f"Speech synthesis failed [{label}]:", e)

async def main():
    """
    Main coroutine to demonstrate speech synthesis through two different clients:
    the OpenAI SDK and the `httpx` library.
    """
    # Default server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8000/v1"
    # Text to synthesize
    text = "Xin chào, đây là VoicePlatform. Chúc bạn một ngày tốt lành."
    # Model name for speech synthesis (reported back as-is by the server)
    model_name = "vieneu-v3-turbo"
    # Preset voice; list the available ones with GET /v1/voices
    voice = "Mai Anh"

    # Map each client label to its corresponding async synthesis function and output file
    synthesizers = {
        "OpenAI SDK": (synthesize_with_openai, os.path.join(RESULTS_DIR, "output_openai.wav")),
        "httpx": (synthesize_with_httpx, os.path.join(RESULTS_DIR, "output_httpx.wav")),
    }
    # Make sure the results directory exists before writing into it
    os.makedirs(RESULTS_DIR, exist_ok = True)

    # Fire both clients concurrently and wait until all of them complete
    await asyncio.gather(*[
        run_synthesizer(label = label,
                        synthesize = synthesize,
                        text = text,
                        model = model_name,
                        voice = voice,
                        openai_api_base = openai_api_base,
                        output_path = output_path)
        for label, (synthesize, output_path) in synthesizers.items()
    ])

# Entry point: Run the main coroutine when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
