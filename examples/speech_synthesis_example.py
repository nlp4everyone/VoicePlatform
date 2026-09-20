from openai import AsyncOpenAI
import asyncio, os, time
import openai
import httpx

# Directory where generated audio files are written
RESULTS_DIR = "results"

async def synthesize_with_openai(text: str,
                                 model: str,
                                 openai_api_base: str,
                                 output_path: str,
                                 language: str | None = None,
                                 response_format: str = "wav",
                                 api_key: str = "EMPTY") -> int:
    """
    Perform non-streaming speech synthesis using the official OpenAI client interface.

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated audio file
        language (str | None): Language hint, e.g. "Vietnamese" or "English" (default: auto-detect)
        response_format (str): Audio container format: wav, pcm, flac, mp3, opus (default: "wav")
        api_key (str): API key for the server (default: "EMPTY")

    Returns:
        int: Number of audio bytes written
    """

    # Initialize the async OpenAI client with local server configuration
    async with AsyncOpenAI(base_url = openai_api_base,
                           api_key = api_key) as client:
        # Request the whole audio clip in a single response
        response = await client.audio.speech.create(
            model = model,
            input = text,
            # OmniVoice has no built-in speakers: omit `voice` for an auto voice,
            # or pass ref_audio/ref_text to clone one (see voice_cloning_example.py)
            voice = openai.NOT_GIVEN,
            response_format = response_format,
            # vLLM-Omni extension: optional language hint for the model
            extra_body = {"language": language} if language else None,
        )
        # Read the full audio payload into memory
        audio_bytes = response.content

    # Persist the generated audio to disk
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    return len(audio_bytes)

async def synthesize_with_httpx(text: str,
                                model: str,
                                openai_api_base: str,
                                output_path: str,
                                language: str | None = None,
                                response_format: str = "wav",
                                timeout: float = 300.0) -> int:
    """
    Perform non-streaming speech synthesis using raw HTTP requests via the `httpx` library.

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated audio file
        language (str | None): Language hint, e.g. "Vietnamese" or "English" (default: auto-detect)
        response_format (str): Audio container format: wav, pcm, flac, mp3, opus (default: "wav")
        timeout (float): Maximum time in seconds to wait for the response (default: 300.0)

    Returns:
        int: Number of audio bytes written
    """

    # Construct the API endpoint URL for speech synthesis
    api_url = f"{openai_api_base}/audio/speech"
    # Set headers for the HTTP request
    headers = {"User-Agent": "Speech-Client"}
    # Set up the request payload for non-streaming synthesis
    payload = {
        "model": model,                       # Specify the model to use
        "input": text,                        # Text to synthesize
        "response_format": response_format,   # Audio container format
    }
    # Optional language hint; OmniVoice auto-detects when omitted
    if language:
        payload["language"] = language

    # Reuse a single async client so the connection is closed deterministically
    async with httpx.AsyncClient(timeout = timeout) as client:
        response = await client.post(api_url,
                                     headers = headers,
                                     json = payload)

    # Raise an informative error when the server rejects the request
    response.raise_for_status()
    # Persist the generated audio to disk
    with open(output_path, "wb") as f:
        f.write(response.content)
    return len(response.content)

async def run_synthesizer(label: str,
                          synthesize,
                          text: str,
                          model: str,
                          openai_api_base: str,
                          output_path: str,
                          language: str | None = None):
    """
    Await a single async synthesis client and report its result and timing.

    Args:
        label (str): Human readable name of the client being exercised
        synthesize (Callable): Async synthesis function to await
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the generated audio file
        language (str | None): Language hint passed through to the server (default: None)

    Returns:
        None: Prints synthesis results to stdout
    """

    try:
        # Record start time for processing measurement
        start_time = time.perf_counter()

        # Synthesize the text with the current client
        num_bytes = await synthesize(text = text,
                                     model = model,
                                     openai_api_base = openai_api_base,
                                     output_path = output_path,
                                     language = language)

        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        # Print synthesis results and processing metrics
        print(f"=== Speech Synthesis Results [{label}] ===")
        print(f"Output File: {output_path} ({num_bytes:,} bytes)")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print("=" * 35)

    except Exception as e:
        # Report the failure but let the other clients finish
        print(f"Speech synthesis failed [{label}]:", e)

async def main():
    """
    Main coroutine to demonstrate non-streaming speech synthesis through two
    different clients: the OpenAI SDK and the `httpx` library.
    """
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8002/v1"
    # Text to synthesize
    text = "Xin chào, đây là VoicePlatform. Chúc bạn một ngày tốt lành."
    # Model name for speech synthesis
    model_name = "kjanh/KhanhTTS-OmniVoice"
    # Language hint for the model (set to None to auto-detect)
    language = "Vietnamese"

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
                        openai_api_base = openai_api_base,
                        output_path = output_path,
                        language = language)
        for label, (synthesize, output_path) in synthesizers.items()
    ])

# Entry point: Run the main coroutine when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())