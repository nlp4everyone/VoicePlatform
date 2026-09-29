from openai import AsyncOpenAI
import asyncio, os, time
import httpx

# Targets vLLM's OpenAI-compatible REST endpoint (`/v1/audio/transcriptions`) serving
# Fun-ASR-MLT-Nano. The whole file is uploaded and the full transcript is returned once the
# server has processed it. The `model` field must match the server's --served-model-name.
# Fun-ASR detects the spoken language itself; `language` is only validated (vLLM defaults to
# "en" when it is omitted) and does not steer the output.

async def transcribe_with_openai(audio_path: str,
                                 model: str,
                                 openai_api_base: str,
                                 language: str = "en",
                                 api_key: str = "EMPTY") -> str:
    """
    Perform asynchronous audio transcription using the official OpenAI client interface.

    Args:
        audio_path (str): Path to the audio file to transcribe
        model (str): Served model name (must match the server's --served-model-name)
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str): Language code for transcription (default: "en")
        api_key (str): API key of the server, or a placeholder when auth is disabled (default: "EMPTY")

    Returns:
        str: The transcribed text
    """

    # Initialize the async OpenAI client with local server configuration
    async with AsyncOpenAI(base_url = openai_api_base,
                           api_key = api_key) as client:
        # Open audio file and send to transcription service
        with open(audio_path, "rb") as audio_file:
            transcription = await client.audio.transcriptions.create(
                model = model,
                file = audio_file,
                language = language,
                temperature = 0,
            )

    # The OpenAI client returns a parsed object exposing the text attribute
    return transcription.text

async def transcribe_with_httpx(audio_path: str,
                                model: str,
                                openai_api_base: str,
                                language: str = "en",
                                api_key: str = "EMPTY",
                                timeout: float = 300.0) -> str:
    """
    Perform asynchronous audio transcription using raw HTTP requests via the `httpx` library.

    Args:
        audio_path (str): Path to the audio file to transcribe
        model (str): Served model name (must match the server's --served-model-name)
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str): Language code for transcription (default: "en")
        api_key (str): API key of the server, or a placeholder when auth is disabled (default: "EMPTY")
        timeout (float): Maximum time in seconds to wait for the response (default: 300.0)

    Returns:
        str: The transcribed text
    """

    # Construct the API endpoint URL for audio transcription
    api_url = f"{openai_api_base}/audio/transcriptions"
    # Set headers for the HTTP request (the bearer token is only checked when the
    # server was started with --api-key / VLLM_API_KEY)
    headers = {"User-Agent": "Transcription-Client",
               "Authorization": f"Bearer {api_key}"}
    # Open the audio file in binary mode for upload
    with open(audio_path, "rb") as f:
        # Prepare the file for multipart form data upload
        files = {"file": (os.path.basename(audio_path), f)}
        # Set up the request parameters for non-streaming transcription
        data = {
            "model": model,                 # Specify the model to use
            "language": language,           # Set transcription language
            "response_format": "json",      # Request JSON response format
            "temperature": "0",             # Greedy decoding for deterministic output
        }

        # Reuse a single async client so the connection is closed deterministically
        async with httpx.AsyncClient(timeout = timeout) as client:
            response = await client.post(api_url,
                                         headers = headers,
                                         files = files,
                                         data = data)

    # Raise an informative error when the server rejects the request
    response.raise_for_status()
    # Extract the transcribed text from the JSON payload
    return response.json()["text"]

async def run_transcriber(label: str,
                          transcribe,
                          audio_path: str,
                          model: str,
                          openai_api_base: str,
                          language: str = "en",
                          api_key: str = "EMPTY"):
    """
    Await a single async transcription client and report its result and timing.

    Args:
        label (str): Human readable name of the client being exercised
        transcribe (Callable): Async transcription function to await
        audio_path (str): Path to the audio file to transcribe
        model (str): Name of the model to use for transcription
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str): Language code for transcription (default: "en")
        api_key (str): API key of the server (default: "EMPTY")

    Returns:
        None: Prints transcription results to stdout
    """

    try:
        # Record start time for processing measurement
        start_time = time.perf_counter()

        # Transcribe the audio file with the current client
        text = await transcribe(audio_path = audio_path,
                                model = model,
                                openai_api_base = openai_api_base,
                                language = language,
                                api_key = api_key)

        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        # Print transcription results and processing metrics
        print(f"=== Audio Transcription Results [{label}] ===")
        print(f"Transcribed Text: {text}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print("=" * 35)

    except Exception as e:
        # Report the failure but let the other clients finish
        print(f"Audio transcription failed [{label}]:", e)

async def main():
    """
    Main coroutine to demonstrate asynchronous audio transcription through two
    different clients: the OpenAI SDK and the `httpx` library.
    """
    # Only needed when the server was started with --api-key / VLLM_API_KEY
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    # Default vLLM endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8002/v1"
    # Path to the audio file for transcription
    audio_path = "resources/sample_vi.mp3"
    # Language code of the audio content (validated by the server, the model auto-detects)
    language = "vi"

    # Ask the server which model it serves so requests use the right `model` name
    async with AsyncOpenAI(base_url = openai_api_base,
                           api_key = api_key) as client:
        model_name = (await client.models.list()).data[0].id
    print(f"Using model: {model_name}")

    # Map each client label to its corresponding async transcription function
    transcribers = {
        "OpenAI SDK": transcribe_with_openai,
        "httpx": transcribe_with_httpx,
    }

    # Fire both clients concurrently and wait until all of them complete
    await asyncio.gather(*[
        run_transcriber(label = label,
                        transcribe = transcribe,
                        audio_path = audio_path,
                        model = model_name,
                        openai_api_base = openai_api_base,
                        language = language,
                        api_key = api_key)
        for label, transcribe in transcribers.items()
    ])

# Entry point: Run the main coroutine when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())
