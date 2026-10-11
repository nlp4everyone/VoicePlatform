import os, time, wave
import requests

# Directory where generated audio files are written
RESULTS_DIR = "results"

# VieNeu-TTS v3 Turbo emits 16-bit mono PCM at 48 kHz
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2

def enroll_voice(name: str,
                 openai_api_base: str,
                 ref_audio_path: str,
                 description: str = "",
                 denoise: bool = True,
                 api_key: str = "EMPTY",
                 timeout: float = 120.0) -> dict:
    """
    Enroll a short reference recording as a named voice via `POST /v1/voices`.

    VieNeu-TTS clones in two steps: the clip (3-8 s, up to 20 MB) is uploaded once under a
    name, then any request can use it with `voice=<name>`. Enrolled voices live in the
    server's memory, so they are lost when the container restarts.

    Args:
        name (str): Voice name: 1-64 letters, digits, spaces, '.', '-' or '_' (built-in names are rejected with 409)
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        ref_audio_path (str): Path to the reference audio whose voice should be cloned
        description (str): Optional short label shown by GET /v1/voices (default: "")
        denoise (bool): Denoise the reference clip before enrolling (default: True)
        api_key (str): API key for the server, i.e. VIENEU_API_KEY (default: "EMPTY")
        timeout (float): Maximum time in seconds to wait for the response (default: 120.0)

    Returns:
        dict: The enrolled voice as returned by the server
    """

    # Construct the API endpoint URL for voice enrollment
    api_url = f"{openai_api_base}/voices"
    # Set headers for the HTTP request
    headers = {"User-Agent": "Speech-Client",
               "Authorization": f"Bearer {api_key}"}
    # The clip is sent as multipart form data, not JSON
    data = {"name": name,
            "denoise": str(denoise).lower(),
            "description": description}

    with open(ref_audio_path, "rb") as f:
        response = requests.post(api_url,
                                 headers = headers,
                                 data = data,
                                 files = {"file": (os.path.basename(ref_audio_path), f)},
                                 timeout = timeout)
    # Raise an informative error when the server rejects the clip
    response.raise_for_status()
    return response.json()

def clone_voice(text: str,
                model: str,
                voice: str,
                openai_api_base: str,
                output_path: str,
                api_key: str = "EMPTY",
                timeout: float = 300.0) -> float:
    """
    Synthesize speech with an enrolled (cloned) voice.

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        voice (str): Name of the voice enrolled with `enroll_voice`
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
    # Set up the request payload: the enrolled voice name selects the cloned voice
    payload = {
        "model": model,                       # Specify the model to use
        "input": text,                        # Text to synthesize
        "voice": voice,                       # Enrolled voice name
        "response_format": "pcm",             # Raw PCM, wrapped into a WAV below
    }

    # Send the request and wait for the whole clip
    response = requests.post(api_url,
                             headers = headers,
                             json = payload,
                             timeout = timeout)
    # Raise an informative error when the server rejects the request
    response.raise_for_status()

    # Persist the generated audio to disk with a correct WAV header
    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(NUM_CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(response.content)
    return len(response.content) / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH)

def main():
    """
    Main function to demonstrate voice cloning from a local reference recording.
    """
    # Default server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8000/v1"
    # Reference recording whose voice will be cloned
    ref_audio_path = "resources/sample_vi.wav"
    # Name the cloned voice is enrolled under
    voice_name = "sample_vi"
    # Text to synthesize with the cloned voice
    text = "Xin chào, đây là giọng nói được nhân bản từ file mẫu."
    # Model name for speech synthesis (reported back as-is by the server)
    model_name = "vieneu-v3-turbo"
    # Where to write the generated audio
    output_path = os.path.join(RESULTS_DIR, "output_cloned.wav")
    # Make sure the results directory exists before writing into it
    os.makedirs(RESULTS_DIR, exist_ok = True)

    print(f"Using model: {model_name}")
    print(f"Reference audio: {ref_audio_path} ({os.path.getsize(ref_audio_path):,} bytes)")

    try:
        # Enroll the reference recording once; re-running replaces the same name
        voice = enroll_voice(name = voice_name,
                             openai_api_base = openai_api_base,
                             ref_audio_path = ref_audio_path)
        print(f"Enrolled voice: {voice['name']}")

        # Record start time for processing measurement
        start_time = time.perf_counter()
        # Synthesize the text with the cloned voice
        duration = clone_voice(text = text,
                               model = model_name,
                               voice = voice_name,
                               openai_api_base = openai_api_base,
                               output_path = output_path)
        # Calculate processing time and real-time factor (generation time / audio duration)
        processing_time = time.perf_counter() - start_time
        rtf = processing_time / duration if duration else float("nan")

        # Print synthesis results and processing metrics
        print("=== Voice Cloning Results ===")
        print(f"Output File: {output_path} ({duration:.2f} s of audio)")
        print(f"Processing Time: {processing_time:.2f} seconds (RTF {rtf:.3f})")
        print("=" * 35)

    except Exception as e:
        print("Voice cloning failed:", e)

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()
