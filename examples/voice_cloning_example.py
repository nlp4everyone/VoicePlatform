import base64, os, time
import requests

# Directory where generated audio files are written
RESULTS_DIR = "results"

def encode_audio_to_data_url(audio_path: str) -> str:
    """
    Encode a local audio file as a base64 data URL accepted by the `ref_audio` field.

    Args:
        audio_path (str): Path to the reference audio file

    Returns:
        str: A `data:<mime>;base64,<payload>` string
    """

    # Pick the MIME type from the file extension (fall back to WAV)
    ext = audio_path.lower().rsplit(".", 1)[-1]
    mime = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
    }.get(ext, "audio/wav")
    # Read the file and base64 encode its bytes
    with open(audio_path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{payload}"

def clone_voice(text: str,
                model: str,
                openai_api_base: str,
                ref_audio_path: str,
                output_path: str,
                ref_text: str | None = None,
                response_format: str = "wav",
                timeout: float = 300.0) -> int:
    """
    Synthesize speech in the voice of a reference recording (zero-shot voice cloning).

    Args:
        text (str): Text to synthesize
        model (str): Name of the TTS model to use
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        ref_audio_path (str): Path to the reference audio whose voice should be cloned
        output_path (str): Where to write the generated audio file
        ref_text (str | None): Transcript of the reference audio. When given, the model
            continues from the reference for a closer match (default: None)
        response_format (str): Audio container format: wav, pcm, flac, mp3, opus (default: "wav")
        timeout (float): Maximum time in seconds to wait for the response (default: 300.0)

    Returns:
        int: Number of audio bytes written
    """

    # Construct the API endpoint URL for speech synthesis
    api_url = f"{openai_api_base}/audio/speech"
    # Set headers for the HTTP request
    headers = {"User-Agent": "Speech-Client"}
    # Set up the request payload: the reference audio drives the cloned voice
    payload = {
        "model": model,                                       # Specify the model to use
        "input": text,                                        # Text to synthesize
        "voice": "default",                                   # Placeholder, ignored by VoxCPM2
        "ref_audio": encode_audio_to_data_url(ref_audio_path),  # Reference voice (base64)
        "response_format": response_format,                   # Audio container format
    }
    # Optionally attach the reference transcript for better voice matching
    if ref_text:
        payload["ref_text"] = ref_text

    # Send the request and wait for the whole clip
    response = requests.post(api_url,
                             headers = headers,
                             json = payload,
                             timeout = timeout)
    # Raise an informative error when the server rejects the request
    response.raise_for_status()
    # Persist the generated audio to disk
    with open(output_path, "wb") as f:
        f.write(response.content)
    return len(response.content)

def main():
    """
    Main function to demonstrate voice cloning from a local reference recording.
    """
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8002/v1"
    # Reference recording whose voice will be cloned
    ref_audio_path = "resources/sample_vi.wav"
    # Transcript of the reference recording (optional, improves similarity)
    ref_text = None
    # Text to synthesize with the cloned voice
    text = "Xin chào, đây là giọng nói được nhân bản từ file mẫu."
    # Model name for speech synthesis
    model_name = "openbmb/VoxCPM2"
    # Where to write the generated audio
    output_path = os.path.join(RESULTS_DIR, "output_cloned.wav")
    # Make sure the results directory exists before writing into it
    os.makedirs(RESULTS_DIR, exist_ok = True)

    print(f"Using model: {model_name}")
    print(f"Reference audio: {ref_audio_path} ({os.path.getsize(ref_audio_path):,} bytes)")

    try:
        # Record start time for processing measurement
        start_time = time.perf_counter()
        # Synthesize the text with the cloned voice
        num_bytes = clone_voice(text = text,
                                model = model_name,
                                openai_api_base = openai_api_base,
                                ref_audio_path = ref_audio_path,
                                output_path = output_path,
                                ref_text = ref_text)
        # Calculate processing time
        processing_time = time.perf_counter() - start_time

        # Print synthesis results and processing metrics
        print("=== Voice Cloning Results ===")
        print(f"Output File: {output_path} ({num_bytes:,} bytes)")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print("=" * 35)

    except Exception as e:
        print("Voice cloning failed:", e)

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()