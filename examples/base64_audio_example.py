import base64, os, time
import requests

# Fun-ASR-MLT-Nano is served by vLLM as a transcription-only model: its prompt is built by
# the `/v1/audio/transcriptions` endpoint, not by chat completions, so base64 audio in a
# JSON chat body is not supported. When audio reaches you base64-encoded (e.g. from a
# browser or another API), decode it back to bytes and upload it as a multipart file.
# The server decodes any format ffmpeg supports, so no client-side conversion is needed.

# vLLM REST endpoint (adjust if your server runs on different port/host)
url = "http://localhost:8002/v1/audio/transcriptions"
# Path to the audio file for transcription
audio_path = "resources/sample_vi.mp3"
# Served model name (must match the server's --served-model-name)
model = "fun-asr-mlt-nano"
# Language code of the audio content (validated by the server, the model auto-detects)
language = "vi"
# Only needed when the server was started with --api-key / VLLM_API_KEY
api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

def encode_audio(path):
    """
    Encode audio file to base64 string format for API transmission.

    Args:
        path (str): Path to the audio file

    Returns:
        str: Base64 encoded audio data
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Encode the audio file to base64 format (stands in for audio received as base64)
audio_base64 = encode_audio(audio_path)

# Decode the base64 payload back to raw bytes for the multipart upload
audio_bytes = base64.b64decode(audio_base64)

# Prepare the multipart form: the file part plus OpenAI-style form fields
files = {"file": (os.path.basename(audio_path), audio_bytes)}
data = {
    "model": model,                 # Specify the model to use
    "language": language,           # Set transcription language
    "temperature": "0",             # Greedy decoding for deterministic output
    "response_format": "json",      # Request JSON response format
}
headers = {"Authorization": f"Bearer {api_key}"}

# Record start time for processing measurement
begin = time.perf_counter()

# Send request to the API (the endpoint returns only once the whole file is processed)
response = requests.post(url, headers=headers, files=files, data=data, timeout=300)
response.raise_for_status()

# Calculate processing time
end = time.perf_counter()
processing_time = end - begin

# Print transcription results and processing metrics
print("=== Base64 Audio Transcription Results ===")
print(f"Transcribed Text: {response.json()['text']}")
print(f"Processing Time: {processing_time:.4f} seconds")
print("=" * 42)
