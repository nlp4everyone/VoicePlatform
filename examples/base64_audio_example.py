import base64, os, time
import requests
from dotenv import load_dotenv

# Read MODEL_NAME from the project's .env (searched upwards from this file)
load_dotenv()

# API endpoint for chat completions
url = "http://localhost:8001/v1/chat/completions"
# Path to the audio file for transcription
audio_path = "resources/sample_vi.mp3"
# Model served by vLLM, matching MODEL_NAME in .env
model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")

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

# Encode the audio file to base64 format
audio_base64 = encode_audio(audio_path)

# Prepare the payload for the API request
payload = {
    "model": model_name,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                },
                {
                    "type": "text",
                    "text": "Transcribe this audio."
                }
            ]
        }
    ]
}

# Record start time for processing measurement
begin = time.perf_counter()

# Send request to the API
response = requests.post(url, json=payload)

# Calculate processing time
end = time.perf_counter()
processing_time = end - begin

# Print transcription results and processing metrics
print("=== Base64 Audio Transcription Results ===")
print(f"Transcribed Text: {response.json()['choices'][0]['message']['content']}")
print(f"Processing Time: {processing_time:.4f} seconds")
print("=" * 42)