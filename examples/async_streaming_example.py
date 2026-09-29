import asyncio, json, os
import httpx

# vLLM streams transcriptions over Server-Sent Events: POST the file to the OpenAI-compatible
# endpoint (`/v1/audio/transcriptions`) with `stream=true`, and the server replies with
# `data: {...}` lines, each carrying the next piece of text in `choices[0].delta.content`,
# followed by `data: [DONE]`. The audio is uploaded in full first; only the text streams.
# Requires: pip install httpx

async def stream_asr_response(api_url: str,
                              audio_path: str,
                              model: str,
                              language: str = "vi",
                              api_key: str = "EMPTY",
                              timeout: float = 300.0):
    """
    Perform asynchronous streaming transcription over the vLLM transcription API.

    Args:
        api_url (str): URL of the `/v1/audio/transcriptions` endpoint
        audio_path (str): Path to the audio file to transcribe
        model (str): Served model name (must match the server's --served-model-name)
        language (str): Language code of the audio content (default: "vi")
        api_key (str): API key of the server, or a placeholder when auth is disabled (default: "EMPTY")
        timeout (float): Maximum time in seconds to wait for the response (default: 300.0)

    Returns:
        None: Prints transcription results to stdout as they stream in
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model,
        "language": language,
        "temperature": "0",
        "stream": "true",
    }

    async with httpx.AsyncClient(timeout = timeout) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f)}
            async with client.stream("POST", api_url,
                                     headers = headers,
                                     files = files,
                                     data = data) as response:
                response.raise_for_status()

                # Print header for streaming output
                print("transcription result [stream]:", end=" ")
                async for line in response.aiter_lines():
                    # SSE events are `data: <payload>` lines separated by blank lines
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {}).get("content")
                    if delta:
                        print(delta, end="", flush=True)

    # Print final newline after streaming completes
    print("\n[Stream finished]")


def main():
    """
    Main function to demonstrate asynchronous streaming audio transcription.
    """
    # Only needed when the server was started with --api-key / VLLM_API_KEY
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    # Default vLLM endpoint (adjust if your server runs on different port/host)
    api_url = "http://localhost:8002/v1/audio/transcriptions"
    # Path to the audio file for transcription
    audio_path = "resources/sample_vi.mp3"

    # Run the async streaming function
    asyncio.run(stream_asr_response(api_url = api_url,
                                    audio_path = audio_path,
                                    model = "fun-asr-mlt-nano",
                                    language = "vi",
                                    api_key = api_key))

# Entry point: Run the main function when script is executed directly
if __name__ == "__main__":
    main()
