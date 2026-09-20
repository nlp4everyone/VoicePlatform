import asyncio, io, os, re, time, wave
import httpx

# Directory where generated audio files are written
RESULTS_DIR = "results"

def split_sentences(text: str) -> list[str]:
    """
    Split a paragraph into sentences on terminal punctuation so each one can be
    synthesized as an independent request.

    Args:
        text (str): Paragraph to split

    Returns:
        list[str]: Non-empty sentences with surrounding whitespace removed
    """

    # Cut after ., !, ? or … followed by whitespace, keeping the punctuation
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in parts if p]

async def synthesize_sentence(client: httpx.AsyncClient,
                              text: str,
                              model: str,
                              openai_api_base: str,
                              language: str | None = None) -> bytes:
    """
    Request one sentence as a complete WAV clip from the server.

    Args:
        client (httpx.AsyncClient): Shared HTTP client
        text (str): Sentence to synthesize
        model (str): Name of the TTS model to use
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        language (str | None): Language hint, e.g. "Vietnamese" or "English" (default: auto-detect)

    Returns:
        bytes: WAV file contents for this sentence
    """

    # Set up the request payload; each sentence is a normal non-streaming request
    payload = {
        "model": model,             # Specify the model to use
        "input": text,              # Sentence to synthesize
        "response_format": "wav",   # WAV so the sample rate travels in the header
    }
    # Optional language hint; OmniVoice auto-detects when omitted
    if language:
        payload["language"] = language

    response = await client.post(f"{openai_api_base}/audio/speech", json = payload)
    # Raise an informative error when the server rejects the request
    response.raise_for_status()
    return response.content

async def stream_by_sentence(text: str,
                             model: str,
                             openai_api_base: str,
                             output_path: str,
                             language: str | None = None,
                             max_concurrency: int = 2,
                             timeout: float = 300.0):
    """
    Emulate streaming for a diffusion TTS model: synthesize sentences concurrently
    and append each clip to the output WAV in order as soon as it is ready, so the
    first audio is available long before the whole paragraph finishes.

    Args:
        text (str): Paragraph to synthesize
        model (str): Name of the TTS model to use
        openai_api_base (str): Base URL of the OpenAI-compatible API server
        output_path (str): Where to write the combined WAV file
        language (str | None): Language hint passed through to the server (default: None)
        max_concurrency (int): Maximum sentences in flight at once (default: 2)
        timeout (float): Per-request timeout in seconds (default: 300.0)

    Returns:
        None: Prints progress to stdout
    """

    sentences = split_sentences(text)
    print(f"Synthesizing {len(sentences)} sentences (max {max_concurrency} in flight)")

    # Record start time so we can measure time-to-first-audio
    start_time = time.perf_counter()
    first_chunk_time = None
    total_frames = 0
    sample_rate = 0
    wav_out = None

    # Bound how many requests hit the server at once (matches MAX_NUM_SEQS-style limits)
    semaphore = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient(timeout = timeout) as client:
        async def bounded(sentence: str) -> bytes:
            async with semaphore:
                return await synthesize_sentence(client = client,
                                                 text = sentence,
                                                 model = model,
                                                 openai_api_base = openai_api_base,
                                                 language = language)

        # Fire all sentences; the semaphore paces them
        tasks = [asyncio.create_task(bounded(s)) for s in sentences]

        try:
            # Consume results in sentence order so the audio stays in sequence
            for index, task in enumerate(tasks):
                clip = await task
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - start_time
                    print(f"First audio chunk after {first_chunk_time:.2f} seconds")

                # Parse the clip and copy its PCM frames into the combined file
                with wave.open(io.BytesIO(clip), "rb") as wav_in:
                    if wav_out is None:
                        # Lazily open the output using the server's real audio parameters
                        wav_out = wave.open(output_path, "wb")
                        wav_out.setparams(wav_in.getparams())
                        sample_rate = wav_in.getframerate()
                    frames = wav_in.readframes(wav_in.getnframes())
                    wav_out.writeframes(frames)
                    total_frames += wav_in.getnframes()

                print(f"[{index + 1}/{len(sentences)}] {sentences[index]}")
        finally:
            # Close the output so the WAV header gets its final sizes
            if wav_out is not None:
                wav_out.close()

    # Print streaming results and processing metrics
    processing_time = time.perf_counter() - start_time
    duration = total_frames / sample_rate if sample_rate else 0.0
    print(f"=== Sentence Streaming Results ===")
    print(f"Output File: {output_path} ({duration:.2f} s of audio at {sample_rate} Hz)")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print("=" * 35)

async def main():
    """
    Main coroutine to demonstrate sentence-level streaming synthesis.
    """
    # Default vLLM server endpoint (adjust if your server runs on different port/host)
    openai_api_base = "http://localhost:8002/v1"
    # Paragraph to synthesize
    text = ("Xin chào, đây là VoicePlatform. "
            "Đoạn văn này được tách thành từng câu và tổng hợp song song. "
            "Câu đầu tiên sẽ có âm thanh sớm hơn nhiều so với khi chờ toàn bộ đoạn. "
            "Cảm ơn bạn đã lắng nghe!")
    # Model name for speech synthesis
    model_name = "kjanh/KhanhTTS-OmniVoice"
    # Language hint for the model (set to None to auto-detect)
    language = "Vietnamese"
    # Where to write the generated audio
    output_path = os.path.join(RESULTS_DIR, "output_sentence_stream.wav")
    # Make sure the results directory exists before writing into it
    os.makedirs(RESULTS_DIR, exist_ok = True)

    print(f"Using model: {model_name}")

    # Start sentence-level streaming synthesis
    await stream_by_sentence(text = text,
                             model = model_name,
                             openai_api_base = openai_api_base,
                             output_path = output_path,
                             language = language)

# Entry point: Run the main coroutine when script is executed directly
if __name__ == "__main__":
    asyncio.run(main())