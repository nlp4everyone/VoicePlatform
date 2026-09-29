FROM vllm/vllm-openai:v0.24.0

# Audio support (soundfile / librosa) is not bundled in the base image: install libsndfile
# for soundfile and ffmpeg so mp3 and other compressed formats can be decoded
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Pin the version so pip keeps the vLLM build of the base image
RUN pip install --no-cache-dir "vllm[audio]==0.24.0"
