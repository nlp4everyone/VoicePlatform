FROM vllm/vllm-openai:v0.24.0

# Bake the audio extras (av, scipy, soundfile, soxr) into the image so they are
# installed once at build time instead of on every container start.
RUN pip install --no-cache-dir "vllm[audio]==0.24.0"
