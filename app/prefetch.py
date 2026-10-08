"""Download the configured ASR model into the HuggingFace cache and exit.

Run `make prefetch` once per machine. The cache is the host volume mounted at
HF_HOME, so later starts find the model locally and can run with
HF_HUB_OFFLINE=1.
"""
import sys

from huggingface_hub import snapshot_download

from app.core.config.settings import settings


def main() -> int:
    print(f"Downloading '{settings.ASR_MODEL_NAME}' into the HuggingFace cache...")
    path = snapshot_download(repo_id=settings.ASR_MODEL_NAME)
    print(f"Model cached at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
