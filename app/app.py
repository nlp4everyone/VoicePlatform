# Dependencies
import logging

# Suppress NeMo and its dependencies before any NeMo import occurs.
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for _logger in ("nemo", "nemo_logger", "lightning", "pytorch_lightning",
                "filelock", "datasets", "huggingface_hub"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

from .services.deployments.asr_deployment import ASRService

# Importing this module must stay free of side effects: Ray and the HTTP proxy
# are started by `serve run config/serve.yaml`, which also owns the replica
# count, resources and host/port (see config/serve.yaml).
logging.getLogger("ray.serve").setLevel(logging.INFO)

# Bind the ASR service deployment
deployment = ASRService.bind()
