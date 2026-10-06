from app.utils.config_loader import get_toml_config

# Load interaction settings from TOML
toml_config = get_toml_config()

serving_config = toml_config.get_section("serving")

# Model config
MAX_BATCH_SIZE = serving_config.get("MAX_BATCH_SIZE", 8)
BATCH_WAIT_TIMEOUT_S = serving_config.get("BATCH_WAIT_TIMEOUT_S", 0.1)
# Threads per replica that decode uploaded audio. Keep config/serve.yaml
# ray_actor_options.num_cpus >= this value
DECODE_WORKERS = serving_config.get("DECODE_WORKERS", 4)