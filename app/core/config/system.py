from app.utils.config_loader import get_toml_config

# Load interaction settings from TOML
toml_config = get_toml_config()

system_config = toml_config.get_section("system")

# Model config
AUDIO_TEMP_DIR = system_config.get("AUDIO_TEMP_DIR", "/dev/shm")
