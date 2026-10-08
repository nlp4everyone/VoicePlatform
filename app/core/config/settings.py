from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import (BaseSettings, PydanticBaseSettingsSource,
                               SettingsConfigDict, TomlConfigSettingsSource)

CONFIG_FILE = Path(__file__).resolve().parents[3] / "config" / "config.toml"


class Settings(BaseSettings):
    """Runtime settings. Precedence: ASR_* env var > config.toml > default."""

    model_config = SettingsConfigDict(env_prefix="ASR_",
                                      toml_file=CONFIG_FILE,
                                      env_ignore_empty=True,
                                      extra="ignore")

    # Serving
    MAX_BATCH_SIZE: int = Field(8, gt=0)
    BATCH_WAIT_TIMEOUT_S: float = Field(0.1, ge=0)
    # Threads per replica that decode uploaded audio. Keep config/serve.yaml
    # ray_actor_options.num_cpus >= this value
    DECODE_WORKERS: int = Field(4, gt=0)

    # Model
    # These two already start with ASR_: the alias skips the prefix, so the env
    # var is ASR_MODEL_NAME rather than ASR_MODEL_NAME
    ASR_MODEL_NAME: str = Field("nvidia/parakeet-ctc-0.6b-vi",
                                validation_alias="ASR_MODEL_NAME")
    ASR_DEVICE: Literal["auto", "cuda", "cpu"] = Field("auto",
                                                       validation_alias="ASR_DEVICE")
    SPLIT_MIXED_BATCH: bool = True

    @classmethod
    def settings_customise_sources(cls, settings_cls,
                                   init_settings: PydanticBaseSettingsSource,
                                   env_settings: PydanticBaseSettingsSource,
                                   dotenv_settings: PydanticBaseSettingsSource,
                                   file_secret_settings: PydanticBaseSettingsSource):
        return (init_settings, env_settings,
                TomlConfigSettingsSource(settings_cls))


settings = Settings()
