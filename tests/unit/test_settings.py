import pytest

from app.core.config.settings import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os
    for key in list(os.environ):
        if key.startswith("ASR_"):
            monkeypatch.delenv(key)


def test_defaults_come_from_config_toml():
    s = Settings()
    assert s.MAX_BATCH_SIZE > 0
    assert s.ASR_MODEL_NAME.startswith("nvidia/")
    assert s.WARMUP is True


def test_env_overrides_toml(monkeypatch):
    monkeypatch.setenv("ASR_MAX_BATCH_SIZE", "3")
    monkeypatch.setenv("ASR_DEVICE", "cpu")
    monkeypatch.setenv("ASR_WARMUP", "false")
    s = Settings()
    assert s.MAX_BATCH_SIZE == 3
    assert s.ASR_DEVICE == "cpu"
    assert s.WARMUP is False


def test_model_name_env_has_single_prefix(monkeypatch):
    monkeypatch.setenv("ASR_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
    assert Settings().ASR_MODEL_NAME == "nvidia/parakeet-tdt-0.6b-v3"


def test_empty_env_is_ignored(monkeypatch):
    monkeypatch.setenv("ASR_MAX_BATCH_SIZE", "")
    assert Settings().MAX_BATCH_SIZE == Settings().MAX_BATCH_SIZE > 0


def test_invalid_value_is_rejected(monkeypatch):
    monkeypatch.setenv("ASR_MAX_BATCH_SIZE", "0")
    with pytest.raises(Exception):
        Settings()
    monkeypatch.setenv("ASR_MAX_BATCH_SIZE", "4")
    monkeypatch.setenv("ASR_DEVICE", "tpu")
    with pytest.raises(Exception):
        Settings()
