import pytest

from app.services.asr import RecognizerFactory
from app.utils.language_detect import LanguageDetector
from app.utils.token_counter.token_counter import approximate_count_tokens


def test_factory_rejects_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported ASR model"):
        RecognizerFactory.create(model_name="not/a-model", device="cpu")


def test_token_count_string_and_messages():
    assert approximate_count_tokens("hello world") > 0
    msgs = [{"role": "user", "content": "hello world"}]
    assert approximate_count_tokens(msgs) == approximate_count_tokens("hello world") + 4 + 2
    assert approximate_count_tokens({"content": "hello world"}) == approximate_count_tokens(msgs)


def test_language_detector_vietnamese():
    props = LanguageDetector.detect("Xin chào, hôm nay trời đẹp quá, chúng ta đi dạo nhé")
    assert props.lang_code == "vi"


def test_language_detector_english():
    props = LanguageDetector.detect("The quick brown fox jumps over the lazy dog")
    assert props.lang_code == "en"
