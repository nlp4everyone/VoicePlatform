from typing import Any, List, Union
from app.schema.transcription.response import TranscriptionResult

class BaseRecognizer:
    """
    Base class for ASR (Automatic Speech Recognition) model recognizers.

    This abstract class defines the interface that all ASR model implementations
    must follow. It provides a common structure for loading models and
    transcribing audio files.

    """

    def __init__(self,
                 model_name :str):
        """
        Initialize the BaseRecognizer with a model name.

        Args:
            model_name (str): The name/identifier of the ASR model to use
        """
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """
        Get the name of the loaded model.

        Returns:
            str: The model name
        """
        return self._model_name

    def check_health(self) -> None:
        """
        Verify the recognizer can still run. Raise if it cannot.

        Called periodically by the serving layer; a raised exception marks the
        replica unhealthy so it gets restarted. The default has nothing to check.
        """

    def warmup(self, audio: Any, batch_sizes: List[int]) -> None:
        """
        Run sample inferences so the first real request does not pay one-time
        costs (CUDA kernel/cuDNN init, allocator growth). The default has
        nothing to warm up.

        Args:
            audio: Sample waveform (mono, 16 kHz tensor) to transcribe.
            batch_sizes: Batch sizes to exercise.
        """

    def transcribe(self,
                    audio :Union[str,bytes]) -> TranscriptionResult:
        """
        Transcribe audio input to text.

        This abstract method must be implemented by subclasses to provide
        the actual transcription logic for their specific ASR model.

        Args:
            audio (Union[str, bytes]): Audio input as either a file path (str)
                                       or audio data as bytes

        Returns:
            TranscriptionResult: The transcription result containing text and
                                optional timestamp information

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError()