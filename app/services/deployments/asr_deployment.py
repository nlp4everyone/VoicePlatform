# FastAPI components
from fastapi import UploadFile, File, Form, FastAPI
# Ray Serve for deployment
from ray import serve
# Type hints
from typing import Union
# ASR Model
from app.services.asr import RecognizerFactory
# Configuration imports
from app.core.config.system import *
from app.core.config.serving import *
from app.core.config.asr import *
# Utils
from app.utils.audio import load_audio_from_bytes
from app.utils.transcription.helper import (get_transcription_type,
                                            process_batch_transcription)
from app.utils.token_counter import approximate_count_tokens
from app.utils.language_detect import LanguageDetector
from app.utils.audio import is_audio_file
# Schema
from app.schema.transcription.response import *
from app.schema.transcription.base import AdvancedTranscribedSegment
from app.schema.transcription.type import TranscriptionType
from app.schema.transcription.base.usage import *
# Custom exceptions
from app.exceptions.transcription import TranscriptedModelNotFoundException
from app.exceptions.audio import (UnsupportedAudioFormatException,
                                  InvalidAudioException)
from app.exceptions.handlers import common_exception_handler
# Other utils
import logging, math, asyncio, functools, concurrent.futures, torch
from pathlib import Path

# Suppress NeMo/Lightning verbose output in actor process (app.py runs in the
# driver, not here — so those suppressions don't carry over).
for _nemo_logger in ("nemo", "nemo_logger", "lightning", "pytorch_lightning",
                     "filelock", "datasets", "huggingface_hub"):
    logging.getLogger(_nemo_logger).setLevel(logging.ERROR)

# Explicitly set to INFO in the actor process — the driver's logging setup does
# not carry over here, and Ray worker setup may still override the level. Setting it here ensures INFO logs are visible in actors.
logging.getLogger("ray.serve").setLevel(logging.INFO)
logger = logging.getLogger("ray.serve")

# Define tags metadata for API documentation
tags_metadata = [
    {
        "name": "Audio",
        "description": "Endpoints for interacting with ASR models"
    },
]

# Initialize FastAPI application for ASR service
asr_app = FastAPI(openapi_tags=tags_metadata)

# Register custom exception handler for ASR model not found errors
asr_app.add_exception_handler(TranscriptedModelNotFoundException, common_exception_handler)
asr_app.add_exception_handler(UnsupportedAudioFormatException, common_exception_handler)
asr_app.add_exception_handler(InvalidAudioException, common_exception_handler)

# Replica count, GPUs, CPUs and max_ongoing_requests are set in
# config/serve.yaml, not here.
@serve.deployment
@serve.ingress(asr_app)
class ASRService:
    """
    Ray Serve deployment for ASR (Automatic Speech Recognition) service.

    This class handles audio transcription requests using configurable ASR models.
    It supports batch processing for improved throughput and provides different
    transcription formats based on client requirements.

    The deployment is configured with:
    - GPU, CPU, replica count and request limit set in config/serve.yaml
    - One CPU per audio decode thread (DECODE_WORKERS)
    """

    def __init__(self):
        """
        Initialize the ASR service.

        Sets up the ASR model.
        Raises ValueError for an unsupported model and RuntimeError if loading fails.
        """
        # Ray Serve calls configure_component_logger() after module import,
        # which may reset ray.serve to WARNING. Re-apply INFO here so factory
        # logs are visible before model loading begins.
        logging.getLogger("ray.serve").setLevel(logging.INFO)

        # Initialize ASR model using factory pattern with configuration
        self._asr_model = RecognizerFactory.create(model_name=ASR_MODEL_NAME,
                                                   device=ASR_DEVICE)

        logger.info(
            f"ASRService ready | max_batch_size={MAX_BATCH_SIZE} "
            f"decode_workers={DECODE_WORKERS}"
        )

        # Dedicated, bounded pool for audio decoding: unlike asyncio's default
        # executor (sized from the host CPU count and shared process-wide), its
        # size is explicit and caps the CPU and memory spent decoding at once.
        self._decode_executor = concurrent.futures.ThreadPoolExecutor(max_workers=DECODE_WORKERS,
                                                                      thread_name_prefix="audio-decode")

        # Single-thread executor keeps GPU work pinned to one thread, avoiding
        # CUDA context migration and pool contention from asyncio's default executor.
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @serve.batch(max_batch_size=MAX_BATCH_SIZE,
                 batch_wait_timeout_s=BATCH_WAIT_TIMEOUT_S)
    async def batched_transcribe(self,
                                 audio_tensors: List[torch.Tensor],
                                 timestamp_granularities: List[Union[str, None]]):
        """
        Batched transcription endpoint with Ray Serve automatic batching.

        Automatically batches individual requests for improved throughput.
        Audio is decoded per request before it gets here, so a broken upload
        never fails the other requests in the batch.

        Args:
            audio_tensors: List of decoded mono 16 kHz waveforms
            timestamp_granularities: Timestamp requirements for each audio file

        Returns:
            List of transcription results
        """
        transcriptions = await asyncio.get_event_loop().run_in_executor(
            self._gpu_executor,
            functools.partial(
                process_batch_transcription,
                asr_model=self._asr_model,
                audio_data=audio_tensors,
                timestamp_granularities=timestamp_granularities,
                split_mixed_batch=SPLIT_MIXED_BATCH
            )
        )
        return transcriptions

    @asr_app.post("/v1/audio/transcriptions",
                  name="Transcribe audio files with optional timestamps",
                  tags=["Audio"])
    async def transcribe_audio(self,
                               file: UploadFile = File(...),
                               model: str = Form(ASR_MODEL_NAME),
                               timestamp_granularity: Optional[Literal["word", "segment"]] = Form(
                                   default=None,
                                   alias="timestamp_granularities[]",
                                   description="Level of timestamp detail: 'word' for word-level timestamps, 'segment' for segment-level timestamps, or None for no timestamps"),
                               response_format: str = Form("verbose_json")):
        """
        ## Transcribe audio files with optional timestamps.

        ### Args:
        - `file`: Audio file to transcribe
        - `model`: ASR model name (must match configured model)
        - `timestamp_granularity`: Level of timestamp detail (word/segment)
        - `response_format`: Output format (currently verbose_json)

        ### Returns:
        - Transcription response in requested format

        ### Raises:
        - `TranscriptedModelNotFoundException`: If requested model is not available
        - `UnsupportedAudioFormatException`: If the file is not an audio format
        - `InvalidAudioException`: If the audio cannot be decoded or has no samples
        """
        logger.debug(f"ASR request received - model: {model}")

        # Validate that requested model matches the loaded model
        # This ensures the client requests a model that is actually available
        if model != self._asr_model.model_name:
            raise TranscriptedModelNotFoundException(model=model)

        # Read uploaded audio file into memory for validation and processing
        audio_bytes = await file.read()

        # Validate that the uploaded file is a valid audio format
        if not is_audio_file(audio_bytes):
            # Extract file extension from filename without the dot (e.g., "mp3", "wav")
            file_extension = Path(file.filename).suffix.lstrip('.') if file.filename else ""
            raise UnsupportedAudioFormatException(file_format=file_extension)

        # Decode here rather than inside the batch: a broken file is rejected with
        # 400 for this request only. Decoding is CPU-bound, so it runs on the
        # dedicated decode pool to keep the event loop free.
        waveform, duration = await asyncio.get_running_loop().run_in_executor(
            self._decode_executor,
            load_audio_from_bytes,
            audio_bytes
        )
        # Release raw bytes immediately — the tensor is all we need from here on.
        audio_bytes = None

        # Call the batch method directly: Ray Serve still batches concurrent calls
        # within this replica. Going through a deployment handle would make every
        # request hold two max_ongoing_requests slots (and can deadlock once the
        # outer requests fill them all), and would serialize the tensor.
        transcription_result: TranscriptionResult = await self.batched_transcribe(
            waveform,
            timestamp_granularity
        )
        transcription_result.duration = duration

        # Determine response format based on granularity
        output_type = get_transcription_type(timestamp_granularity)

        # Return appropriate response format
        if output_type == TranscriptionType.Text:
            # Simple text transcription with token usage
            output_tokens = approximate_count_tokens(transcription_result.text)
            # Return transcription response
            return TranscriptionResponse(
                text=transcription_result.text,
                usage=Usage(
                    input_tokens=0,
                    input_token_details=InputTokenDetails(
                        text_tokens=0,
                        audio_tokens=0
                    ),
                    output_tokens=output_tokens,
                    total_tokens=output_tokens
                )
            )

        elif output_type == TranscriptionType.Word:
            # Word-level transcription with timestamps
            lang_property = LanguageDetector.detect(transcription_result.text, True)
            duration = transcription_result.duration

            # Return transcription response
            return WordResponse(
                text=transcription_result.text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                words=transcription_result.words
            )

        elif output_type == TranscriptionType.Segment:
            # Segment-level transcription with timestamps
            lang_property = LanguageDetector.detect(transcription_result.text, True)
            duration = transcription_result.duration

            # Build segment list with IDs
            segments = []
            for index, segment in enumerate(transcription_result.segments):
                segments.append(AdvancedTranscribedSegment(
                    id=index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text
                ))

            # Return transcription response
            return SegmentResponse(
                text=transcription_result.text,
                language=lang_property.language,
                duration=duration,
                usage=DurationUsage(seconds=math.ceil(duration)),
                segments=segments
            )
