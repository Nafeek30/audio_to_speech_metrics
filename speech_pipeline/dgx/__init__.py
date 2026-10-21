"""DGX GPU-accelerated and Spark-distributed audio transcription, diarization, and fine-tuning."""

from speech_pipeline.dgx.device_utils import (
    clear_gpu_memory,
    get_platform_info,
    optimize_cuda_env,
    resolve_device,
    resolve_dtype,
)
from speech_pipeline.dgx.benchmark_engines import (
    BaseAudioEngine,
    MossDiarizeEngine,
    SenseVoiceEngine,
    WhisperEngine,
    get_engine,
)
from speech_pipeline.dgx.evaluation import (
    compute_boundary_f1,
    compute_cpwer,
    compute_wer,
    evaluate_3tier,
)
from speech_pipeline.dgx.spark_runner import (
    get_spark_session,
    is_spark_available,
    run_local_dgx_batch,
    run_spark_batch,
)

__all__ = [
    "clear_gpu_memory",
    "get_platform_info",
    "optimize_cuda_env",
    "resolve_device",
    "resolve_dtype",
    "BaseAudioEngine",
    "WhisperEngine",
    "MossDiarizeEngine",
    "SenseVoiceEngine",
    "get_engine",
    "compute_wer",
    "compute_cpwer",
    "compute_boundary_f1",
    "evaluate_3tier",
    "is_spark_available",
    "get_spark_session",
    "run_spark_batch",
    "run_local_dgx_batch",
]
