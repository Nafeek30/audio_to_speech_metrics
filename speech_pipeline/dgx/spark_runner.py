"""Distributed and batch execution framework for DGX Spark clusters and multi-GPU nodes."""
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def is_spark_available() -> bool:
    """Check if PySpark is installed in the current environment."""
    try:
        import pyspark
        return True
    except ImportError:
        return False


def get_spark_session(app_name: str = "SpeechPipeline_DGX_Batch", master: Optional[str] = None):
    """Create or retrieve an optimized SparkSession for DGX GPU workloads."""
    if not is_spark_available():
        raise RuntimeError("PySpark is not installed. Install with `pip install pyspark` on the DGX Spark node.")
        
    from pyspark.sql import SparkSession
    
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    else:
        # Default local[*] with maximum available threads if no cluster master provided
        builder = builder.master("local[*]")
        
    # Standard PySpark GPU configuration recommendations for DGX
    builder = (
        builder
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.task.maxFailures", "1")
    )
    
    spark = builder.getOrCreate()
    return spark


def run_spark_batch(
    audio_paths: List[Path],
    engine_name: str,
    output_dir: Path,
    model_path_or_name: Optional[str] = None,
    device: str = "auto",
    num_partitions: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Distribute audio transcription and diarization across Spark executors."""
    spark = get_spark_session()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths_str = [str(p.resolve()) for p in audio_paths]
    rdd = spark.sparkContext.parallelize(paths_str, numSlices=num_partitions or max(1, len(paths_str) // 4))
    
    # Process partition on worker
    def process_partition(iterator):
        # Imports and engine initialization occur on the executor inside the partition
        from speech_pipeline.dgx.benchmark_engines import get_engine
        from speech_pipeline.dgx.device_utils import clear_gpu_memory
        
        engine = get_engine(engine_name, model_path_or_name=model_path_or_name, device=device)
        results = []
        for path_str in iterator:
            p = Path(path_str)
            try:
                res = engine.transcribe_and_diarize(p)
                results.append(res)
            except Exception as e:
                results.append({"audio_file": path_str, "error": str(e)})
            finally:
                clear_gpu_memory()
        return iter(results)

    print(f"[*] Dispatching {len(paths_str)} audio jobs across Spark executors...", flush=True)
    collected_results = rdd.mapPartitions(process_partition).collect()
    
    # Save collected batch output
    summary_path = output_dir / f"{engine_name}_spark_batch_results.json"
    summary_path.write_text(json.dumps(collected_results, indent=2), encoding="utf-8")
    print(f"[+] Completed Spark batch job. Results saved to {summary_path}", flush=True)
    return collected_results


def run_local_dgx_batch(
    audio_paths: List[Path],
    engine_name: str,
    output_dir: Path,
    model_path_or_name: Optional[str] = None,
    device: str = "auto"
) -> List[Dict[str, Any]]:
    """Sequential or batch execution on local GPU workstation when Spark is not running."""
    from speech_pipeline.dgx.benchmark_engines import get_engine
    from speech_pipeline.dgx.device_utils import clear_gpu_memory
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    engine = get_engine(engine_name, model_path_or_name=model_path_or_name, device=device)
    results = []
    
    total = len(audio_paths)
    for idx, audio_path in enumerate(audio_paths, 1):
        print(f"[{idx}/{total}] Processing {audio_path.name} with {engine_name}...", flush=True)
        try:
            res = engine.transcribe_and_diarize(audio_path)
            results.append(res)
        except Exception as e:
            print(f"[-] Error processing {audio_path.name}: {e}", flush=True)
            results.append({"audio_file": str(audio_path), "error": str(e)})
        finally:
            clear_gpu_memory()
            
    summary_path = output_dir / f"{engine_name}_batch_results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[+] Saved batch results to {summary_path}", flush=True)
    return results
