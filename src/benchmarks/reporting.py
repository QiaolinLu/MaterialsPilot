"""Attach the fixed post-hoc evaluation protocol to a benchmark result."""
from pathlib import Path
import os

from src.evaluation.protocol import evaluate_result_protocol


def _attach_protocol_metrics(result, task: dict, output_dir: Path,
                             enable_mlip: bool, mlip_model: str,
                             mlip_device: str = None) -> None:
    """Attach fixed post-hoc protocol metrics to a result."""
    try:
        result.metadata["evaluation_protocol"] = evaluate_result_protocol(
            result=result,
            task=task,
            output_dir=output_dir,
            enable_mlip=enable_mlip,
            mlip_model=mlip_model,
            mlip_device=mlip_device or os.getenv("MATERIALSPILOT_MLIP_DEVICE", "auto"),
        )
        protocol = result.metadata["evaluation_protocol"]
        result.success = bool(protocol["structural_valid"] and protocol["task_satisfied"])
        result.score = float(protocol.get("avg_score") or 0.0)
        result.metadata["reported_score"] = "paper_avg_score"
    except Exception as exc:
        result.metadata["evaluation_protocol_error"] = str(exc)
