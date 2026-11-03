"""
Base classes for unified benchmark infrastructure.

Provides common dataclasses and abstract base class for benchmark runners.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""
    task_id: str
    task_type: str
    method: str
    model: str
    success: bool
    score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    generated_code: str = ""
    execution_output: str = ""
    latency_seconds: float = 0.0
    tokens_used: int = 0
    timestamp: str = ""
    output_cif_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""
    task_type: str
    method: str
    model: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    average_score: float
    average_latency: float
    total_tokens: int
    timestamp: str = ""
    results: List[BenchmarkResult] = field(default_factory=list)
    protocol_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['results'] = [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.results]
        return d

    @classmethod
    def from_results(cls, results: List[BenchmarkResult],
                     task_type: str, method: str, model: str) -> "BenchmarkSummary":
        """Create summary from list of results."""
        if not results:
            return cls(
                task_type=task_type,
                method=method,
                model=model,
                total_tasks=0,
                successful_tasks=0,
                success_rate=0.0,
                average_score=0.0,
                average_latency=0.0,
                total_tokens=0,
                results=[],
            )

        successful = [r for r in results if r.success]
        scores = [r.score for r in results]

        return cls(
            task_type=task_type,
            method=method,
            model=model,
            total_tasks=len(results),
            successful_tasks=len(successful),
            success_rate=len(successful) / len(results),
            average_score=sum(scores) / len(scores),
            average_latency=sum(r.latency_seconds for r in results) / len(results),
            total_tokens=sum(r.tokens_used for r in results),
            results=results,
            protocol_metrics=_summarize_protocol(results),
        )


def _summarize_protocol(results: List[BenchmarkResult]) -> Dict[str, Any]:
    try:
        from src.evaluation.protocol import summarize_protocol_results

        return summarize_protocol_results(results)
    except Exception:
        return {}


class BaseBenchmarkRunner(ABC):
    """Abstract base class for benchmark runners.

    Subclasses should implement:
    - load_tasks(): Load task data
    - run_single_task(): Run a single task and return result
    - get_evaluator(): Return appropriate evaluator for task type
    """

    def __init__(
        self,
        model: str,
        output_dir: Path,
        success_threshold: float = 0.5,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.success_threshold = success_threshold

    @abstractmethod
    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load benchmark tasks."""
        pass

    @abstractmethod
    def run_single_task(self, task: Dict[str, Any], method: str) -> BenchmarkResult:
        """Run a single benchmark task.

        Args:
            task: Task specification dictionary
            method: Method to use ('prompt', 'agent', etc.)

        Returns:
            BenchmarkResult for this task
        """
        pass

    def run_benchmark(
        self,
        method: str,
        task_filter: Optional[str] = None,
        limit: Optional[int] = None,
        rate_limit_seconds: float = 1.0,
    ) -> List[BenchmarkResult]:
        """Run benchmark on all tasks.

        Args:
            method: Method to benchmark ('prompt', 'agent', etc.)
            task_filter: Optional filter by task_type
            limit: Maximum number of tasks to run
            rate_limit_seconds: Delay between API calls

        Returns:
            List of BenchmarkResult
        """
        tasks = self.load_tasks()

        # Apply filters
        if task_filter:
            tasks = [t for t in tasks if t.get('task_type') == task_filter]

        if limit:
            tasks = tasks[:limit]

        logger.info(f"Running {method} benchmark on {len(tasks)} tasks")

        results = []
        for i, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{i}')
            logger.info(f"Progress: {i+1}/{len(tasks)} - {task_id}")

            try:
                result = self.run_single_task(task, method)
                self._attach_protocol_metrics(result, task)
                results.append(result)

                # Save intermediate results
                self._save_results(results, method)

            except Exception as e:
                logger.error(f"Error on task {task_id}: {e}")
                result = BenchmarkResult(
                    task_id=task_id,
                    task_type=task.get('task_type', 'unknown'),
                    method=method,
                    model=self.model,
                    success=False,
                    score=0.0,
                    errors=[str(e)],
                )
                self._attach_protocol_metrics(result, task)
                results.append(result)

            # Rate limiting
            time.sleep(rate_limit_seconds)

        return results

    def _attach_protocol_metrics(self, result: BenchmarkResult, task: Dict[str, Any]) -> None:
        """Attach fixed post-hoc protocol metrics when structure outputs exist."""
        try:
            from src.evaluation.protocol import evaluate_result_protocol

            protocol_mlip_env = os.getenv("BENCHMARK_PROTOCOL_ENABLE_MLIP")
            if protocol_mlip_env is None:
                protocol_enable_mlip = bool(getattr(self, "enable_mlip", False))
            else:
                protocol_enable_mlip = protocol_mlip_env.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }

            result.metadata["evaluation_protocol"] = evaluate_result_protocol(
                result=result,
                task=task,
                output_dir=self.output_dir,
                enable_mlip=protocol_enable_mlip,
                mlip_model=str(getattr(self, "mlip_model_type", "chgnet")),
                mlip_device=str(getattr(self, "mlip_device", "auto")),
                compute_quality=os.getenv("BENCHMARK_PROTOCOL_QUALITY", "1") != "0",
            )
        except Exception as exc:
            result.metadata["evaluation_protocol_error"] = str(exc)

    def compute_summary(self, results: List[BenchmarkResult]) -> Dict[str, BenchmarkSummary]:
        """Compute summary statistics grouped by task type."""
        if not results:
            return {}

        # Group by task type
        by_type: Dict[str, List[BenchmarkResult]] = {}
        for r in results:
            task_type = r.task_type
            if task_type not in by_type:
                by_type[task_type] = []
            by_type[task_type].append(r)

        # Create summaries
        summaries = {}
        for task_type, type_results in by_type.items():
            summaries[task_type] = BenchmarkSummary.from_results(
                type_results, task_type, type_results[0].method, self.model
            )

        # Add overall summary
        summaries['overall'] = BenchmarkSummary.from_results(
            results, 'overall', results[0].method, self.model
        )

        return summaries

    def print_summary(self, summaries: Dict[str, BenchmarkSummary]):
        """Print summary to console."""
        print("\n" + "=" * 70)
        print(f"Benchmark Summary - Model: {self.model}")
        print("=" * 70)

        for name, summary in summaries.items():
            print(f"\n{name.upper()} ({summary.method}):")
            print(f"  Tasks:        {summary.total_tasks}")
            print(f"  Successful:   {summary.successful_tasks}")
            print(f"  Success Rate: {summary.success_rate:.1%}")
            print(f"  Avg Score:    {summary.average_score:.3f}")
            print(f"  Avg Latency:  {summary.average_latency:.2f}s")

        print("\n" + "=" * 70)

    def _save_results(self, results: List[BenchmarkResult], method: str):
        """Save results to JSON file."""
        output_file = self.output_dir / f"{self.model}_{method}_results.json"
        with open(output_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

    def save_summary(self, summaries: Dict[str, BenchmarkSummary], method: str):
        """Save summary to JSON file."""
        summary_file = self.output_dir / f"{self.model}_{method}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in summaries.items()}, f, indent=2)
