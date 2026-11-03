"""Four-family benchmark adapter for the paper workflow only."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseBenchmarkRunner, BenchmarkResult
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def save_structure_to_cif(structure: Any, output_dir: Path, task_id: str, method: str, model: str) -> Optional[str]:
    """Save structure to CIF file.

    Args:
        structure: Pymatgen Structure object
        output_dir: Output directory for CIF files
        task_id: Task identifier
        method: Method name (prompt/agent)
        model: Model name

    Returns:
        Relative path to saved CIF file, or None if save failed
    """
    if structure is None:
        return None

    try:
        import re
        from datetime import datetime

        cif_dir = output_dir / "structures"
        cif_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp in filesystem-friendly format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize model name for filename (replace all non-alphanumeric chars with -)
        model_safe = re.sub(r'[^\w\-]', '-', model)

        cif_filename = f"{task_id}_{method}_{model_safe}_{timestamp}.cif"
        cif_path = cif_dir / cif_filename

        structure.to(filename=str(cif_path), fmt="cif")

        # Return relative path from output_dir
        return f"structures/{cif_filename}"
    except Exception as e:
        logger.warning(f"Failed to save CIF for {task_id}: {e}")
        return None


class UnifiedBenchmarkRunner(BaseBenchmarkRunner):
    """Load the four public datasets and run the paper execution-feedback loop."""

    TASK_CONFIGS = {
        "defect": {
            "data_path": "tasks/jarvis_defect/data/complex_tasks_v3.json",
            "evaluator_class": "DefectEvaluator",
        },
        "adsorption": {
            "data_path": "tasks/ocp_adsorption/data/adsorption_tasks.json",
            "evaluator_class": "AdsorptionEvaluator",
        },
        "grain_boundary": {
            "data_path": "tasks/grain_boundary/data/grain_boundary_tasks.json",
            "evaluator_class": "GrainBoundaryEvaluator",
        },
        "interface": {
            "data_path": "tasks/heterostructure/data/interface_tasks_expanded.json",
            "evaluator_class": "InterfaceEvaluator",
        },
    }

    def __init__(
        self, task_type: str, model: str = "deepseek-v3.2",
        output_dir: Optional[Path] = None, project_root: Optional[Path] = None,
        success_threshold: float = 0.5, enable_mlip: bool = False,
        mlip_model_type: str = "chgnet", mlip_device: Optional[str] = None,
        use_advanced_rag: bool = False, use_unified_validation: bool = False,
        use_multi_agent: bool = False, max_agent_iterations: int = 3,
        use_methods_feedback: bool = True, methods_feedback_transformer: str = "rule",
        methods_feedback_context_expansion: bool = True,
        methods_enable_reflection: Optional[bool] = None,
    ):
        if task_type not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task type: {task_type}. Available: {list(self.TASK_CONFIGS)}")
        if use_advanced_rag or use_unified_validation or use_multi_agent or not use_methods_feedback:
            raise ValueError("This package supports only the paper agent_methods workflow")
        if methods_feedback_transformer != "rule":
            raise ValueError("The paper workflow uses deterministic rule feedback")
        mlip_device = mlip_device if mlip_device is not None else os.getenv("MATERIALSPILOT_MLIP_DEVICE", "auto")
        if mlip_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("mlip_device must be auto, cuda, or cpu")
        self.task_type = task_type
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        self.enable_mlip = enable_mlip
        self.mlip_model_type = mlip_model_type
        self.mlip_device = mlip_device
        self.max_agent_iterations = max_agent_iterations
        self.use_methods_feedback = True
        self.methods_feedback_transformer = "rule"
        self.methods_feedback_context_expansion = methods_feedback_context_expansion
        self.methods_enable_reflection = False
        if output_dir is None:
            output_dir = self.project_root / f"benchmarks/results/{task_type}"
        super().__init__(model, output_dir, success_threshold)
        self.llm_client = LLMClient(model_key=model)
        self.execution_feedback_orchestrator = None
        self._ensure_execution_feedback_orchestrator()

    def _get_methods_rag_system(self):
        """Get RAG backend for the Methods-inspired feedback orchestrator."""
        try:
            from src.rag import UnifiedRAG
            return UnifiedRAG(task_type=self.task_type)
        except Exception as e:
            logger.warning(f"Failed to initialize UnifiedRAG for Methods orchestrator: {e}")
            return None

    def _ensure_execution_feedback_orchestrator(self):
        """Lazily initialize the Methods-inspired execution-feedback orchestrator."""
        if self.execution_feedback_orchestrator is not None:
            return self.execution_feedback_orchestrator

        try:
            from src.agents.paper_aligned_orchestrator import PaperAlignedOrchestrator

            self.execution_feedback_orchestrator = PaperAlignedOrchestrator(
                llm_client=self.llm_client,
                rag_system=self._get_methods_rag_system(),
                max_iterations=self.max_agent_iterations,
                enable_refinement=True,
                enable_reflection=self.methods_enable_reflection,
                enable_mlip=self.enable_mlip,
                mlip_model=self.mlip_model_type,
                mlip_device=self.mlip_device,
                feedback_transformer=self.methods_feedback_transformer,
                feedback_context_expansion=self.methods_feedback_context_expansion,
                project_root=self.project_root,
            )
            logger.info(
                "Initialized Methods-inspired ExecutionFeedbackOrchestrator "
                f"(max_iterations={self.max_agent_iterations}, "
                f"feedback_transformer={self.methods_feedback_transformer})"
            )
            return self.execution_feedback_orchestrator
        except Exception as e:
            logger.error(f"Failed to initialize Methods-inspired orchestrator: {e}")
            self.execution_feedback_orchestrator = None
            self.use_methods_feedback = False
            return None

    def _methods_task_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured task input for the Methods feedback loop.

        Some benchmark tasks store core fields at the top level (adsorption),
        while others nest them under ``input`` and keep expected constraints in
        sibling fields. The feedback loop needs both so its deterministic
        checks and early-stop logic see the same constraints as the final
        protocol.
        """
        nested_input = task.get("input")
        if isinstance(nested_input, dict):
            structured: Dict[str, Any] = dict(nested_input)
        else:
            structured = {
                key: value
                for key, value in task.items()
                if key not in {"_metadata", "requires_rag"}
            }

        for key in (
            "task_id",
            "task_type",
            "category",
            "difficulty",
            "description",
            "expected_properties",
            "validation",
            "constraints",
            "metal",
            "facet",
            "adsorbate",
            "site",
            "surface_coords",
            "coverage",
            "adsorbates",
            "preferred_sites",
        ):
            if key in task and key not in structured:
                structured[key] = task[key]

        if isinstance(nested_input, dict):
            structured["_raw_input"] = dict(nested_input)
        structured["_full_task"] = {
            key: value
            for key, value in task.items()
            if key not in {"_metadata", "requires_rag"}
        }
        return structured

    @staticmethod
    def _is_retryable_generation_failure(message: str) -> bool:
        markers = (
            "expecting value",
            "jsondecodeerror",
            "502 bad gateway",
            "503",
            "temporarily unavailable",
            "timeout",
            "too many requests",
            "rate limit",
            "empty message content",
            "finish_reason=length",
        )
        lower = str(message).lower()
        return any(marker in lower for marker in markers)

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from configured data path.

        Handles both formats:
        - Direct array: [task1, task2, ...]
        - Nested object: {"tasks": [task1, task2, ...], "_metadata": {...}}
        """
        config = self.TASK_CONFIGS[self.task_type]
        data_path = self.project_root / config["data_path"]

        if not data_path.exists():
            raise FileNotFoundError(f"Task data not found: {data_path}")

        with open(data_path) as f:
            data = json.load(f)

        # Handle both formats
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
        else:
            raise ValueError(f"Unknown task data format in {data_path}")

        # Validate that all tasks are dictionaries
        validated_tasks = []
        for i, task in enumerate(tasks):
            if isinstance(task, dict):
                validated_tasks.append(task)
            else:
                logger.warning(f"Skipping invalid task at index {i}: expected dict, got {type(task).__name__}")

        if not validated_tasks:
            raise ValueError(f"No valid tasks found in {data_path}")

        return validated_tasks

    def run_single_task(self, task: Dict[str, Any], method: str = "agent_methods") -> BenchmarkResult:
        if method not in {"agent_methods", "methods_agent", "execution_feedback"}:
            raise ValueError("This package supports only the paper agent_methods workflow")
        return self._run_methods_agent(task, time.time(), method=method)

    def _run_methods_agent(
        self,
        task: Dict[str, Any],
        start_time: float,
        method: str = "agent_methods",
    ) -> BenchmarkResult:
        """Run the Methods-inspired execution-feedback agent."""
        task_id = task.get("task_id", "unknown")
        prompt = self._get_task_prompt(task)

        orchestrator = self._ensure_execution_feedback_orchestrator()
        if orchestrator is None:
            return BenchmarkResult(
                task_id=task_id,
                task_type=self.task_type,
                method=method,
                model=self.model,
                success=False,
                score=0.0,
                errors=["Methods-inspired execution-feedback orchestrator is unavailable"],
                latency_seconds=time.time() - start_time,
            )

        logger.info(f"Using Methods-inspired execution-feedback agent for task {task_id}")

        try:
            result = orchestrator.run(
                task_description=prompt,
                task_input=self._methods_task_input(task),
            )

            code = result.final_code
            structure = result.structure
            success = result.success
            tokens = result.total_tokens

            if success and structure is not None:
                # Preserve the candidate independently of the legacy evaluator.
                # The fixed paper protocol is attached after generation, once.
                final_state = (result.evaluations[-1].get("critic_state", {})
                               if result.evaluations else {})
                score = float(final_state.get("overall_score", 0.0))
                errors = []
                warnings = [item.get("description", "") for item in final_state.get("issues", [])]
            else:
                score = 0.0
                errors = result.metadata.get('execution_result', {}).get('output', 'No structure generated')
                if isinstance(errors, str):
                    errors = [errors]
                warnings = []

            cif_path = save_structure_to_cif(structure, self.output_dir, task_id, method, self.model)
            if structure is not None and not cif_path:
                raise OSError("Generated structure could not be persisted; this is a local infrastructure failure")

            metadata = {
                'method_profile': result.metadata.get('method_profile'),
                'retrieval_log': result.metadata.get('retrieval_log', []),
                'automatic_structure_repair': result.metadata.get('automatic_structure_repair', False),
                'full_history_conditioning': result.metadata.get('full_history_conditioning', False),
                'methods_inspired': False,
                'paper_aligned': True,
                'multi_agent': False,
                'iterations': result.iterations,
                'evaluations': result.evaluations,
                'state_history': result.metadata.get('state_history', []),
                'active_context_count': result.metadata.get('active_context_count', 0),
                'feedback_transformer': result.metadata.get('feedback_transformer'),
                'accepted_by_methods_critic': result.metadata.get('accepted_by_methods_critic', False),
                'acceptance_decision': result.metadata.get('acceptance_decision', {}),
                'selected_best_candidate': result.metadata.get('selected_best_candidate', False),
                'selected_from_best_so_far': result.metadata.get('selected_from_best_so_far', False),
                'selected_iteration': result.metadata.get('selected_iteration'),
                'selected_candidate_score': result.metadata.get('selected_candidate_score'),
                'selected_acceptance_decision': result.metadata.get('selected_acceptance_decision'),
                'best_candidate': result.metadata.get('best_candidate'),
            }

            return BenchmarkResult(
                task_id=task_id,
                task_type=self.task_type,
                method=method,
                model=self.model,
                success=success,
                score=score,
                errors=errors,
                warnings=warnings,
                generated_code=code,
                execution_output=result.metadata.get('execution_result', {}).get('output', ''),
                latency_seconds=time.time() - start_time,
                tokens_used=tokens,
                output_cif_path=cif_path,
                metadata=self._json_safe(metadata),
            )

        except Exception as e:
            logger.error(f"Methods-inspired execution-feedback agent failed: {e}")
            import traceback
            traceback.print_exc()
            return BenchmarkResult(
                task_id=task_id,
                task_type=self.task_type,
                method=method,
                model=self.model,
                success=False,
                score=0.0,
                errors=[str(e)],
                latency_seconds=time.time() - start_time,
                metadata={
                    "retryable_failure": self._is_retryable_generation_failure(str(e)),
                },
            )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert metadata to JSON-serializable values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): UnifiedBenchmarkRunner._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [UnifiedBenchmarkRunner._json_safe(v) for v in value]

        try:
            import numpy as np
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        return str(value)

    def _get_task_prompt(self, task: Dict[str, Any]) -> str:
        """Get prompt for task based on task type."""
        # Import task-specific prompt generators
        from src.prompts.benchmark_prompts import (
            get_defect_task_prompt,
            get_adsorption_task_prompt,
            get_grain_boundary_task_prompt,
            get_interface_task_prompt,
        )

        prompt_generators = {
            "defect": get_defect_task_prompt,
            "defect_complex": get_defect_task_prompt,
            "adsorption": get_adsorption_task_prompt,
            "grain_boundary": get_grain_boundary_task_prompt,
            "interface": get_interface_task_prompt,
        }

        generator = prompt_generators.get(self.task_type)
        if generator:
            return generator(task)

        # Fallback to description or raw task
        return task.get("description", task.get("prompt", str(task)))

