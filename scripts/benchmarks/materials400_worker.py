#!/usr/bin/env python3
"""Durable single-family worker for the frozen four-family benchmark.

Exit codes: 0 complete, 20 retryable API failure, 21 fatal API/model failure,
22 local configuration, data, or scoring failure. Completed model failures are
results, not retries. This wrapper does not change generation or evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelIdentityError(RuntimeError):
    pass


class ResponsePolicyError(RuntimeError):
    """A completed upstream response violated this experiment's request policy."""

    def __init__(self, violations):
        self.violations = violations
        super().__init__("Upstream response policy violation: " + json.dumps(violations, sort_keys=True))


class APIRequestDeadlineError(RuntimeError):
    """The wall-clock request deadline expired, independent of read heartbeats."""


@contextmanager
def api_request_deadline(seconds: float):
    """Bound one SDK call on the worker's main thread and restore signal state.

    An existing earlier alarm is rejected instead of silently extending another
    component's deadline. A later alarm is restored with elapsed time deducted.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("API request deadline must be a positive finite number")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("API request deadline requires the worker main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0 and previous_timer[0] <= seconds:
        raise RuntimeError("API request deadline cannot mask an earlier SIGALRM timer")
    started = time.monotonic()
    expired = False

    def expire(signum, frame):
        nonlocal expired
        expired = True
        raise APIRequestDeadlineError(f"SDK request exceeded total wall-clock deadline of {seconds:g} seconds")

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        except Exception as exc:
            # OpenAI's synchronous SDK wraps exceptions raised during send()
            # as APIConnectionError. Preserve the real deadline classification.
            if expired and not isinstance(exc, APIRequestDeadlineError):
                raise APIRequestDeadlineError(
                    f"SDK request exceeded total wall-clock deadline of {seconds:g} seconds"
                ) from exc
            raise
        if expired:
            raise APIRequestDeadlineError(f"SDK request exceeded total wall-clock deadline of {seconds:g} seconds")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = max(1e-6, previous_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_results(path: Path) -> list[dict]:
    """Fail closed on damaged journals instead of regenerating completed tasks."""
    if not path.exists():
        return []
    records = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id in seen:
                raise ValueError(f"Duplicate completed task in journal: {task_id}")
            seen.add(task_id)
            records.append(record)
    return records


def selected_tasks(runtime: Path, manifest: dict, task_type: str) -> list[dict]:
    families = [family for family in manifest["families"] if family["task_type"] == task_type]
    if len(families) != 1:
        raise ValueError(f"Expected one manifest family for {task_type}")
    family = families[0]
    source = (runtime / family["data_path"]).resolve()
    if not source.is_relative_to(runtime.resolve()):
        raise ValueError("Manifest data_path must be inside frozen runtime")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != family["data_sha256"]:
        raise ValueError(f"Frozen dataset hash mismatch: {task_type}")
    data = json.loads(raw)
    data = data if isinstance(data, list) else data["tasks"]
    tasks = []
    seen = set()
    for sample in family["samples"]:
        task = data[sample["index"]]
        digest = hashlib.sha256(json.dumps(
            task, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if task.get("task_id") != sample["task_id"] or digest != sample["task_sha256"]:
            raise ValueError(f"Frozen task identity/hash mismatch: {sample['task_id']}")
        if task["task_id"] in seen:
            raise ValueError(f"Duplicate manifest task: {task['task_id']}")
        seen.add(task["task_id"])
        tasks.append(task)
    if len(tasks) != family["count"]:
        raise ValueError("Manifest family count mismatch")
    return tasks


def fatal_api_error(exc: Exception) -> bool:
    return isinstance(exc, ModelIdentityError) or getattr(exc, "status_code", None) in {401, 402, 403}


def tracked_client_class(base_class, api_error_class, usage_path: Path, expected_model: str):
    """Build the subclass only after the frozen runtime owns the import path."""
    class TrackedClient(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.request_timeout_seconds = float(os.getenv("MATERIALSPILOT_API_TIMEOUT_SECONDS", "180"))
            if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
                raise ValueError("API timeout must be a positive finite number")
            self.provider_options = json.loads(os.getenv(
                "MATERIALSPILOT_PROVIDER_OPTIONS_JSON",
                '{"require_parameters":true,"ignore":["GMICloud"]}',
            ))
            if not isinstance(self.provider_options, dict):
                raise ValueError("Provider options must be a JSON object")
            if any(key in self.provider_options for key in ("order", "only", "sort")):
                raise ValueError("Default provider routing must not specify order, only, or sort")
            # Keep OpenRouter's selection within the frozen support filter;
            # the durable coordinator owns infrastructure retries.
            self.client = self.client.with_options(
                timeout=self.request_timeout_seconds,
                max_retries=0,
            )
            self.last_api_error = None
            self.task_api_errors = []
            self.current_task_id = None
            self.task_usage = []

        def begin_task(self, task_id):
            self.last_api_error = None
            self.task_api_errors = []
            self.current_task_id = task_id
            self.task_usage = []

        def _create_completion(self, request_kwargs):
            if self.last_api_error is not None:
                raise self.last_api_error
            request_kwargs = dict(request_kwargs)
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body["provider"] = dict(self.provider_options)
            extra_body["reasoning"] = {"enabled": False}
            request_kwargs["extra_body"] = extra_body
            started = time.monotonic()
            try:
                # Call the SDK directly: the generic client's mandatory-reasoning
                # retry must not silently replace this experiment's policy.
                with api_request_deadline(self.request_timeout_seconds):
                    response = self.client.chat.completions.create(**request_kwargs)
                usage = getattr(response, "usage", None)
                record = {
                    "timestamp": utc_now(),
                    "task_id": self.current_task_id,
                    "request_model": request_kwargs.get("model"),
                    "request_max_tokens": request_kwargs.get("max_tokens"),
                    "request_max_completion_tokens": request_kwargs.get("max_completion_tokens"),
                    "request_reasoning": dict(extra_body["reasoning"]),
                    "request_provider": dict(extra_body["provider"]),
                    "request_total_deadline_seconds": self.request_timeout_seconds,
                    "response_id": getattr(response, "id", None),
                    "response_model": getattr(response, "model", None),
                    "upstream_provider": getattr(response, "provider", None),
                    "latency_seconds": time.monotonic() - started,
                    **{key: getattr(usage, key, None) for key in (
                        "prompt_tokens", "completion_tokens", "total_tokens", "cost")},
                    "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
                    "cached_tokens": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
                    "finish_reason": getattr(response.choices[0], "finish_reason", None) if response.choices else None,
                }
                violations, warnings = [], []
                reasoning_tokens = record["reasoning_tokens"]
                if isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0:
                    violations.append({"policy": "reasoning_disabled", "observed_reasoning_tokens": reasoning_tokens,
                                       "requested_reasoning": {"enabled": False}})
                completion_tokens, requested_max = record["completion_tokens"], record["request_max_tokens"]
                if (isinstance(completion_tokens, (int, float)) and isinstance(requested_max, (int, float))
                        and completion_tokens > requested_max):
                    warnings.append({"observation": "completion_tokens_exceed_request_max_tokens",
                                     "completion_tokens": completion_tokens, "request_max_tokens": requested_max})
                record["response_policy_violations"] = violations
                record["response_policy_status"] = "violated" if violations else "no_reported_violation"
                record["response_audit_warnings"] = warnings
                append_jsonl(usage_path, record)
                self.task_usage.append(record)
                if record["response_model"] != expected_model:
                    raise ModelIdentityError(f"Expected {expected_model}, got {record['response_model']}")
                if violations:
                    # Preserve paid usage above, but never evaluate or journal
                    # this response as an ordinary model benchmark outcome.
                    raise ResponsePolicyError(violations)
                return response
            except (api_error_class, ModelIdentityError, ResponsePolicyError, APIRequestDeadlineError, json.JSONDecodeError) as exc:
                self.last_api_error = exc
                self.task_api_errors.append({
                    "timestamp": utc_now(), "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                    "error": str(exc), "latency_seconds": time.monotonic() - started,
                    "request_max_tokens": request_kwargs.get("max_tokens"),
                    "request_reasoning": dict(extra_body["reasoning"]),
                    "request_provider": dict(extra_body["provider"]),
                    "request_total_deadline_seconds": self.request_timeout_seconds,
                    "response_policy_violations": getattr(exc, "violations", []),
                })
                raise

        def generate(self, *args, **kwargs):
            if self.last_api_error is not None:
                raise self.last_api_error
            try:
                return super().generate(*args, **kwargs)
            except ValueError as exc:
                if "returned empty message content" in str(exc) and self.task_usage:
                    # A completed empty model response is an executable-output
                    # failure, not a local RAG or configuration failure.
                    return "", int(self.task_usage[-1].get("total_tokens") or 0)
                raise

    return TrackedClient


def run_tracked_task(runner, task: dict, method: str):
    """Handle both swallowed and propagated failures through the same latch."""
    try:
        return runner.run_single_task(task, method)
    except Exception:
        from src.evaluation.mlip_resources import get_resource_failure
        if runner.llm_client.last_api_error is None and get_resource_failure() is None:
            raise
        return None


def record_mlip_resource_failure(output, task, update_status, phase, result=None):
    """Recover resource faults even when a generic runner catches the exception."""
    from src.evaluation.mlip_resources import get_resource_failure
    exc = get_resource_failure()
    if exc is None:
        return False
    failure = {"timestamp": utc_now(), "task_id": task["task_id"], "task_type": task["task_type"],
               "fatal": False, "error_type": type(exc).__name__, "error": str(exc),
               "phase": phase, "failure_kind": "mlip_resource", "status_code": None}
    append_jsonl(output / "infrastructure_failures.jsonl", failure)
    if result is not None:
        atomic_json(output / f"mlip_resource_error_{task['task_id']}.json", result.to_dict())
    update_status(state="retryable_mlip_resource_error", error=failure)
    return True


def save_compatible_results(output: Path, records: list[dict], task_type: str, model: str, method: str) -> None:
    from src.benchmarks.base import BenchmarkResult, BenchmarkSummary
    results = [BenchmarkResult(**record) for record in records]
    summary = BenchmarkSummary.from_results(results, task_type=task_type, method=method, model=model)
    atomic_json(output / f"{model}_{method}_results.json", records)
    atomic_json(output / f"{model}_{method}_summary.json", summary.to_dict())


def run_worker(run_dir: Path, task_type: str, shard_index: int = 0, shard_count: int = 1) -> int:
    run_dir = run_dir.resolve()
    runtime = run_dir / "runtime"
    output = run_dir / "workers" / task_type
    if shard_count > 1:
        output = output / f"shard_{shard_index:02d}"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "run_status.json"
    status = {"task_type": task_type, "pid": os.getpid(), "started_at": utc_now(), "state": "initializing"}

    def update_status(**values):
        status.update(values, updated_at=utc_now())
        atomic_json(status_path, status)

    with (output / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Worker already running for {task_type}", flush=True)
            return 22
        try:
            update_status()
            config = json.loads((run_dir / "run_config.json").read_text())
            manifest = json.loads((run_dir / "manifest.json").read_text())
            tasks = selected_tasks(runtime, manifest, task_type)
            if not 0 <= shard_index < shard_count:
                raise ValueError("Invalid shard assignment")
            limit = config.get("study_limit_per_family", 100)
            tasks = [task for i, task in enumerate(tasks[:limit]) if i % shard_count == shard_index]
            model, method = config["model"], config["method"]
            if model != "openrouter-deepseek-v3.2":
                raise ValueError("This run requires model openrouter-deepseek-v3.2")
            records = read_results(output / "results.jsonl")
            selected_ids = {task["task_id"] for task in tasks}
            if any(record["task_id"] not in selected_ids or record["task_type"] != task_type
                   or record["model"] != model or record["method"] != method for record in records):
                raise ValueError("Completed journal contains records outside the frozen experiment")
            completed = {record["task_id"] for record in records}
            update_status(total=len(tasks), completed=len(completed))

            # Never import project modules before selecting the frozen runtime.
            sys.path.insert(0, str(runtime))
            os.chdir(runtime)
            if config.get("env_file"):
                from dotenv import load_dotenv
                load_dotenv(config["env_file"], override=False)
            base_url = config.get("base_url", "https://openrouter.ai/api/v1")
            if base_url.rstrip("/") != "https://openrouter.ai/api/v1":
                raise ValueError("This run requires the OpenRouter API endpoint")
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENROUTER_BASE_URL"] = base_url
            if config.get("mlip_device"):
                os.environ["MATERIALSPILOT_MLIP_DEVICE"] = config["mlip_device"]
            os.environ["MATERIALSPILOT_MAX_MLIP_ATOMS"] = str(config.get("max_mlip_atoms", 300))
            os.environ["MATERIALSPILOT_API_TIMEOUT_SECONDS"] = str(config.get("api_timeout_seconds", 180))
            from openai import APIError
            import src.benchmarks.runner as runner_module
            from src.benchmarks.llm_client import LLMClient
            from src.evaluation.mlip_resources import reset_resource_failure
            from src.benchmarks.reporting import _attach_protocol_metrics

            save_compatible_results(output, records, task_type, model, method)
            if len(completed) == len(tasks):
                update_status(state="complete", finished_at=utc_now())
                return 0
            tracked = tracked_client_class(LLMClient, APIError, output / "api_usage.jsonl",
                                           config.get("expected_response_model", "deepseek/deepseek-v3.2"))
            # Must happen before construction: orchestrator agents retain this client.
            runner_module.LLMClient = tracked
            methods = method in {"agent_methods", "methods_agent", "execution_feedback"}
            runner = runner_module.UnifiedBenchmarkRunner(
                task_type=task_type, model=model, output_dir=output, project_root=runtime,
                enable_mlip=config.get("enable_mlip", False),
                mlip_model_type=config.get("mlip_model", "chgnet"),
                mlip_device=config.get("mlip_device", "auto"),
                use_advanced_rag=config.get("use_advanced_rag", False),
                use_unified_validation=config.get("use_unified_validation", False),
                use_multi_agent=method == "agent",
                max_agent_iterations=config.get("max_agent_iterations", 3),
                use_methods_feedback=methods,
                methods_feedback_transformer=config.get("methods_feedback_transformer", "rule"),
                methods_feedback_context_expansion=config.get("methods_feedback_context_expansion", True),
            )
            if methods and runner.execution_feedback_orchestrator is None:
                raise RuntimeError("Methods feedback orchestrator initialization failed")
            client = runner.llm_client
            if not isinstance(client, tracked):
                raise RuntimeError("Runner did not receive the tracked LLM client")
            for task in tasks:
                if (run_dir / "STOP").exists():
                    update_status(state="stopped")
                    return 0
                task_id = task["task_id"]
                if task_id in completed:
                    continue
                client.begin_task(task_id)
                reset_resource_failure()
                update_status(state="running", current_task_id=task_id, task_started_at=utc_now())
                print(f"[{task_type}] {len(completed) + 1}/{len(tasks)} {task_id}", flush=True)
                result = run_tracked_task(runner, task, method)
                # Runner catches API exceptions as failed model results. Recover
                # the latched exception before any result enters the journal.
                if client.last_api_error is not None:
                    exc = client.last_api_error
                    fatal = fatal_api_error(exc)
                    failure = {"timestamp": utc_now(), "task_id": task_id, "task_type": task_type,
                               "fatal": fatal, "error_type": type(exc).__name__, "error": str(exc),
                               "status_code": getattr(exc, "status_code", None),
                               "api_errors": client.task_api_errors}
                    append_jsonl(output / "infrastructure_failures.jsonl", failure)
                    update_status(state="fatal_api_error" if fatal else "retryable_api_error", error=failure)
                    return 21 if fatal else 20
                if record_mlip_resource_failure(output, {**task, "task_type": task_type}, update_status, "generation", result):
                    return 20
                if result is None or result.metadata.get("method_profile") != "paper_aligned_v1" or not result.metadata.get("state_history"):
                    if result is not None:
                        atomic_json(output / f"local_error_{task_id}.json", result.to_dict())
                    raise RuntimeError("Paper workflow did not complete a policy iteration; preserve as infrastructure error")
                _attach_protocol_metrics(result, task, output,
                                         config.get("enable_mlip", False), config.get("mlip_model", "chgnet"))
                if record_mlip_resource_failure(output, {**task, "task_type": task_type}, update_status, "posthoc_evaluation", result):
                    return 20
                if "evaluation_protocol_error" in result.metadata:
                    # Preserve the generated candidate for diagnosis without
                    # treating a scorer failure as a valid benchmark outcome.
                    atomic_json(output / f"protocol_error_{task_id}.json", result.to_dict())
                    raise RuntimeError(f"Protocol evaluation failed: {result.metadata['evaluation_protocol_error']}")
                result.metadata["api_tracking"] = {
                    "response_models": sorted({call["response_model"] for call in client.task_usage}),
                    "upstream_providers": sorted({call["upstream_provider"] for call in client.task_usage
                                                   if call["upstream_provider"]}),
                    "call_count": len(client.task_usage),
                    "cost": sum(call["cost"] or 0.0 for call in client.task_usage),
                    "routing": "openrouter_default_with_parameter_support_filter",
                    "provider_options": client.provider_options,
                    "request_total_deadline_seconds": client.request_timeout_seconds,
                    "response_policy": "reject_reported_reasoning_tokens_when_disabled",
                    "reasoning": {"enabled": False},
                }
                result.metadata["artifact_directory"] = str(output.relative_to(run_dir))
                record = result.to_dict()
                append_jsonl(output / "results.jsonl", record)
                records.append(record)
                completed.add(task_id)
                save_compatible_results(output, records, task_type, model, method)
                update_status(completed=len(completed), last_completed_task_id=task_id,
                              last_completed_at=utc_now(), current_task_id=None)
                print(f"[{task_type}] completed={len(completed)} success={result.success} score={result.score:.4f}", flush=True)
                if len(completed) < len(tasks):
                    time.sleep(max(0.0, float(config.get("rate_limit_seconds", 2))))
            update_status(state="complete", finished_at=utc_now())
            return 0
        except Exception as exc:
            update_status(state="local_error", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
            return 22


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-type", choices=("defect", "adsorption", "grain_boundary", "interface"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    return run_worker(args.run_dir, args.task_type, args.shard_index, args.shard_count)


if __name__ == "__main__":
    raise SystemExit(main())
