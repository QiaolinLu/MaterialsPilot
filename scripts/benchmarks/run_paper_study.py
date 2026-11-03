#!/usr/bin/env python3
"""Run a frozen paper-aligned study with disjoint family shards and resumable journals."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from study_io import PROJECT_ROOT, TASKS, atomic_json, now, prepare, read_jsonl, stop_workers, verify_snapshot


def study_families(config):
    """Use frozen scope; older study configurations retain all four families."""
    families = config.get("study_task_types", config.get("task_types", list(TASKS)))
    if (not isinstance(families, (list, tuple)) or not families
            or any(family not in TASKS for family in families)
            or len(set(families)) != len(families)):
        raise ValueError("Study families must be a nonempty, unique subset of the four task families")
    if "study_task_types" in config and config.get("task_types") != list(families):
        raise ValueError("Frozen study_task_types and task_types disagree")
    return tuple(families)


def study_total(config):
    total = len(study_families(config)) * config["study_limit_per_family"]
    if "study_task_types" in config and config.get("total_tasks") != total:
        raise ValueError("Frozen total_tasks differs from the selected study size")
    return total


def scoped_journal_rows(run_dir, families, filename):
    """Reject accidental mixing of a dedicated study with other families."""
    records = []
    for path in sorted((run_dir / "workers").glob("**/" + filename)):
        rows = read_jsonl(path)
        family = path.relative_to(run_dir / "workers").parts[0]
        if family not in families and rows:
            raise RuntimeError(f"Journal outside frozen study families: {path}")
        records.extend(rows)
    return records


def collect_rows(run_dir):
    rows = {}
    for path in sorted((run_dir / "workers").glob("**/results.jsonl")):
        for row in read_jsonl(path):
            key = (row["task_type"], row["task_id"])
            if key in rows:
                raise RuntimeError(f"Duplicate study observation: {key}")
            rows[key] = row
    return rows


def report(run_dir, config, active, state):
    families = study_families(config)
    sys.path.insert(0, str(run_dir / "runtime"))
    from src.evaluation import protocol as frozen_protocol
    if not Path(frozen_protocol.__file__).resolve().is_relative_to((run_dir / "runtime").resolve()):
        raise RuntimeError("Progress aggregation imported an unfrozen evaluation module")
    summarize_protocol_results = frozen_protocol.summarize_protocol_results
    from src.evaluation.exploratory import summarize_relaxed_tsr
    rows = list(collect_rows(run_dir).values())
    if any(row["task_type"] not in families for row in rows):
        raise RuntimeError("Results outside frozen study families cannot be mixed into this study")
    usage = scoped_journal_rows(run_dir, families, "api_usage.jsonl")
    failures = scoped_journal_rows(run_dir, families, "infrastructure_failures.jsonl")
    categories = {}
    for family in families:
        subset = [r for r in rows if r["task_type"] == family]
        metrics = summarize_protocol_results(subset)
        unknowns = sum(bool(r.get("metadata", {}).get("evaluation_protocol", {}).get("unknown_constraints")) for r in subset)
        categories[family] = {"completed": len(subset), "target": config["study_limit_per_family"],
                              "samples_with_unknown_constraints": unknowns, **metrics,
                              **summarize_relaxed_tsr(subset)}
    result = {
        "updated_at": now(), "state": state, "completed": len(rows),
        "total": study_total(config), "task_types": list(families), "by_task": categories,
        "overall": {**summarize_protocol_results(rows), **summarize_relaxed_tsr(rows)},
        "model": config["model_id"], "method_profile": "paper_aligned_v1",
        "active_workers": [{"name": name, "pid": proc.pid} for name, proc in active.items()],
        "api_calls": len(usage), "recorded_cost_usd": sum(u.get("cost") or 0 for u in usage),
        "calls_without_reported_cost": sum(u.get("cost") is None for u in usage),
        "infrastructure_failures": len(failures),
    }
    atomic_json(run_dir / "progress.json", result)
    lines = ["Paper-aligned MaterialsPilot study", f"state={state}; {len(rows)}/{result['total']}; updated={result['updated_at']}",
             "Study families: " + ", ".join(families),
             "family\tcompleted\tSR\tCSR\tTSR\tAvgScore"]
    for family, stats in categories.items():
        lines.append(f"{family}\t{stats['completed']}\t{stats.get('sr', 0):.4f}\t{stats.get('csr', 0):.4f}\t{stats.get('tsr', 0):.4f}\t{stats.get('avg_score', 0):.4f}")
    lines.append(f"API calls={len(usage)}; reported USD={result['recorded_cost_usd']:.6f}; calls without reported cost={result['calls_without_reported_cost']}")
    lines.append("Partial results use completed observations only. Unknown measurements are disclosed in per-sample protocols.")
    lines.append(f"Exploratory relaxed TSR={result['overall']['relaxed_tsr']:.4f}; this excludes unknown constraints and is not the paper TSR.")
    (run_dir / "REPORT.txt").write_text("\n".join(lines) + "\n")
    return result


def supervise(run_dir, config):
    families = study_families(config)
    from dotenv import load_dotenv
    load_dotenv(config["env_file"], override=False)
    env = os.environ.copy()
    # Preserve the project's configured OPENAI_API_KEY priority. The model
    # configuration uses OPENROUTER_API_KEY only when that primary key is absent.
    env.update(PYTHONPATH=str(run_dir / "runtime"), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OPENAI_BASE_URL="https://openrouter.ai/api/v1", OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
               MATERIALSPILOT_MLIP_DEVICE=config["mlip_device"],
               MATERIALSPILOT_MAX_MLIP_ATOMS=str(config.get("max_mlip_atoms", 300)),
               MATERIALSPILOT_PROVIDER_OPTIONS_JSON=json.dumps(config.get("request_options", {}).get("provider", {})),
               MATERIALSPILOT_MLIP_SLOT_DIR=str(run_dir / "mlip_resource_slots"),
               MATERIALSPILOT_MLIP_MAX_CONCURRENT=str(config.get("mlip_max_concurrent", 4)),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    jobs = [(family, shard) for shard in range(config["study_shards_per_family"]) for family in families]
    active, handles, attempts, finished = {}, {}, {}, set()
    api_failures_by_task = {}
    state = "running"
    try:
        while True:
            if (run_dir / "STOP").exists():
                state = "stopped"; break
            fatal = False
            for key in list(active):
                code = active[key].poll()
                if code is None:
                    continue
                handles.pop(key).close(); del active[key]
                if code == 0:
                    finished.add(key)
                elif code == 21:
                    state = "blocked_api_credentials_balance_or_identity"; fatal = True
                elif code == 20:
                    family, shard = key.split(":")
                    directory = run_dir / "workers" / family
                    if config["study_shards_per_family"] > 1:
                        directory /= f"shard_{int(shard):02d}"
                    status = json.loads((directory / "run_status.json").read_text())
                    failed_task = (key, status.get("current_task_id", "unknown"))
                    api_failures_by_task[failed_task] = api_failures_by_task.get(failed_task, 0) + 1
                    if api_failures_by_task[failed_task] >= config.get("max_infrastructure_attempts_per_task", config.get("max_api_attempts_per_task", 3)):
                        state = "retry_exhausted_for_task"; fatal = True
                else:
                    state = "local_or_retry_exhausted_error"; fatal = True
                print(now(), key, "exit", code, flush=True)
            if fatal:
                break
            for family, shard in jobs:
                key = f"{family}:{shard}"
                if key in active or key in finished:
                    continue
                if len(active) >= config["study_max_parallel"]:
                    break
                attempts[key] = attempts.get(key, 0) + 1
                directory = run_dir / "workers" / family
                if config["study_shards_per_family"] > 1:
                    directory /= f"shard_{shard:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                command = [config["python"], "-u", "-B", str(run_dir / "runtime/scripts/benchmarks/materials400_worker.py"),
                           "--run-dir", str(run_dir), "--task-type", family,
                           "--shard-index", str(shard), "--shard-count", str(config["study_shards_per_family"])]
                handle = (directory / "run.log").open("a")
                process = subprocess.Popen(command, cwd=run_dir / "runtime", env=env, stdin=subprocess.DEVNULL,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[key], handles[key] = process, handle
                atomic_json(directory / "launch.json", {"pid": process.pid, "command": command,
                                                        "started_at": now(), "attempt": attempts[key]})
            progress = report(run_dir, config, active, state)
            if len(finished) == len(jobs):
                state = "complete" if progress["completed"] == progress["total"] else "incomplete"
                break
            time.sleep(5)
    except BaseException:
        state = "coordinator_failed"
        raise
    finally:
        stop_workers(active)
        for handle in handles.values():
            handle.close()
        result = report(run_dir, config, {}, state)
        atomic_json(run_dir / "exit_status.json", result)
        if state == "complete":
            atomic_json(run_dir / "summary.json", result)
    return 0 if state == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--limit-per-family", type=int, default=100)
    parser.add_argument("--shards-per-family", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=16)
    parser.add_argument("--mlip-device", choices=("cuda", "cpu", "auto"), default="cpu")
    parser.add_argument("--mlip-parallel", type=int, default=4)
    parser.add_argument("--families", choices=TASKS, nargs="+", default=None,
                        help="Task families for a new study (default: all four); resume uses the frozen scope")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    families = args.families if args.families is not None else list(TASKS)
    if len(set(families)) != len(families):
        raise ValueError("Duplicate study family arguments are not allowed")
    if not 1 <= args.limit_per_family <= 100 or not 1 <= args.shards_per_family <= args.limit_per_family or args.max_parallel < 1 or args.mlip_parallel < 1:
        raise ValueError("Invalid study size/sharding")
    run_dir = args.run_dir.resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        os.environ["OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
        new = not (run_dir / "run_config.json").exists()
        config = prepare(run_dir, PROJECT_ROOT / "configs/benchmark_400.json",
                         PROJECT_ROOT / "configs/benchmark_four_tasks_400_manifest.json",
                         args.env_file, args.mlip_device)
        if new:
            config.update(study_limit_per_family=args.limit_per_family,
                          study_task_types=list(families), task_types=list(families),
                          total_tasks=len(families) * args.limit_per_family,
                          study_shards_per_family=args.shards_per_family,
                          study_max_parallel=args.max_parallel,
                          workers=args.max_parallel,
                          max_mlip_atoms=300,
                          api_timeout_seconds=180,
                          sdk_retries=0,
                          max_api_attempts_per_task=3,
                          max_infrastructure_attempts_per_task=3,
                          mlip_max_concurrent=args.mlip_parallel,
                          mlip_prediction_mode="energy_only_no_grad",
                          method_profile="paper_aligned_v1")
            config["request_options"] = {
                "reasoning": {"enabled": False},
                "provider": {"require_parameters": True, "ignore": ["GMICloud"]},
            }
            config["api_total_deadline_seconds"] = config["api_timeout_seconds"]
            config["response_policy"] = "Reject positive reported reasoning tokens; retry as infrastructure failure"
            atomic_json(run_dir / "run_config.json", config)
            hashes = json.loads((run_dir / "snapshot_hashes.json").read_text())
            hashes["run_config.json"] = hashlib.sha256((run_dir / "run_config.json").read_bytes()).hexdigest()
            atomic_json(run_dir / "snapshot_hashes.json", hashes)
        elif args.families is not None and set(families) != set(study_families(config)):
            raise ValueError("Cannot change frozen study families; prepare a separate run directory")
        study_total(config)  # Validate the frozen scope before any worker starts.
        verify_snapshot(run_dir)
        # prepare imports src.config from the source tree to resolve the model.
        # The controller has no active src objects after configuration resolves;
        # discard that package cache so progress imports the frozen scorer too.
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
        sys.path.insert(0, str(run_dir / "runtime"))
        atomic_json(run_dir / "coordinator.json", {"pid": os.getpid(), "updated_at": now()})
        if args.prepare_only:
            print(json.dumps({"run_dir": str(run_dir), "tasks": study_total(config),
                              "task_types": list(study_families(config)),
                              "parallel": config["study_max_parallel"]})); return 0
        return supervise(run_dir, config)


if __name__ == "__main__":
    raise SystemExit(main())
