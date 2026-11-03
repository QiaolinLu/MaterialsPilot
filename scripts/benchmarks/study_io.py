"""Snapshot, dataset verification, journal I/O and process helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("defect", "adsorption", "grain_boundary", "interface")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A live worker may still be appending the final line.
    return records


def verify_snapshot(run_dir):
    for name, digest in json.loads((run_dir / "snapshot_hashes.json").read_text()).items():
        if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen file changed: {name}")


def prepare(run_dir, config_path, manifest_path, env_file, mlip_device):
    if (run_dir / "run_config.json").exists():
        verify_snapshot(run_dir)
        return json.loads((run_dir / "run_config.json").read_text())
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if [t["task_type"] for t in config["tasks"]] != list(TASKS):
        raise ValueError("Exactly the four task families must be configured")
    if any(t["limit"] != 100 for t in config["tasks"]):
        raise ValueError("Each task family must contain 100 samples")
    if manifest["total_tasks"] != 400 or len(manifest["families"]) != 4:
        raise ValueError("Expected the fixed 400-task manifest")
    if [f["task_type"] for f in manifest["families"]] != list(TASKS):
        raise ValueError("Manifest families differ from the configuration")
    for family in manifest["families"]:
        source = PROJECT_ROOT / family["data_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != family["data_sha256"]:
            raise ValueError(f"Dataset changed: {source}")
        data = json.loads(source.read_text())
        tasks = data["tasks"] if isinstance(data, dict) else data
        if len(tasks) != 100 or len(family["samples"]) != 100:
            raise ValueError("Dataset and manifest must both contain 100 records")
        if [s["index"] for s in family["samples"]] != list(range(100)):
            raise ValueError("Manifest must cover each source row once, in order")
        ids = [t["task_id"] for t in tasks]
        if len(set(ids)) != 100:
            raise ValueError("Duplicate task IDs")
        for item, task in zip(family["samples"], tasks):
            digest = hashlib.sha256(json.dumps(task, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
            if item["task_id"] != task["task_id"] or digest != item["task_sha256"]:
                raise ValueError("Task content differs from the audited manifest")
    from dotenv import load_dotenv
    load_dotenv(env_file, override=False)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import get_model_config, get_api_key
    model = get_model_config(config["model"])
    get_api_key(model.api_key_env)  # Validate availability, never serialize the key.
    if model.model_id != "deepseek/deepseek-v3.2":
        raise ValueError("Expected DeepSeek V3.2 on OpenRouter")
    if model.base_url.rstrip("/") != "https://openrouter.ai/api/v1":
        raise ValueError("Expected the normal OpenRouter endpoint")
    runtime = run_dir / "runtime"
    runtime.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".env")
    for name in ("src", "tasks", "data", "config", "configs"):
        shutil.copytree(PROJECT_ROOT / name, runtime / name, ignore=ignore)
    shutil.copy2(PROJECT_ROOT / "run.py", runtime)
    scripts = runtime / "scripts/benchmarks"
    scripts.mkdir(parents=True)
    for name in ("materials400_worker.py", "study_io.py", "run_paper_study.py"):
        shutil.copy2(PROJECT_ROOT / "scripts/benchmarks" / name, scripts / name)
    shutil.copytree(PROJECT_ROOT / "docs", runtime / "docs", ignore=ignore)
    for name in ("requirements.txt", "NOTICE.txt"):
        if (PROJECT_ROOT / name).exists():
            shutil.copy2(PROJECT_ROOT / name, runtime / name)
    atomic_json(run_dir / "manifest.json", manifest)
    config.update(created_at=now(), model_id=model.model_id,
                  expected_response_model=model.model_id, base_url=model.base_url,
                  temperature=model.temperature, max_tokens=model.max_tokens,
                  env_file=str(env_file.resolve()), python=sys.executable,
                  mlip_device=mlip_device, workers=4, task_types=list(TASKS), total_tasks=400,
                  request_options={"reasoning": {"enabled": False}},
                  max_worker_attempts=3, resume_policy="Preserve first completed observation, including failures")
    atomic_json(run_dir / "run_config.json", config)
    packages = {}
    for package in ("numpy", "scipy", "pymatgen", "ase", "torch", "chgnet", "openai"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    checkpoints = {}
    try:
        distribution = importlib.metadata.distribution("chgnet")
        for relative in distribution.files or []:
            if str(relative).endswith((".pth.tar", ".pth", ".pt")):
                path = Path(distribution.locate_file(relative))
                if path.is_file():
                    checkpoints[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    except importlib.metadata.PackageNotFoundError:
        pass
    atomic_json(run_dir / "environment.json", {"python": sys.version, "packages": packages,
                                               "chgnet_checkpoint_hashes": checkpoints})
    hashes = {str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(runtime.rglob("*")) if p.is_file()}
    for name in ("manifest.json", "run_config.json", "environment.json"):
        hashes[name] = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
    atomic_json(run_dir / "snapshot_hashes.json", hashes)
    return config


def stop_workers(active):
    for process in active.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in active.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
