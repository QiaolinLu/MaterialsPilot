<h1 align="center">[NeurIPS 2026] MaterialsPilot</h1>

<p align="center">
  <strong>Code for our paper</strong>
</p>

<p align="center">
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/155349">
    <strong>MaterialsPilot: An Execution-Feedback Framework for Generative Design<br/>
    of Complex Atomistic Architectures</strong>
  </a>
</p>

<p align="center">
  Qiaolin Lu · Tongliang Liu · Qiang Qu · Bo Han · Yi Chang · Chengqi Zhang
</p>

<p align="center">
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/155349">Paper</a> ·
  <a href="https://openreview.net/forum?id=6I2wy66kZ8">OpenReview</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#data-and-knowledge">Data</a> ·
  <a href="docs/implementation.md">Method &amp; evaluation</a>
</p>

MaterialsPilot generates atomistic structures from natural-language specifications through iterative Python program synthesis and execution. This repository provides the core implementation, materials knowledge, pretrained retrieval assets, and a four-category benchmark.

## Method

- **Knowledge retrieval:** three scientific channels combine composition similarity from pretrained Mat2Vec, structural similarity from pretrained CGCNN, and property similarity through reciprocal rank fusion (RRF). The property channel operates when a supported explicit target is available. The bundled models use real pretrained weights.
- **Execution feedback:** generated construction programs execute in a subprocess. Geometry, chemistry, CHGNet reference-relative energy, and task constraints inform subsequent revisions, for at most five generation attempts. The final attempt is returned; candidates are not selected by benchmark score.
- **Evaluation:** task success rate (TSR) and constraint satisfaction rate (CSR) use the same hard-constraint set. TSR requires structural validity and every required constraint to pass; CSR averages constraint scores per sample. Invalid outputs contribute zero to aggregate scores.

## Data and knowledge

The benchmark contains **400 project-defined tasks**, with 100 each for defect engineering, surface adsorption, grain boundaries, and heterostructure interfaces. These are not official JARVIS or Open Catalyst Project (OCP) splits. The [task manifest](configs/benchmark_four_tasks_400_manifest.json) records task identifiers and source hashes.

Knowledge is organized into **L1 foundation**, **L2 structures**, and **L3 applications** under [`data/knowledge`](data/knowledge). Foundation knowledge includes general physical and API references. Retrieval assets and their provenance are stored separately in [`data/retrieval_assets`](data/retrieval_assets).

```text
run.py                 Benchmark entry point
src/                   Generation, execution feedback, retrieval and evaluation
data/knowledge/        L1 / L2 / L3 knowledge
data/retrieval_assets/  Pretrained assets, provenance and licenses
tasks/                 Four datasets, 100 tasks each
config/                Generator configuration and prompt
configs/               Run profile and task manifest
scripts/               Benchmark workers and run snapshots
```

## Installation

Use Python 3.11 on Linux in a dedicated environment, with a PyTorch installation appropriate for your hardware. PyTorch is required for CGCNN retrieval as well as CHGNet evaluation. Run these commands from the repository root:

```bash
pip install -r requirements.txt
```

## Quick start

Copy the credential template outside the repository, then set `OPENROUTER_API_KEY` in that file:

```bash
cp .env.example ../materialspilot.env
```

The supplied profile uses OpenRouter `deepseek/deepseek-v3.2` with reasoning disabled. An existing `OPENAI_API_KEY` takes precedence, so ensure it is intended for this endpoint. Running the benchmark makes paid API requests.

Run all 400 tasks with 16 concurrent workers and at most four concurrent CPU CHGNet evaluations:

```bash
python -B run.py \
  --run-dir ../runs/materialspilot_400 \
  --env-file ../materialspilot.env \
  --shards-per-family 4 --max-parallel 16 \
  --mlip-device cpu --mlip-parallel 4
```

For an optional eight-task smoke run, use a separate output directory:

```bash
python -B run.py \
  --run-dir ../runs/materialspilot_smoke \
  --env-file ../materialspilot.env \
  --limit-per-family 2 --shards-per-family 2 --max-parallel 4 \
  --mlip-device cpu --mlip-parallel 4
```

Each new run automatically freezes code, inputs, knowledge, assets, and configuration before execution. Add `--prepare-only` to create the snapshot without generation. Resume with the same command and run directory; completed outcomes, including failures, are retained. To stop, create `STOP` inside the run directory. Remove that file before resuming. Existing runs retain their frozen settings; use a new directory to change the configuration.

## Outputs and interpretation

The run directory contains `progress.json`, `REPORT.txt`, and `exit_status.json`; `summary.json` is written on completion. Per-family shard directories under `workers/` contain `results.jsonl`, generated structures, `api_usage.jsonl`, infrastructure-failure records, and logs. `runtime/` and `snapshot_hashes.json` preserve the evaluated snapshot. Partial summaries use completed observations.

Missing references or unsupported measurements remain explicitly unknown, with conservative scores and disclosed coverage; they are not confirmed physical failures. Supplementary relaxed TSR excludes unknown constraints and is **not the paper TSR**. Reproduction assumptions, resource limits, and execution precautions are documented in [Implementation notes](docs/implementation.md); this release does not claim exact numerical reproduction of the paper.

## Attribution and citation

Third-party terms are preserved in the [Mat2Vec license](data/retrieval_assets/mat2vec_LICENSE), [CGCNN license](data/retrieval_assets/cgcnn_LICENSE), and [NOTICE](NOTICE.txt). These notices do not assign a new license to the project itself.

Please cite the paper through its [NeurIPS 2026 conference record](https://neurips.cc/virtual/2026/loc/sydney/poster/155349) or [OpenReview record](https://openreview.net/forum?id=6I2wy66kZ8).
