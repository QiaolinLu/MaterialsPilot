#!/usr/bin/env python3
"""Run the MaterialsPilot benchmark with frozen inputs and resumable workers."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "benchmarks"))

from run_paper_study import main


if __name__ == "__main__":
    raise SystemExit(main())
