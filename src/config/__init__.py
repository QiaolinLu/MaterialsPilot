"""
CrystalPilot Configuration Module.

Centralized configuration for models, API keys, and paths.
"""

from pathlib import Path

from .models import (
    ModelConfig,
    AVAILABLE_MODELS,
    SILICONFLOW_MODELS,
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    EMBEDDING_MODEL,
    PLATFORM_BASE_URLS,
    PLATFORM_API_KEY_ENVS,
    get_api_key,
    get_base_url,
    get_model_id,
    get_model_config,
    list_models,
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Output directories
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
RESULTS_DIR = BENCHMARKS_DIR / "results"
TASKS_DIR = PROJECT_ROOT / "tasks"
DATA_DIR = PROJECT_ROOT / "data"

# ChromaDB configuration
CHROMA_DB_PATH = str(DATA_DIR / "benchmark_v1" / "chroma_db")
COLLECTION_NAME = "material_science_docs"

__all__ = [
    # Model config
    "ModelConfig",
    "AVAILABLE_MODELS",
    "SILICONFLOW_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "EMBEDDING_MODEL",
    "PLATFORM_BASE_URLS",
    "PLATFORM_API_KEY_ENVS",
    "get_api_key",
    "get_base_url",
    "get_model_id",
    "get_model_config",
    "list_models",
    # Paths
    "PROJECT_ROOT",
    "BENCHMARKS_DIR",
    "RESULTS_DIR",
    "TASKS_DIR",
    "DATA_DIR",
    # ChromaDB
    "CHROMA_DB_PATH",
    "COLLECTION_NAME",
]
