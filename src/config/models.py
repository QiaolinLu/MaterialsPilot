"""
LLM Model Configuration.

Centralized configuration for all LLM models used in CrystalPilot.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str
    model_id: str
    api_key_env: str = "SILICONFLOW_API_KEY"
    base_url: str = "https://api.siliconflow.cn/v1"
    max_tokens: int = 4096
    temperature: float = 0.3


# =============================================================================
# API Configuration
# =============================================================================

# Default platform base URLs (can be overridden by environment variables)
_DEFAULT_BASE_URLS = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "bianxie": "https://api.bianxie.ai/v1",
}

# OpenAI-compatible providers are routed through the OpenAI SDK interface.
# Prefer the generic OpenAI env names; keep legacy provider-specific names as
# fallbacks so existing local .env files continue to work.
_OPENAI_COMPAT_API_KEY_ENV = "OPENAI_API_KEY"
_OPENAI_COMPAT_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("OPENROUTER_BASE_URL", _DEFAULT_BASE_URLS["openrouter"]),
)
_API_KEY_ENV_FALLBACKS = {
    "OPENAI_API_KEY": ("OPENROUTER_API_KEY",),
}

# Platform base URLs (read from environment or use defaults)
PLATFORM_BASE_URLS = {
    "siliconflow": os.getenv("SILICONFLOW_BASE_URL", _DEFAULT_BASE_URLS["siliconflow"]),
    "bailian": os.getenv("DASHSCOPE_BASE_URL", _DEFAULT_BASE_URLS["bailian"]),
    "openrouter": _OPENAI_COMPAT_BASE_URL,
    "bianxie": os.getenv("BIANXIE_BASE_URL", _DEFAULT_BASE_URLS["bianxie"]),
}

# Platform API key environment variable names
PLATFORM_API_KEY_ENVS = {
    "siliconflow": "SILICONFLOW_API_KEY",
    "bailian": "DASHSCOPE_API_KEY",
    "openrouter": _OPENAI_COMPAT_API_KEY_ENV,
    "bianxie": "BIANXIE_API_KEY",
}

PLATFORM_BASE_URL_ENVS = {
    "siliconflow": ("SILICONFLOW_BASE_URL",),
    "bailian": ("DASHSCOPE_BASE_URL",),
    "openrouter": ("OPENAI_BASE_URL", "OPENROUTER_BASE_URL"),
    "bianxie": ("BIANXIE_BASE_URL",),
}

DEFAULT_BASE_URL = PLATFORM_BASE_URLS["siliconflow"]


def get_base_url(platform: str = "siliconflow") -> str:
    """Get API base URL for a platform.

    Args:
        platform: Platform name (siliconflow, bailian, openrouter)

    Returns:
        Base URL for the platform
    """
    for env_name in PLATFORM_BASE_URL_ENVS.get(platform, ()):
        if env_name in os.environ:
            return os.environ[env_name]
    if platform in PLATFORM_BASE_URLS:
        return PLATFORM_BASE_URLS[platform]
    return DEFAULT_BASE_URL


def get_api_key(env_name: str = "SILICONFLOW_API_KEY") -> str:
    """Get API key from environment variable."""
    env_names = (env_name, *_API_KEY_ENV_FALLBACKS.get(env_name, ()))
    for candidate in env_names:
        api_key = os.getenv(candidate)
        if api_key:
            return api_key

    if len(env_names) == 1:
        raise ValueError(
            f"{env_name} environment variable is not set. "
            f"Please set it with: export {env_name}='your-api-key'"
        )

    joined_names = " or ".join(env_names)
    raise ValueError(
        f"{joined_names} environment variable is not set. "
        f"Please set it with: export {env_name}='your-api-key'"
    )


# =============================================================================
# Available Models
# =============================================================================

AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    # SiliconFlow models
    "deepseek-v3.2": ModelConfig(
        name="deepseek-v3.2",
        model_id="deepseek-ai/DeepSeek-V3.2",
    ),
    "qwen-2.5": ModelConfig(
        name="qwen-2.5",
        model_id="Qwen/Qwen2.5-72B-Instruct",
    ),
    "kimi-k2": ModelConfig(
        name="kimi-k2",
        model_id="moonshotai/Kimi-K2-Instruct",
    ),
    # Bailian (DashScope) models
    "bailian-qwen3-max": ModelConfig(
        name="bailian-qwen3-max",
        model_id="qwen3-max",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=PLATFORM_BASE_URLS["bailian"],
    ),
    "bailian-qwen-plus": ModelConfig(
        name="bailian-qwen-plus",
        model_id="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=PLATFORM_BASE_URLS["bailian"],
    ),
    "bailian-qwen-flash": ModelConfig(
        name="bailian-qwen-flash",
        model_id="qwen-flash",
        api_key_env="DASHSCOPE_API_KEY",
        base_url=PLATFORM_BASE_URLS["bailian"],
    ),
    # OpenAI-compatible routed models
    "openai-compatible-deepseek-v3.2": ModelConfig(
        name="openai-compatible-deepseek-v3.2",
        model_id="deepseek/deepseek-v3.2",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openai-compatible-deepseek-v4-flash": ModelConfig(
        name="openai-compatible-deepseek-v4-flash",
        model_id="deepseek/deepseek-v4-flash",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openai-compatible-deepseek-v4-pro": ModelConfig(
        name="openai-compatible-deepseek-v4-pro",
        model_id="deepseek/deepseek-v4-pro",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openai-compatible-qwen3.6-plus": ModelConfig(
        name="openai-compatible-qwen3.6-plus",
        model_id="qwen/qwen3.6-plus",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openai-compatible-kimi-k2.6": ModelConfig(
        name="openai-compatible-kimi-k2.6",
        model_id="moonshotai/kimi-k2.6",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openai-compatible-glm-5.1": ModelConfig(
        name="openai-compatible-glm-5.1",
        model_id="z-ai/glm-5.1",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    # OpenRouter models
    "openrouter-claude-sonnet": ModelConfig(
        name="openrouter-claude-sonnet",
        model_id="anthropic/claude-sonnet-4",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-gpt-4o": ModelConfig(
        name="openrouter-gpt-4o",
        model_id="openai/gpt-4o",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-deepseek-v3": ModelConfig(
        name="openrouter-deepseek-v3",
        model_id="deepseek/deepseek-chat-v3-0324",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-deepseek-v3.2": ModelConfig(
        name="openrouter-deepseek-v3.2",
        model_id="deepseek/deepseek-v3.2",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-gpt-5-mini": ModelConfig(
        name="openrouter-gpt-5-mini",
        model_id="openai/gpt-5-mini",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    # Latest OpenRouter China-model endpoints verified on 2026-04-22
    "openrouter-glm-5.1": ModelConfig(
        name="openrouter-glm-5.1",
        model_id="z-ai/glm-5.1",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-glm-4.7": ModelConfig(
        name="openrouter-glm-4.7",
        model_id="z-ai/glm-4.7",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-minimax-m2.7": ModelConfig(
        name="openrouter-minimax-m2.7",
        model_id="minimax/minimax-m2.7",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-minimax-m2.1": ModelConfig(
        name="openrouter-minimax-m2.1",
        model_id="minimax/minimax-m2.1",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-qwen3.6-plus": ModelConfig(
        name="openrouter-qwen3.6-plus",
        model_id="qwen/qwen3.6-plus",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-kimi-k2.6": ModelConfig(
        name="openrouter-kimi-k2.6",
        model_id="moonshotai/kimi-k2.6",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    "openrouter-gemini-3-flash-preview": ModelConfig(
        name="openrouter-gemini-3-flash-preview",
        model_id="google/gemini-3-flash-preview",
        api_key_env=PLATFORM_API_KEY_ENVS["openrouter"],
        base_url=PLATFORM_BASE_URLS["openrouter"],
    ),
    # Bianxie OpenAI-compatible routed models
    "bianxie-deepseek-v3.2": ModelConfig(
        name="bianxie-deepseek-v3.2",
        model_id="deepseek-v3.2",
        api_key_env=PLATFORM_API_KEY_ENVS["bianxie"],
        base_url=PLATFORM_BASE_URLS["bianxie"],
    ),
    "bianxie-deepseek-v3.2-thinking": ModelConfig(
        name="bianxie-deepseek-v3.2-thinking",
        model_id="deepseek-v3.2-thinking",
        api_key_env=PLATFORM_API_KEY_ENVS["bianxie"],
        base_url=PLATFORM_BASE_URLS["bianxie"],
    ),
    "bianxie-gpt-5.4-mini": ModelConfig(
        name="bianxie-gpt-5.4-mini",
        model_id="gpt-5.4-mini",
        api_key_env=PLATFORM_API_KEY_ENVS["bianxie"],
        base_url=PLATFORM_BASE_URLS["bianxie"],
    ),
    "bianxie-gemini-3.1-flash-lite-preview": ModelConfig(
        name="bianxie-gemini-3.1-flash-lite-preview",
        model_id="gemini-3.1-flash-lite-preview",
        api_key_env=PLATFORM_API_KEY_ENVS["bianxie"],
        base_url=PLATFORM_BASE_URLS["bianxie"],
    ),
}

# Model ID mapping (short name -> full model ID)
SILICONFLOW_MODELS: Dict[str, str] = {
    config.name: config.model_id for config in AVAILABLE_MODELS.values()
}

# Default model for experiments
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"


def get_model_id(model_name: str) -> str:
    """Get full model ID from short name.

    Args:
        model_name: Short model name or full model ID

    Returns:
        Full model ID
    """
    if model_name in SILICONFLOW_MODELS:
        return SILICONFLOW_MODELS[model_name]
    return model_name


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key.

    Args:
        model_key: Model key from AVAILABLE_MODELS

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If model_key is not found
    """
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available: {list(AVAILABLE_MODELS.keys())}"
        )
    config = AVAILABLE_MODELS[model_key]
    # CLIs load .env after importing src.config. Resolve explicit endpoint
    # overrides at lookup time without mutating the shared registry entry.
    for platform, key_env in PLATFORM_API_KEY_ENVS.items():
        if config.api_key_env == key_env:
            for env_name in PLATFORM_BASE_URL_ENVS[platform]:
                if env_name in os.environ:
                    return replace(config, base_url=os.environ[env_name])
    return config


def list_models() -> list:
    """List available model keys."""
    return list(AVAILABLE_MODELS.keys())
