"""
Multi-tier model loader for NARVL SLM (Qwen2.5-0.5B-Instruct-GGUF).

4-Tier Resolution Architecture:
1. Environment Variable: os.getenv("NARVL_MODEL_DIR")
2. Local User Cache: Path.home() / ".cache" / "narvl" / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
3. HuggingFace Hub: hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct-GGUF", filename="qwen2.5-0.5b-instruct-q4_k_m.gguf", local_dir=cache_dir)
4. Air-gapped Fallback: /app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf (e.g. baked Docker image)
"""

from __future__ import annotations

import importlib.resources
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("narvl.engine.model_loader")

DEFAULT_MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"


class ModelResolutionError(FileNotFoundError):
    """Raised when the GGUF model cannot be located across all 4 resolution tiers."""


def resolve_model_path(
    model_filename: str = DEFAULT_MODEL_FILENAME,
    repo_id: str = DEFAULT_REPO_ID,
    allow_download: bool = True,
) -> Path:
    """Resolve the SLM model path using 4-tier resolution.
    
    Returns:
        Path to existing .gguf model file.
    
    Raises:
        ModelResolutionError: If model is not found in any tier.
    """
    # Tier 1: Check NARVL_MODEL_DIR environment variable
    env_dir = os.getenv("NARVL_MODEL_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if env_path.is_file() and env_path.name.endswith(".gguf") and env_path.exists():
            logger.info("Resolved model via Tier 1 (NARVL_MODEL_DIR direct file): %s", env_path)
            return env_path
        candidate_tier1 = env_path / model_filename
        if candidate_tier1.exists():
            logger.info("Resolved model via Tier 1 (NARVL_MODEL_DIR directory): %s", candidate_tier1)
            return candidate_tier1

    # Tier 2: Check local user cache directory ~/.cache/narvl/models/
    cache_dir = Path.home() / ".cache" / "narvl" / "models"
    candidate_tier2 = cache_dir / model_filename
    if candidate_tier2.exists():
        logger.info("Resolved model via Tier 2 (User Cache): %s", candidate_tier2)
        return candidate_tier2

    # Check package-level resources if bundled / configured via importlib.resources
    try:
        traversable = importlib.resources.files("narvl") / "models" / model_filename
        if hasattr(traversable, "is_file") and traversable.is_file():
            res_path = Path(str(traversable))
            if res_path.exists():
                logger.info("Resolved model via importlib.resources: %s", res_path)
                return res_path
    except Exception:
        pass

    # Tier 3: HuggingFace Hub download (with resume=True) into ~/.cache/narvl/models
    if allow_download:
        try:
            from huggingface_hub import hf_hub_download

            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Attempting Tier 3 download from HuggingFace Hub (%s / %s)...",
                repo_id,
                model_filename,
            )
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                local_dir=str(cache_dir),
                resume_download=True,
            )
            resolved = Path(downloaded_path)
            if resolved.exists():
                logger.info("Resolved model via Tier 3 (HuggingFace Hub): %s", resolved)
                return resolved
        except Exception as exc:
            logger.warning(
                "Tier 3 download skipped or failed (offline / network error): %s", exc
            )

    # Tier 4: Fallback to /app/models/ (Air-gapped container directory)
    candidate_tier4 = Path("/app/models") / model_filename
    if candidate_tier4.exists():
        logger.info("Resolved model via Tier 4 (Air-gapped /app/models): %s", candidate_tier4)
        return candidate_tier4

    # Also check current working directory / models
    cwd_candidate = Path.cwd() / "models" / model_filename
    if cwd_candidate.exists():
        logger.info("Resolved model via CWD fallback: %s", cwd_candidate)
        return cwd_candidate

    raise ModelResolutionError(
        f"Could not locate {model_filename} in any of the 4 resolution tiers:\n"
        f"  1. NARVL_MODEL_DIR ({env_dir})\n"
        f"  2. Cache dir ({candidate_tier2})\n"
        f"  3. HuggingFace Hub ({repo_id})\n"
        f"  4. Air-gapped container path ({candidate_tier4})\n"
        "To run offline, download the model and place it at ~/.cache/narvl/models/ "
        "or set NARVL_MODEL_DIR."
    )


def load_llama_model(
    model_path: Optional[str | Path] = None,
    n_threads: int = 4,
    n_ctx: int = 2048,
    mmap: bool = True,
    **kwargs: Any,
) -> Any:
    """Load local llama.cpp model instance with standard NARVL configuration.
    
    Args:
        model_path: Optional explicit model path. If None, resolves via 4-tier loader.
        n_threads: Number of CPU threads (default: 4).
        n_ctx: Context window size in tokens (default: 2048).
        mmap: Memory mapping flag (default: True).
        **kwargs: Additional parameters passed to llama_cpp.Llama.
        
    Returns:
        Llama model instance.
    """
    try:
        from llama_cpp import Llama
    except ImportError as err:
        raise ImportError(
            "llama-cpp-python is required to load SLM models. "
            "Install it via `pip install llama-cpp-python`."
        ) from err

    resolved_path = Path(model_path) if model_path else resolve_model_path()
    logger.info("Loading Llama model from %s with n_threads=%d, n_ctx=%d", resolved_path, n_threads, n_ctx)

    return Llama(
        model_path=str(resolved_path),
        n_threads=n_threads,
        n_ctx=n_ctx,
        mmap=mmap,
        verbose=kwargs.pop("verbose", False),
        **kwargs,
    )
