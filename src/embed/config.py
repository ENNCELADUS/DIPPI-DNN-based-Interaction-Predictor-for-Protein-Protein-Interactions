"""Configuration helpers for the embed package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .core.types import EmbeddingConfig


_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "off"}


def _expand(path_value: str) -> Path:
    """Expand user notation and return a Path without resolving."""

    return Path(path_value).expanduser()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    raise ValueError(f"Invalid boolean for {name!r}: {raw}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Invalid integer for {name!r}: {raw}") from exc


def _env_optional(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


@dataclass(slots=True)
class PathSettings:
    """Directory locations used by embed pipelines."""

    workspace: Path
    data_root: Path
    cache_root: Path
    model_cache: Path

    @staticmethod
    def from_env() -> "PathSettings":
        base = _expand(os.getenv("EMBED_WORKSPACE", str(Path.cwd())))
        data_root = _expand(os.getenv("EMBED_DATA_ROOT", str(base / "data")))
        cache_root = _expand(os.getenv("EMBED_CACHE_ROOT", str(base / ".cache")))
        model_cache = _expand(
            os.getenv("EMBED_MODEL_CACHE", str(cache_root / "models"))
        )
        return PathSettings(
            workspace=base,
            data_root=data_root,
            cache_root=cache_root,
            model_cache=model_cache,
        )

    def as_dict(self) -> Dict[str, Path]:
        """Return a dictionary representation."""

        return {
            "workspace": self.workspace,
            "data_root": self.data_root,
            "cache_root": self.cache_root,
            "model_cache": self.model_cache,
        }


def default_config() -> EmbeddingConfig:
    """Return an EmbeddingConfig populated with environment defaults."""

    paths = PathSettings.from_env()
    base = EmbeddingConfig()

    overrides: Dict[str, Any] = {
        "workspace": paths.workspace,
        "data_root": paths.data_root,
        "cache_root": paths.cache_root,
        "model_cache_dir": paths.model_cache,
        "model_name": os.getenv("EMBED_MODEL_NAME", base.model_name),
        "model_revision": _env_optional("EMBED_MODEL_REVISION", base.model_revision),
        "use_local_model": _env_bool("EMBED_USE_LOCAL_MODEL", base.use_local_model),
        "device": os.getenv("EMBED_DEVICE", base.device),
        "batch_size": _env_int("EMBED_BATCH_SIZE", base.batch_size),
        "max_sequence_length": _env_int(
            "EMBED_MAX_SEQUENCE_LENGTH", base.max_sequence_length
        ),
        "truncate_long_sequences": _env_bool(
            "EMBED_TRUNCATE_LONG_SEQUENCES", base.truncate_long_sequences
        ),
        "timeout_seconds": _env_int("EMBED_TIMEOUT_SECONDS", base.timeout_seconds),
        "retry_attempts": _env_int("EMBED_RETRY_ATTEMPTS", base.retry_attempts),
        "forge_api_url": os.getenv("EMBED_FORGE_API_URL", base.forge_api_url),
        "forge_api_token": _env_optional("EMBED_FORGE_API_TOKEN", base.forge_api_token),
    }

    return base.with_updates(**overrides)
