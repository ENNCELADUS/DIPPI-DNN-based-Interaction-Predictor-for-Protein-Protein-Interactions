"""Configuration helpers for the centralized pipeline runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

ConfigDict = dict[str, object]


def as_str(value: object, field_name: str) -> str:
    """Convert a config value to string.

    Args:
        value: Raw config value.
        field_name: Field name used in error messages.

    Returns:
        Parsed string value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be a string")


def as_int(value: object, field_name: str) -> int:
    """Convert a config value to integer.

    Args:
        value: Raw config value.
        field_name: Field name used in error messages.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be int-compatible") from error
    raise ValueError(f"{field_name} must be int-compatible")


def as_float(value: object, field_name: str) -> float:
    """Convert a config value to float.

    Args:
        value: Raw config value.
        field_name: Field name used in error messages.

    Returns:
        Parsed floating-point value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"{field_name} must be float-compatible") from error
    raise ValueError(f"{field_name} must be float-compatible")


def as_bool(value: object, field_name: str) -> bool:
    """Convert a config value to bool.

    Args:
        value: Raw config value.
        field_name: Field name used in error messages.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be bool-compatible")


def as_str_list(value: object, field_name: str) -> list[str]:
    """Convert a config value to list of strings.

    Args:
        value: Raw config value.
        field_name: Field name used in error messages.

    Returns:
        List of string values.

    Raises:
        ValueError: If the value is not a sequence.
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    raise ValueError(f"{field_name} must be a sequence of strings")


def load_config(config_path: str | Path) -> ConfigDict:
    """Load YAML config into a dictionary.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed configuration mapping.

    Raises:
        ValueError: If the config root is not a mapping.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return cast(ConfigDict, raw)


def get_section(config: ConfigDict, section_name: str) -> ConfigDict:
    """Return a required mapping section from the root config.

    Args:
        config: Root configuration mapping.
        section_name: Required section key.

    Returns:
        Section mapping.

    Raises:
        ValueError: If the section is missing or not a mapping.
    """
    value = config.get(section_name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping")
    return cast(ConfigDict, value)


def extract_model_kwargs(config: ConfigDict) -> tuple[str, ConfigDict]:
    """Extract model name and model kwargs from global config.

    Args:
        config: Root configuration mapping.

    Returns:
        Tuple of model name and model kwargs.

    Raises:
        ValueError: If the model name is invalid.
    """
    model_config = get_section(config, "model_config")
    model_name = model_config.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_config.model must be a non-empty string")
    kwargs = dict(model_config)
    kwargs.pop("model", None)
    return model_name.lower(), kwargs
