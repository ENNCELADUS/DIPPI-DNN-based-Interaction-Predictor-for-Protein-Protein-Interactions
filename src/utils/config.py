"""
Config parsing utilities for DIPPI pipeline.

This module provides a minimal, progress-oriented interface for config management:
- load_config(): Load YAML and track access
- extract_keys(): Extract section by path, return flattened dict
- enforce_used_keys(): Validate all keys consumed (raises on unused)

Usage pattern in run.py:
    cfg = load_config("configs/v3.yaml")
    run_cfg = extract_keys(cfg, "run_config")
    model_params = extract_keys(cfg, "model_config.v3")
    # ... extract other sections as needed
    enforce_used_keys(cfg)  # Raise if any keys unused
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import yaml

from src.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


class TrackedConfig:
    """
    Dictionary wrapper that tracks which config keys are accessed.

    Tracks access via dot-separated paths (e.g., "model_config.v3.d_model")
    to enable strict validation that all config keys are consumed.

    Attributes:
        _data: Underlying config dictionary
        _accessed_paths: Set of dot-separated paths that were accessed
    """

    def __init__(self, data: Dict[str, Any]):
        """
        Initialize tracked config.

        Args:
            data: Configuration dictionary (typically from YAML)
        """
        self._data = data
        self._accessed_paths: Set[str] = set()

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get value by dot-separated path (e.g., "model_config.v3.d_model").

        Args:
            path: Dot-separated path to config value
            default: Default value if path not found

        Returns:
            Config value at path, or default if not found
        """
        keys = path.split(".")
        value = self._data

        # Navigate through nested dict
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        # Mark as accessed
        self._accessed_paths.add(path)
        return value

    def _extract_section(self, path: str) -> Dict[str, Any]:
        """
        Extract a section by path and return as dict.

        Args:
            path: Dot-separated path to section (e.g., "model_config.v3")

        Returns:
            Dictionary at the specified path

        Raises:
            KeyError: If path does not exist
        """
        keys = path.split(".")
        value = self._data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Config path '{path}' not found")

        if not isinstance(value, dict):
            raise ValueError(
                f"Config path '{path}' does not point to a dictionary section"
            )

        return value

    def _mark_accessed(self, path: str, data: Any) -> None:
        """
        Recursively mark all paths under a section as accessed.
        Also marks all parent paths leading to this section.

        Args:
            path: Base path for this section
            data: Data at this path (dict, list, or scalar)
        """
        # Mark all parent paths
        if path:
            parts = path.split(".")
            for i in range(1, len(parts) + 1):
                parent_path = ".".join(parts[:i])
                self._accessed_paths.add(parent_path)

        if isinstance(data, dict):
            # Recursively mark all nested keys
            for key, value in data.items():
                nested_path = f"{path}.{key}" if path else key
                self._mark_accessed(nested_path, value)
        elif isinstance(data, list):
            # Mark list items (if they're dicts)
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    list_item_path = f"{path}[{idx}]"
                    self._mark_accessed(list_item_path, item)

    def _get_all_paths(self, data: Any = None, prefix: str = "") -> Set[str]:
        """
        Get all possible paths in the config tree.

        Args:
            data: Config data (defaults to root)
            prefix: Path prefix for recursion

        Returns:
            Set of all dot-separated paths
        """
        if data is None:
            data = self._data

        paths = set()

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{prefix}.{key}" if prefix else key
                paths.add(current_path)
                # Recurse into nested structures only if value is dict or list
                if isinstance(value, (dict, list)):
                    paths.update(self._get_all_paths(value, current_path))
        elif isinstance(data, list):
            # For lists, mark the path but don't recurse deeply into items
            if prefix:
                paths.add(prefix)
        # For scalars (str, int, bool, None, etc.), don't add or recurse

        return paths

    def get_unused_keys(self) -> List[str]:
        """
        Get list of config keys that were never accessed.

        Returns:
            Sorted list of unused dot-separated paths
        """
        all_paths = self._get_all_paths()
        unused = all_paths - self._accessed_paths
        return sorted(unused)

    def __getitem__(self, key: str) -> Any:
        """Direct dictionary-style access (for compatibility)."""
        if key in self._data:
            self._accessed_paths.add(key)
            return self._data[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data


def load_config(path: Union[str, Path]) -> TrackedConfig:
    """
    Load YAML config file and wrap in tracker for validation.

    Args:
        path: Path to YAML config file

    Returns:
        TrackedConfig wrapper around parsed YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML

    Example:
        >>> cfg = load_config("configs/v3.yaml")
        >>> run_cfg = extract_keys(cfg, "run_config")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if is_main_process():
        logger.info(f"Loading config from: {path}")

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {path}: {e}")

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a YAML dictionary, got {type(data)}"
        )

    if is_main_process():
        logger.info(f"Config loaded successfully with {len(data)} top-level sections")
    return TrackedConfig(data)


def extract_keys(cfg: TrackedConfig, section: str) -> Dict[str, Any]:
    """
    Extract a config section by dot-separated path and return flattened dict.

    Marks the entire section as accessed for validation. Returns a plain dict
    (not tracked) containing only the parameters in that section.

    Args:
        cfg: TrackedConfig instance from load_config()
        section: Dot-separated path to section (e.g., "model_config.v3", "pretrain_config")

    Returns:
        Flattened dictionary of parameters in that section

    Raises:
        KeyError: If section path doesn't exist in config
        ValueError: If section path doesn't point to a dict

    Examples:
        >>> cfg = load_config("configs/v3.yaml")
        >>> model_params = extract_keys(cfg, "model_config.v3")
        >>> # Returns: {"d_model": 384, "encoder_layers": 2, ...}
        >>>
        >>> pretrain_cfg = extract_keys(cfg, "pretrain_config")
        >>> # Returns: {"epochs": 30, "batch_size": 32, ...}
    """
    try:
        section_data = cfg._extract_section(section)
    except KeyError:
        raise KeyError(f"Config section '{section}' not found in config file")
    except ValueError as e:
        raise ValueError(str(e))

    # Mark entire section as accessed
    cfg._mark_accessed(section, section_data)

    # Return a plain dict copy (flattened at this level)
    result = dict(section_data)

    if is_main_process():
        logger.debug(f"Extracted config section '{section}' with {len(result)} keys")
    return result


def enforce_used_keys(cfg: TrackedConfig, used_paths: List[str] = None) -> None:
    """
    Validate that all config keys were accessed; raise if unused keys remain.

    This is called at the end of run.py setup to catch typos or unused params.
    Optionally accepts explicit paths to mark as used (for manual tracking).

    Args:
        cfg: TrackedConfig instance from load_config()
        used_paths: Optional list of additional paths to mark as used

    Raises:
        ValueError: If any config keys were never accessed, with list of unused keys

    Example:
        >>> cfg = load_config("configs/v3.yaml")
        >>> # ... extract various sections ...
        >>> enforce_used_keys(cfg)  # Raises if any keys unused
    """
    # Mark any explicitly provided paths
    if used_paths:
        for path in used_paths:
            cfg._accessed_paths.add(path)

    unused_keys = cfg.get_unused_keys()

    if unused_keys:
        # Format error message with all unused keys
        error_msg = (
            f"Found {len(unused_keys)} unused config key(s). "
            f"These may be typos or unnecessary parameters:\n"
        )
        for key in unused_keys:
            error_msg += f"  - {key}\n"
        error_msg += (
            "\nPlease remove unused keys or verify they are accessed correctly. "
            "This strict validation prevents silent config typos."
        )

        if is_main_process():
            logger.error(error_msg)
        raise ValueError(error_msg)

    if is_main_process():
        logger.info("Config validation passed: all keys were accessed")
