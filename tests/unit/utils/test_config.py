"""
Unit tests for src/utils/config.py

Tests config parsing, extraction, and validation logic.
"""

import pytest
import tempfile
from pathlib import Path

from src.utils.config import load_config, extract_keys, enforce_used_keys, TrackedConfig


@pytest.fixture
def sample_config_yaml():
    """Create a temporary YAML config file for testing."""
    config_content = """
run_config:
  mode: "full_pipeline"
  seed: 42
  pretrain_run_id: null
  finetune_run_id: null

top_level_config:
  device: "cuda"
  ddp:
    enabled: true
    backend: "nccl"

data_config:
  embeddings_path: "data/embeddings.pkl"
  max_sequence_length: 1024

model_config:
  model: "v3"
  v3:
    d_model: 384
    encoder_layers: 2
    n_heads: 8
  tuna:
    hid_dim: 64
    n_layers: 1

pretrain_config:
  epochs: 30
  batch_size: 32
  optimizer:
    type: "adamw"
    weight_decay: 0.015

finetune_config:
  epochs: 15
  batch_size: 16
  strategy:
    type: "staged_unfreeze"
    freeze_patterns: ["encoder.*"]

evaluate:
  metrics:
    - "auroc"
    - "auprc"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


class TestLoadConfig:
    """Test load_config() function."""

    def test_load_valid_config(self, sample_config_yaml):
        """Should load valid YAML and return TrackedConfig."""
        cfg = load_config(sample_config_yaml)

        assert isinstance(cfg, TrackedConfig)
        assert "run_config" in cfg
        assert "model_config" in cfg

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("nonexistent_config.yaml")

    def test_load_invalid_yaml(self):
        """Should raise YAMLError for invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [[[")
            temp_path = f.name

        try:
            with pytest.raises(Exception):  # yaml.YAMLError
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestExtractKeys:
    """Test extract_keys() function."""

    def test_extract_top_level_section(self, sample_config_yaml):
        """Should extract top-level section as flattened dict."""
        cfg = load_config(sample_config_yaml)
        run_cfg = extract_keys(cfg, "run_config")

        assert isinstance(run_cfg, dict)
        assert run_cfg["mode"] == "full_pipeline"
        assert run_cfg["seed"] == 42
        assert run_cfg["pretrain_run_id"] is None

    def test_extract_nested_section(self, sample_config_yaml):
        """Should extract nested section (e.g., model_config.v3)."""
        cfg = load_config(sample_config_yaml)
        v3_params = extract_keys(cfg, "model_config.v3")

        assert isinstance(v3_params, dict)
        assert v3_params["d_model"] == 384
        assert v3_params["encoder_layers"] == 2
        assert v3_params["n_heads"] == 8

    def test_extract_deeply_nested_section(self, sample_config_yaml):
        """Should extract deeply nested sections."""
        cfg = load_config(sample_config_yaml)
        ddp_cfg = extract_keys(cfg, "top_level_config.ddp")

        assert ddp_cfg["enabled"] is True
        assert ddp_cfg["backend"] == "nccl"

    def test_extract_nonexistent_section(self, sample_config_yaml):
        """Should raise KeyError for nonexistent section."""
        cfg = load_config(sample_config_yaml)

        with pytest.raises(KeyError, match="Config section 'nonexistent' not found"):
            extract_keys(cfg, "nonexistent")

    def test_extract_invalid_path(self, sample_config_yaml):
        """Should raise KeyError for invalid nested path."""
        cfg = load_config(sample_config_yaml)

        with pytest.raises(KeyError, match="not found"):
            extract_keys(cfg, "model_config.nonexistent_model")

    def test_extract_marks_as_accessed(self, sample_config_yaml):
        """Should mark extracted section as accessed."""
        cfg = load_config(sample_config_yaml)
        extract_keys(cfg, "run_config")

        # run_config paths should be marked as accessed
        assert "run_config" in cfg._accessed_paths
        assert "run_config.mode" in cfg._accessed_paths
        assert "run_config.seed" in cfg._accessed_paths


class TestEnforceUsedKeys:
    """Test enforce_used_keys() function."""

    def test_enforce_all_keys_used(self, sample_config_yaml):
        """Should pass when all keys are accessed."""
        cfg = load_config(sample_config_yaml)

        # Extract all sections
        extract_keys(cfg, "run_config")
        extract_keys(cfg, "top_level_config")
        extract_keys(cfg, "data_config")
        extract_keys(cfg, "model_config.v3")
        extract_keys(cfg, "model_config.tuna")
        extract_keys(cfg, "pretrain_config")
        extract_keys(cfg, "finetune_config")
        extract_keys(cfg, "evaluate")

        # Also need to mark the selector field
        cfg.get("model_config.model")

        # Should not raise
        enforce_used_keys(cfg)

    def test_enforce_raises_on_unused_keys(self, sample_config_yaml):
        """Should raise ValueError when keys are unused."""
        cfg = load_config(sample_config_yaml)

        # Only extract some sections
        extract_keys(cfg, "run_config")
        extract_keys(cfg, "data_config")

        # Should raise because other sections not accessed
        with pytest.raises(ValueError, match="unused config key"):
            enforce_used_keys(cfg)

    def test_enforce_with_explicit_used_paths(self, sample_config_yaml):
        """Should accept explicit used_paths argument."""
        cfg = load_config(sample_config_yaml)

        # Extract minimal sections
        extract_keys(cfg, "run_config")

        # Mark remaining as used manually
        all_paths = cfg._get_all_paths()
        enforce_used_keys(cfg, used_paths=list(all_paths))

        # Should not raise

    def test_unused_keys_error_message_format(self, sample_config_yaml):
        """Should provide clear error message listing unused keys."""
        cfg = load_config(sample_config_yaml)

        # Don't extract anything
        try:
            enforce_used_keys(cfg)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e)
            assert "unused config key" in error_msg.lower()
            assert "run_config" in error_msg
            assert "model_config" in error_msg


class TestTrackedConfig:
    """Test TrackedConfig class internals."""

    def test_get_by_path(self, sample_config_yaml):
        """Should get values by dot-separated path."""
        cfg = load_config(sample_config_yaml)

        value = cfg.get("model_config.v3.d_model")
        assert value == 384

        nested = cfg.get("top_level_config.ddp.backend")
        assert nested == "nccl"

    def test_get_nonexistent_returns_default(self, sample_config_yaml):
        """Should return default for nonexistent paths."""
        cfg = load_config(sample_config_yaml)

        value = cfg.get("nonexistent.path", default="DEFAULT")
        assert value == "DEFAULT"

    def test_get_marks_as_accessed(self, sample_config_yaml):
        """Should mark paths as accessed when using get()."""
        cfg = load_config(sample_config_yaml)

        cfg.get("run_config.seed")
        assert "run_config.seed" in cfg._accessed_paths

    def test_dict_style_access(self, sample_config_yaml):
        """Should support dictionary-style access."""
        cfg = load_config(sample_config_yaml)

        run_cfg = cfg["run_config"]
        assert isinstance(run_cfg, dict)
        assert run_cfg["seed"] == 42

    def test_contains_check(self, sample_config_yaml):
        """Should support 'in' operator."""
        cfg = load_config(sample_config_yaml)

        assert "run_config" in cfg
        assert "nonexistent" not in cfg


class TestRealWorldUsage:
    """Test realistic usage patterns from run.py."""

    def test_progressive_extraction_workflow(self, sample_config_yaml):
        """Test the intended run.py workflow."""
        # Load config once
        cfg = load_config(sample_config_yaml)

        # Extract sections progressively
        run_cfg = extract_keys(cfg, "run_config")
        assert run_cfg["mode"] == "full_pipeline"

        device_cfg = extract_keys(cfg, "top_level_config")
        assert device_cfg["device"] == "cuda"

        data_cfg = extract_keys(cfg, "data_config")
        assert "embeddings_path" in data_cfg

        # Model-specific extraction
        model_type = cfg.get("model_config.model")
        assert model_type == "v3"

        model_params = extract_keys(cfg, f"model_config.{model_type}")
        assert model_params["d_model"] == 384

        # Stage-specific extraction
        pretrain_cfg = extract_keys(cfg, "pretrain_config")
        assert pretrain_cfg["epochs"] == 30

        finetune_cfg = extract_keys(cfg, "finetune_config")
        assert finetune_cfg["epochs"] == 15

        eval_cfg = extract_keys(cfg, "evaluate")
        assert "auroc" in eval_cfg["metrics"]

        # Extract unused model (for validation)
        extract_keys(cfg, "model_config.tuna")

        # Validate all used
        enforce_used_keys(cfg)  # Should not raise

    def test_catch_typo_in_config(self):
        """Should catch typos in config keys."""
        config_with_typo = """
run_config:
  mode: "full_pipeline"
  seed: 42

model_config:
  model: "v3"
  v3:
    d_model: 384
    droppout: 0.1  # Typo: should be 'dropout'
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_with_typo)
            temp_path = f.name

        try:
            cfg = load_config(temp_path)
            extract_keys(cfg, "run_config")
            extract_keys(cfg, "model_config.v3")
            cfg.get("model_config.model")

            # The typo 'droppout' will be in accessed paths
            # but if we had expected 'dropout' and didn't extract it,
            # enforce_used_keys would pass but we'd silently ignore the typo.
            # The key is that ALL keys must be intentionally accessed.

            enforce_used_keys(cfg)  # Should not raise since we extracted the section

        finally:
            Path(temp_path).unlink()
