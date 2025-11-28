"""Unit tests for src.run module.

Focuses on Milestone 1 additions: config parameter validation and passing
to trainers in pipeline orchestration functions.
"""

import pytest
from unittest.mock import Mock, patch
import logging

from src.run import (
    bootstrap_runtime,
    mode_to_loaders,
    run_evaluation,
    run_full_pipeline,
    run_finetune_from_pretrain,
    run_resume_finetune,
    _apply_resume_state,
)


class TestBootstrapRuntime:
    """Test bootstrap_runtime function."""

    def test_bootstrap_runtime_basic(self):
        """Test basic bootstrap_runtime functionality."""
        cfg = {"top_level_config": {"device": "cuda"}, "run": {"seed": 42}}

        with patch("src.run.DeviceManager") as mock_dm_cls:
            with patch("src.run.set_seed") as mock_set_seed:
                mock_dm = Mock()
                mock_dm_cls.return_value = mock_dm

                dm, logger = bootstrap_runtime(cfg)

                # Verify DeviceManager creation
                mock_dm_cls.assert_called_once_with(prefer_gpu=True, use_ddp=False)
                assert dm is mock_dm

                # Verify seeding
                mock_set_seed.assert_called_once_with(
                    42, deterministic=True, logger=logger
                )

                # Verify logger type
        assert isinstance(logger, logging.Logger)
        assert logger.name == "run_orchestrator"

    def test_bootstrap_runtime_cpu_device(self):
        """Test bootstrap_runtime with CPU device."""
        cfg = {"top_level_config": {"device": "cpu"}, "run": {"seed": 123}}

        with patch("src.run.DeviceManager") as mock_dm_cls:
            with patch("src.run.set_seed"):
                bootstrap_runtime(cfg)

                # Should set prefer_gpu=False for cpu
        mock_dm_cls.assert_called_once_with(prefer_gpu=False, use_ddp=False)


class TestBootstrapRuntimeDDP:
    def test_bootstrap_runtime_ddp_hard_error_when_worldsize_one(self):
        cfg = {
            "top_level_config": {"device": "cuda", "ddp": {"enabled": True}},
            "run": {"seed": 42},
        }

        with patch("src.run.validate_ddp_config") as _:
            with patch("src.run.init_process_group") as __:
                with patch("src.run.get_world_size", return_value=1):
                    with pytest.raises(RuntimeError):
                        bootstrap_runtime(cfg)

    def test_bootstrap_runtime_ddp_enabled_worldsize_two_sets_use_ddp_true(self):
        cfg = {
            "top_level_config": {"device": "cuda", "ddp": {"enabled": True}},
            "run": {"seed": 42},
        }

        with patch("src.run.DeviceManager") as mock_dm_cls:
            with patch("src.run.set_seed"):
                with patch("src.run.validate_ddp_config"):
                    with patch("src.run.init_process_group"):
                        with patch("src.run.get_world_size", return_value=2):
                            bootstrap_runtime(cfg)
                            mock_dm_cls.assert_called_once()
                            # Expect use_ddp=True passed when world_size>1
                            kwargs = mock_dm_cls.call_args.kwargs
                            assert kwargs.get("use_ddp") is True


class TestModeToLoaders:
    """Test mode_to_loaders function."""

    def test_mode_to_loaders_full_pipeline(self):
        """Test full_pipeline mode loader requirements."""
        loaders = mode_to_loaders("full_pipeline")
        expected = {
            "pretrain_train",
            "pretrain_val",
            "finetune_train",
            "finetune_val",
            "eval",
        }
        assert loaders == expected

    def test_mode_to_loaders_finetune_from_pretrain(self):
        """Test finetune_from_pretrain mode loader requirements."""
        loaders = mode_to_loaders("finetune_from_pretrain")
        expected = {"finetune_train", "finetune_val", "eval"}
        assert loaders == expected

    def test_mode_to_loaders_resume_finetune(self):
        """Test resume_finetune mode loader requirements."""
        loaders = mode_to_loaders("resume_finetune")
        expected = {"finetune_train", "finetune_val", "eval"}
        assert loaders == expected

    def test_mode_to_loaders_resume_pretrain(self):
        """Test resume_pretrain mode loader requirements."""
        loaders = mode_to_loaders("resume_pretrain")
        expected = {"pretrain_train", "pretrain_val"}
        assert loaders == expected

    def test_mode_to_loaders_eval_only(self):
        """Test eval_only mode loader requirements."""
        loaders = mode_to_loaders("eval_only")
        expected = {"eval"}
        assert loaders == expected


# ====================== MILESTONE 1 TESTS ======================


class TestRunPipelineMilestone1:
    """Test Milestone 1 features in run pipeline functions."""

    @patch("src.run.build_dataloaders")
    @patch("src.run.build_emb_index")
    @patch("src.run.ModelFactory.from_config")
    @patch("src.run.bootstrap_runtime")
    @patch("src.run.v3Pretrainer.from_config")
    @patch("src.run.FinetuneStrategy.from_config")
    @patch("src.run.v3Finetuner")
    @patch("src.run.run_evaluation")
    @patch("src.run.resolve_run_id")
    @patch("src.run.get_stage_logger")
    @patch("src.run.resolve_checkpoint")
    @patch("src.run.load_checkpoint")
    def test_run_full_pipeline_validates_m1_config(
        self,
        mock_load_checkpoint,
        mock_resolve_checkpoint,
        mock_get_logger,
        mock_resolve_run_id,
        mock_run_eval,
        mock_finetuner_cls,
        mock_finetune_strategy,
        mock_pretrainer,
        mock_bootstrap,
        mock_model_factory,
        mock_emb_index,
        mock_dataloaders,
    ):
        """Test that run_full_pipeline validates M1 parameters."""
        # Setup mocks
        mock_bootstrap.return_value = (Mock(), Mock())
        mock_model_factory.return_value = Mock()
        mock_emb_index.return_value = {}
        mock_dataloaders.return_value = {
            "pretrain_train": Mock(),
            "pretrain_val": Mock(),
            "finetune_train": Mock(),
            "finetune_val": Mock(),
            "eval": Mock(),
        }
        mock_resolve_run_id.side_effect = ["pretrain_123", "finetune_456"]
        mock_get_logger.side_effect = [Mock(), Mock()]
        mock_pretrainer_instance = Mock()
        mock_pretrainer.return_value = mock_pretrainer_instance
        mock_finetune_strategy.return_value = Mock()
        mock_resolve_checkpoint.return_value = "/fake/path"

        # Config with INVALID M1 parameters
        cfg = {
            "top_level_config": {"device": "cpu"},
            "run": {"seed": 42},
            "data_config": {"embeddings_path": "/fake/path"},
            "pretrain_config": {
                "epochs": 10,
                "learning_rate": 0.001,
                "log_every_n_steps": 0,  # INVALID: must be positive
            },
            "finetune_config": {
                "epochs": 5,
                "strategy_block": {},
                "validation_frequency": 1,  # Valid
            },
            "evaluate": {"metrics": ["accuracy@0.5"]},
        }

        # Should raise validation error during pretrain config validation
        with pytest.raises(
            ValueError, match="log_every_n_steps must be a positive integer"
        ):
            run_full_pipeline(cfg)

    @patch("src.run.build_dataloaders")
    @patch("src.run.build_emb_index")
    @patch("src.run.ModelFactory.from_config")
    @patch("src.run.bootstrap_runtime")
    @patch("src.run.v3Pretrainer.from_config")
    @patch("src.run.FinetuneStrategy.from_config")
    @patch("src.run.v3Finetuner")
    @patch("src.run.validate_training_config")
    @patch("src.run.run_evaluation")
    @patch("src.run.resolve_run_id")
    @patch("src.run.get_stage_logger")
    @patch("src.run.resolve_checkpoint")
    @patch("src.run.load_checkpoint")
    def test_run_full_pipeline_calls_validate_training_config(
        self,
        mock_load_checkpoint,
        mock_resolve_checkpoint,
        mock_get_logger,
        mock_resolve_run_id,
        mock_run_eval,
        mock_validate_config,
        mock_finetuner_cls,
        mock_finetune_strategy,
        mock_pretrainer,
        mock_bootstrap,
        mock_model_factory,
        mock_emb_index,
        mock_dataloaders,
    ):
        """Test that run_full_pipeline calls validate_training_config for finetune."""
        # Setup mocks
        mock_bootstrap.return_value = (Mock(), Mock())
        mock_model_factory.return_value = Mock()
        mock_emb_index.return_value = {}
        mock_dataloaders.return_value = {
            "pretrain_train": Mock(),
            "pretrain_val": Mock(),
            "finetune_train": Mock(),
            "finetune_val": Mock(),
            "eval": Mock(),
        }
        mock_resolve_run_id.side_effect = ["pretrain_123", "finetune_456"]
        mock_get_logger.side_effect = [Mock(), Mock()]

        mock_pretrainer_instance = Mock()
        mock_pretrainer.return_value = mock_pretrainer_instance

        mock_strategy = Mock()
        mock_finetune_strategy.return_value = mock_strategy

        mock_finetuner_instance = Mock()
        mock_finetuner_cls.return_value = mock_finetuner_instance

        mock_resolve_checkpoint.return_value = "/fake/checkpoint"

        cfg = {
            "top_level_config": {"device": "cpu"},
            "run": {"seed": 42},
            "data_config": {"embeddings_path": "/fake/path"},
            "pretrain_config": {"epochs": 5, "learning_rate": 0.001},
            "finetune_config": {
                "epochs": 3,
                "strategy_block": {},
                "log_every_n_steps": 20,
                "validation_frequency": 2,
            },
            "evaluate": {"metrics": ["accuracy@0.5"]},
        }

        run_full_pipeline(cfg)

        # Verify validate_training_config was called with finetune_config
        mock_validate_config.assert_called_once_with(cfg["finetune_config"])

    @patch("src.run.build_dataloaders")
    @patch("src.run.build_emb_index")
    @patch("src.run.ModelFactory.from_config")
    @patch("src.run.bootstrap_runtime")
    @patch("src.run.v3Pretrainer.from_config")
    @patch("src.run.FinetuneStrategy.from_config")
    @patch("src.run.v3Finetuner")
    @patch("src.run.validate_training_config")
    @patch("src.run.run_evaluation")
    @patch("src.run.resolve_run_id")
    @patch("src.run.get_stage_logger")
    @patch("src.run.resolve_checkpoint")
    @patch("src.run.load_checkpoint")
    def test_run_full_pipeline_passes_m1_to_finetuner(
        self,
        mock_load_checkpoint,
        mock_resolve_checkpoint,
        mock_get_logger,
        mock_resolve_run_id,
        mock_run_eval,
        mock_validate_config,
        mock_finetuner_cls,
        mock_finetune_strategy,
        mock_pretrainer,
        mock_bootstrap,
        mock_model_factory,
        mock_emb_index,
        mock_dataloaders,
    ):
        """Test that run_full_pipeline passes M1 parameters to v3Finetuner."""
        # Setup mocks
        mock_bootstrap.return_value = (Mock(), Mock())
        mock_model_factory.return_value = Mock()
        mock_emb_index.return_value = {}
        mock_dataloaders.return_value = {
            "pretrain_train": Mock(),
            "pretrain_val": Mock(),
            "finetune_train": Mock(),
            "finetune_val": Mock(),
            "eval": Mock(),
        }
        mock_resolve_run_id.side_effect = ["pretrain_123", "finetune_456"]
        mock_get_logger.side_effect = [Mock(), Mock()]

        mock_pretrainer_instance = Mock()
        mock_pretrainer.return_value = mock_pretrainer_instance

        mock_strategy = Mock()
        mock_finetune_strategy.return_value = mock_strategy

        mock_finetuner_instance = Mock()
        mock_finetuner_cls.return_value = mock_finetuner_instance

        mock_resolve_checkpoint.return_value = "/fake/checkpoint"

        cfg = {
            "top_level_config": {"device": "cpu"},
            "run": {"seed": 42},
            "data_config": {"embeddings_path": "/fake/path"},
            "pretrain_config": {"epochs": 5, "learning_rate": 0.001},
            "finetune_config": {
                "epochs": 15,
                "strategy_block": {},
                "gradient_clip_norm": 1.5,
                # M1 parameters
                "log_every_n_steps": 25,
                "validation_frequency": 3,
            },
            "evaluate": {"metrics": ["accuracy@0.5"]},
        }

        run_full_pipeline(cfg)

        # Verify v3Finetuner was called with M1 parameters
        mock_finetuner_cls.assert_called_once()
        call_args = mock_finetuner_cls.call_args

        # Check that M1 parameters were passed
        assert call_args.kwargs["epochs"] == 15
        assert call_args.kwargs["gradient_clip_norm"] == 1.5
        assert call_args.kwargs["log_every_n_steps"] == 25
        assert call_args.kwargs["validation_frequency"] == 3

    @patch("src.run.build_dataloaders")
    @patch("src.run.build_emb_index")
    @patch("src.run.ModelFactory.from_config")
    @patch("src.run.bootstrap_runtime")
    @patch("src.run.FinetuneStrategy.from_config")
    @patch("src.run.v3Finetuner")
    @patch("src.run.validate_training_config")
    @patch("src.run.run_evaluation")
    @patch("src.run.resolve_run_id")
    @patch("src.run.get_stage_logger")
    @patch("src.run.resolve_checkpoint")
    @patch("src.run.load_checkpoint")
    def test_run_finetune_from_pretrain_m1_validation(
        self,
        mock_load_checkpoint,
        mock_resolve_checkpoint,
        mock_get_logger,
        mock_resolve_run_id,
        mock_run_eval,
        mock_validate_config,
        mock_finetuner_cls,
        mock_finetune_strategy,
        mock_bootstrap,
        mock_model_factory,
        mock_emb_index,
        mock_dataloaders,
    ):
        """Test that run_finetune_from_pretrain validates M1 parameters."""
        # Setup mocks
        mock_bootstrap.return_value = (Mock(), Mock())
        mock_model_factory.return_value = Mock()
        mock_emb_index.return_value = {}
        mock_dataloaders.return_value = {
            "finetune_train": Mock(),
            "finetune_val": Mock(),
            "eval": Mock(),
        }
        mock_resolve_run_id.return_value = "finetune_456"
        mock_get_logger.return_value = Mock()

        mock_strategy = Mock()
        mock_finetune_strategy.return_value = mock_strategy

        mock_finetuner_instance = Mock()
        mock_finetuner_cls.return_value = mock_finetuner_instance

        mock_resolve_checkpoint.return_value = "/fake/checkpoint"

        cfg = {
            "top_level_config": {"device": "cpu"},
            "run": {"seed": 42},
            "data_config": {"embeddings_path": "/fake/path"},
            "finetune_config": {
                "epochs": 8,
                "strategy_block": {},
                "log_every_n_steps": 30,
                "validation_frequency": 4,
            },
            "evaluate": {"metrics": ["accuracy@0.5"]},
        }

        run_finetune_from_pretrain(cfg)

        # Verify validate_training_config was called
        mock_validate_config.assert_called_once_with(cfg["finetune_config"])

        # Verify M1 parameters were passed to v3Finetuner
        mock_finetuner_cls.assert_called_once()
        call_args = mock_finetuner_cls.call_args
        assert call_args.kwargs["log_every_n_steps"] == 30
        assert call_args.kwargs["validation_frequency"] == 4

    @patch("src.run.build_dataloaders")
    @patch("src.run.build_emb_index")
    @patch("src.run.ModelFactory.from_config")
    @patch("src.run.bootstrap_runtime")
    @patch("src.run.FinetuneStrategy.from_config")
    @patch("src.run.v3Finetuner")
    @patch("src.run.validate_training_config")
    @patch("src.run.run_evaluation")
    @patch("src.run.resolve_run_id")
    @patch("src.run.get_stage_logger")
    @patch("src.run.resolve_checkpoint")
    @patch("src.run.load_checkpoint")
    def test_run_resume_finetune_m1_validation(
        self,
        mock_load_checkpoint,
        mock_resolve_checkpoint,
        mock_get_logger,
        mock_resolve_run_id,
        mock_run_eval,
        mock_validate_config,
        mock_finetuner_cls,
        mock_finetune_strategy,
        mock_bootstrap,
        mock_model_factory,
        mock_emb_index,
        mock_dataloaders,
    ):
        """Test that run_resume_finetune validates M1 parameters."""
        # Setup mocks
        mock_bootstrap.return_value = (Mock(), Mock())
        mock_model_factory.return_value = Mock()
        mock_emb_index.return_value = {}
        mock_dataloaders.return_value = {
            "finetune_train": Mock(),
            "finetune_val": Mock(),
            "eval": Mock(),
        }
        mock_resolve_run_id.return_value = "finetune_456"
        mock_get_logger.return_value = Mock()

        mock_strategy = Mock()
        mock_finetune_strategy.return_value = mock_strategy

        mock_finetuner_instance = Mock()
        mock_finetuner_cls.return_value = mock_finetuner_instance

        mock_resolve_checkpoint.return_value = "/fake/checkpoint"

        cfg = {
            "top_level_config": {"device": "cpu"},
            "run": {"seed": 42},
            "data_config": {"embeddings_path": "/fake/path"},
            "finetune_config": {
                "epochs": 12,
                "strategy_block": {},
                "log_every_n_steps": 15,
                "validation_frequency": 2,
            },
            "evaluate": {"metrics": ["accuracy@0.5"]},
        }

        run_resume_finetune(cfg)

        # Verify validate_training_config was called
        mock_validate_config.assert_called_once_with(cfg["finetune_config"])

        # Verify M1 parameters were passed to v3Finetuner
        mock_finetuner_cls.assert_called_once()
        call_args = mock_finetuner_cls.call_args
        assert call_args.kwargs["log_every_n_steps"] == 15
        assert call_args.kwargs["validation_frequency"] == 2

    @patch("src.run.build_dataloaders")
    @patch("src.run.build_emb_index")
    @patch("src.run.ModelFactory.from_config")
    @patch("src.run.bootstrap_runtime")
    @patch("src.run.FinetuneStrategy.from_config")
    @patch("src.run.v3Finetuner")
    @patch("src.run.validate_training_config")
    def test_finetune_validation_error_propagates(
        self,
        mock_validate_config,
        mock_finetuner_cls,
        mock_finetune_strategy,
        mock_bootstrap,
        mock_model_factory,
        mock_emb_index,
        mock_dataloaders,
    ):
        """Test that M1 validation errors in finetune config propagate correctly."""
        # Make validation fail
        mock_validate_config.side_effect = ValueError(
            "validation_frequency must be a positive integer, got: 0"
        )

        # Setup basic mocks
        mock_bootstrap.return_value = (Mock(), Mock())
        mock_model_factory.return_value = Mock()
        mock_emb_index.return_value = {}
        mock_dataloaders.return_value = {
            "finetune_train": Mock(),
            "finetune_val": Mock(),
            "eval": Mock(),
        }

        cfg = {
            "top_level_config": {"device": "cpu"},
            "run": {"seed": 42},
            "data_config": {"embeddings_path": "/fake/path"},
            "finetune_config": {
                "epochs": 5,
                "strategy_block": {},
                "validation_frequency": 0,  # Invalid
            },
            "evaluate": {"metrics": ["accuracy@0.5"]},
        }

        # Should propagate the validation error
        with pytest.raises(
            ValueError, match="validation_frequency must be a positive integer"
        ):
            run_finetune_from_pretrain(cfg)

        # v3Finetuner should not have been called due to early validation failure
        mock_finetuner_cls.assert_not_called()

    def test_run_evaluation_unchanged(self):
        """Test that run_evaluation function is unchanged by M1."""
        # This is a simple regression test to ensure run_evaluation still works
        mock_model = Mock()
        mock_dataloaders = {"eval": Mock()}
        mock_device_manager = Mock()
        mock_evaluate_config = {"metrics": ["accuracy@0.5"]}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                mock_evaluator = Mock()
                mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                mock_evaluator_cls.return_value = mock_evaluator

                # Should run without errors
                run_evaluation(
                    mock_model,
                    mock_dataloaders,
                    mock_device_manager,
                    mock_evaluate_config,
                    "finetune",
                    "test_run_123",
                )

                # Verify evaluation was called
                mock_evaluator.evaluate.assert_called_once()


class TestRunDiagnosticIntegration:
    """Test diagnostic evaluation integration in run_evaluation."""

    @pytest.fixture
    def mock_setup(self):
        """Setup common mocks for diagnostic tests."""
        mock_model = Mock()
        mock_dataloaders = {"eval": Mock()}
        mock_device_manager = Mock()
        mock_evaluate_config = {"metrics": ["accuracy@0.5"]}

        return {
            "model": mock_model,
            "dataloaders": mock_dataloaders,
            "device_manager": mock_device_manager,
            "evaluate_config": mock_evaluate_config,
            "stage": "finetune",
            "run_id": "test_run_123",
        }

    def test_run_evaluation_with_save_plots_disabled(self, mock_setup):
        """Test run_evaluation when save_plots is disabled (no diagnostics)."""
        logging_config = {"save_plots": False}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                mock_evaluator = Mock()
                mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                mock_evaluator_cls.return_value = mock_evaluator

                run_evaluation(**mock_setup, logging_config=logging_config)

                # Should call standard evaluate, not diagnostics
                mock_evaluator.evaluate.assert_called_once()
                mock_evaluator.evaluate_with_diagnostics.assert_not_called()

    def test_run_evaluation_with_save_plots_enabled(self, mock_setup):
        """Test run_evaluation when save_plots is enabled (runs diagnostics)."""
        logging_config = {"save_plots": True, "max_diagnostic_samples": 8}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                with patch("torch.distributed.is_initialized", return_value=False):
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                    mock_evaluator.evaluate_with_diagnostics.return_value = {
                        "diagnostics_saved": "/fake/path",
                        "sample_count": 8,
                    }
                    mock_evaluator_cls.return_value = mock_evaluator

                    run_evaluation(**mock_setup, logging_config=logging_config)

                    # Should call both standard evaluate and diagnostics
                    mock_evaluator.evaluate.assert_called_once()
                    mock_evaluator.evaluate_with_diagnostics.assert_called_once()

                    # Check diagnostic call parameters
                    diagnostic_call = mock_evaluator.evaluate_with_diagnostics.call_args
                    assert diagnostic_call[1]["max_diagnostic_samples"] == 8
                    assert diagnostic_call[1]["save_plots"] is True

    def test_run_evaluation_diagnostic_ddp_rank_0(self, mock_setup):
        """Test diagnostic evaluation runs only on rank 0 in DDP."""
        logging_config = {"save_plots": True}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                with patch("torch.distributed.is_initialized", return_value=True):
                    with patch("torch.distributed.get_rank", return_value=0):  # Rank 0
                        mock_evaluator = Mock()
                        mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                        mock_evaluator.evaluate_with_diagnostics.return_value = {
                            "diagnostics_saved": "/fake/path",
                            "sample_count": 4,
                        }
                        mock_evaluator_cls.return_value = mock_evaluator

                        run_evaluation(**mock_setup, logging_config=logging_config)

                        # Should run diagnostics on rank 0
                        mock_evaluator.evaluate_with_diagnostics.assert_called_once()

    def test_run_evaluation_diagnostic_ddp_rank_1(self, mock_setup):
        """Test diagnostic evaluation skipped on non-rank-0 in DDP."""
        logging_config = {"save_plots": True}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                with patch("torch.distributed.is_initialized", return_value=True):
                    with patch("torch.distributed.get_rank", return_value=1):  # Rank 1
                        mock_evaluator = Mock()
                        mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                        mock_evaluator_cls.return_value = mock_evaluator

                        run_evaluation(**mock_setup, logging_config=logging_config)

                        # Should skip diagnostics on non-rank-0
                        mock_evaluator.evaluate_with_diagnostics.assert_not_called()

    def test_run_evaluation_diagnostic_uses_default_samples(self, mock_setup):
        """Test diagnostic evaluation uses default sample count when not specified."""
        logging_config = {"save_plots": True}  # No max_diagnostic_samples specified

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                with patch("torch.distributed.is_initialized", return_value=False):
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                    mock_evaluator.evaluate_with_diagnostics.return_value = {
                        "diagnostics_saved": "/fake/path",
                        "sample_count": 16,
                    }
                    mock_evaluator_cls.return_value = mock_evaluator

                    run_evaluation(**mock_setup, logging_config=logging_config)

                    # Should use default value (16)
                    diagnostic_call = mock_evaluator.evaluate_with_diagnostics.call_args
                    assert diagnostic_call[1]["max_diagnostic_samples"] == 16

    def test_run_evaluation_diagnostic_failure_graceful(self, mock_setup):
        """Test that diagnostic failure doesn't break main evaluation."""
        logging_config = {"save_plots": True}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                with patch("torch.distributed.is_initialized", return_value=False):
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                    # Make diagnostics fail
                    mock_evaluator.evaluate_with_diagnostics.side_effect = RuntimeError(
                        "Diagnostic failed"
                    )
                    mock_evaluator_cls.return_value = mock_evaluator

                    # Should not raise exception despite diagnostic failure
                    run_evaluation(**mock_setup, logging_config=logging_config)

                    # Standard evaluation should still have run
                    mock_evaluator.evaluate.assert_called_once()

    def test_run_evaluation_multiple_dataloaders_uses_first(self, mock_setup):
        """Test diagnostic evaluation uses first dataloader when multiple available."""
        # Setup multiple eval loaders
        mock_setup["dataloaders"] = {
            "eval_balanced": Mock(),
            "eval_realistic": Mock(),
        }
        logging_config = {"save_plots": True}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                with patch("torch.distributed.is_initialized", return_value=False):
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                    mock_evaluator.evaluate_with_diagnostics.return_value = {
                        "diagnostics_saved": "/fake/path",
                        "sample_count": 4,
                    }
                    mock_evaluator_cls.return_value = mock_evaluator

                    run_evaluation(**mock_setup, logging_config=logging_config)

                    # Should call diagnostics with first available loader
                    mock_evaluator.evaluate_with_diagnostics.assert_called_once()
                    diagnostic_call = mock_evaluator.evaluate_with_diagnostics.call_args
                    diagnostic_loader = diagnostic_call[1]["dataloader"]
                    # Should be one of the eval loaders
                    assert diagnostic_loader in mock_setup["dataloaders"].values()

    def test_run_evaluation_no_logging_config(self, mock_setup):
        """Test run_evaluation works when no logging_config provided."""
        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                mock_evaluator = Mock()
                mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                mock_evaluator_cls.return_value = mock_evaluator

                # Call without logging_config (should default to None)
                run_evaluation(**mock_setup, logging_config=None)

                # Should not run diagnostics
                mock_evaluator.evaluate.assert_called_once()
                mock_evaluator.evaluate_with_diagnostics.assert_not_called()

    def test_run_evaluation_diagnostic_parameters_passed_correctly(self, mock_setup):
        """Test that all diagnostic parameters are passed correctly."""
        logging_config = {"save_plots": True, "max_diagnostic_samples": 32}

        with patch("src.run.Evaluator") as mock_evaluator_cls:
            with patch("src.run.log_eval_summary"):
                with patch("torch.distributed.is_initialized", return_value=False):
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate.return_value = {"accuracy@0.5": 0.85}
                    mock_evaluator.evaluate_with_diagnostics.return_value = {
                        "diagnostics_saved": "/fake/path",
                        "sample_count": 32,
                    }
                    mock_evaluator_cls.return_value = mock_evaluator

                    run_evaluation(**mock_setup, logging_config=logging_config)

                    # Verify all parameters are passed correctly
                    diagnostic_call = mock_evaluator.evaluate_with_diagnostics.call_args
                    call_kwargs = diagnostic_call[1]

                    assert call_kwargs["model"] == mock_setup["model"]
                    assert call_kwargs["device_manager"] == mock_setup["device_manager"]
                    assert call_kwargs["model_name"] == "v3"
                    assert call_kwargs["stage"] == "finetune"
                    assert call_kwargs["run_id"] == "test_run_123"
                    assert call_kwargs["max_diagnostic_samples"] == 32
                    assert call_kwargs["save_plots"] is True


class TestResumeHelpers:
    """Tests covering resume helper behavior in run module."""

    def test_apply_resume_state_uses_inferred_global_step(self, monkeypatch):
        trainer = Mock()
        trainer.scheduler = Mock()
        trainer.train_loader = list(range(3))
        trainer.logger = Mock()
        trainer.set_resume_state = Mock()

        checkpoint_report = {"epoch": 7, "checkpoint": {"extra": {}}}

        infer_mock = Mock(return_value=(321, "scheduler.last_epoch"))
        monkeypatch.setattr("src.run.infer_resume_global_step", infer_mock)

        _apply_resume_state(trainer, checkpoint_report)

        infer_mock.assert_called_once_with(
            checkpoint_report,
            scheduler=trainer.scheduler,
            epoch_length=len(trainer.train_loader),
        )

        trainer.set_resume_state.assert_called_once_with(last_epoch=7, global_step=321)
