"""
DIPPI Pipeline Orchestrator (run.py)

This module is the single entry point for pretrain/finetune/evaluate workflows.
It owns:
- Config parsing and validation
- Seed/device/DDP setup
- Run directory creation
- Model and dataloader instantiation
- Training/validation loop control (delegates computation to Trainer/Evaluator)
- Checkpointing, logging, early stopping decisions

It does NOT:
- Perform actual training/evaluation computation (delegates to Trainer/Evaluator)
- Write CSV rows directly (delegates to utils.logging)
- Manage optimizer/scheduler internals (delegates to Trainer/Strategy)

Invocation:
  Single GPU:   python -m src.run configs/v3.yaml
  Multi-GPU:    torchrun --standalone --nproc_per_node=N -m src.run configs/v3.yaml
"""

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config, extract_keys
from src.utils.data_io import build_loader
from src.utils.sequence_data_io import build_sequence_loader
from src.utils.device import get_device
from src.utils.distributed import (
    init_if_enabled,
    cleanup,
    get_rank,
    barrier,
)
from src.model.v1 import V1
from src.model.v2 import V2
from src.model.v3 import V3
from src.model.v4 import V4
from src.model.v5 import V5
from src.model.v6 import V6
from src.model.tuna import TUnA
from src.stages import run_pretrain, run_finetune, run_evaluation


def setup_logging(run_id: str, stage: str, model_name: str, log_dir: Path) -> None:
    """
    Configure Python's root logger for this run.

    Args:
        run_id: Unique run identifier (e.g., "20251110_143022")
        stage: "pretrain", "finetune", or "evaluate"
        model_name: Model name (e.g., "v3", "tuna")
        log_dir: Directory to write log.log file
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "log.log"

    # Basic configuration: write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # override any prior logging configuration from imported libs
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(
        f"Starting {stage} stage for model '{model_name}' with run_id '{run_id}'"
    )
    logger.info(f"Logs will be written to: {log_file}")
    logger.info("=" * 80)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: more deterministic CUDA operations (may hurt performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def generate_run_id() -> str:
    """Generate timestamp-based run ID: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def enable_perf_optimizations(device: torch.device) -> None:
    """Enable GPU-side speed optimizations that do not increase host memory."""
    if device.type != "cuda":
        return

    # Allow TF32 and autotuned kernels for faster matmuls/convs without extra CPU RAM.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")
    logging.info("Enabled TF32 matmul and cuDNN benchmarking for faster training")


def create_run_directories(
    model_name: str, stage: str, run_id: str
) -> Tuple[Path, Path]:
    """
    Create log and checkpoint directories for a run.

    Args:
        model_name: Model name (e.g., "v3")
        stage: "pretrain", "finetune", or "evaluate"
        run_id: Unique run identifier

    Returns:
        (log_dir, checkpoint_dir) paths
    """
    log_dir = Path(f"logs/{model_name}/{stage}/{run_id}")
    checkpoint_dir = Path(f"models/{model_name}/{stage}/{run_id}")

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Created directories: {log_dir}, {checkpoint_dir}")
    return log_dir, checkpoint_dir


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters with requires_grad=True."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(cfg) -> Tuple[nn.Module, int]:
    """
    Instantiate model based on config.model_config.model.

    Args:
        cfg: TrackedConfig from load_config()

    Returns:
        (model, trainable_parameter_count)

    Raises:
        ValueError: If model name is unknown
        NotImplementedError: If model class not yet implemented
    """
    # Get model name (selector field)
    model_name = cfg.get("model_config.model")
    logging.info(f"Building model: {model_name}")

    # Extract model-specific config section (only the needed params)
    model_cfg = extract_keys(cfg, "model_config")
    # Remove the 'model' selector field (not a model parameter)
    model_cfg = {k: v for k, v in model_cfg.items() if k != "model"}

    # Simple if-elif branching for model selection (MVP pattern)
    if model_name == "v1":
        model = V1(**model_cfg)
    elif model_name == "v2":
        model = V2(**model_cfg)
    elif model_name == "v3":
        model = V3(**model_cfg)
    elif model_name == "v4":
        model = V4(**model_cfg)
    elif model_name == "v5":
        model = V5(**model_cfg)
    elif model_name == "v6":
        model = V6(**model_cfg)
    elif model_name == "tuna":
        model = TUnA(**model_cfg)
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. Supported models: 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'tuna'"
        )

    return model, count_trainable_parameters(model)


def build_loaders(
    cfg,
    stage: str,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders for a stage.

    Args:
        cfg: Full parsed config
        stage: "pretrain" or "finetune"
        device: Target device for data

    Returns:
        (train_loader, val_loader)
    """
    logging.info(f"Building dataloaders for {stage} stage")

    data_cfg = cfg["data_config"]
    stage_cfg = cfg[f"{stage}_config"]
    dataloader_cfg = data_cfg.get("dataloader", {})
    sampling_cfg = data_cfg.get(stage, {}).get("sampling", {})

    model_name = cfg["model_config"]["model"]
    if model_name == "v6":
        train_loader = build_sequence_loader(
            csv_path=data_cfg[stage]["train_csv"],
            sequence_path=data_cfg["sequence_path"],
            batch_size=stage_cfg["batch_size"],
            max_len=data_cfg["max_sequence_length"],
            ddp=cfg["top_level_config"]["ddp"]["enabled"],
            shuffle=True,
            sampling_cfg=sampling_cfg,
            dataloader_cfg=dataloader_cfg,
        )

        val_loader = build_sequence_loader(
            csv_path=data_cfg[stage]["valid_csv"],
            sequence_path=data_cfg["sequence_path"],
            batch_size=stage_cfg["batch_size"],
            max_len=data_cfg["max_sequence_length"],
            ddp=cfg["top_level_config"]["ddp"]["enabled"],
            shuffle=False,
            dataloader_cfg=dataloader_cfg,
        )
    else:
        train_loader = build_loader(
            csv_path=data_cfg[stage]["train_csv"],
            embeddings_path=data_cfg["embeddings_path"],
            batch_size=stage_cfg["batch_size"],
            max_len=data_cfg["max_sequence_length"],
            dtype=data_cfg["embedding_dtype"],
            ddp=cfg["top_level_config"]["ddp"]["enabled"],
            shuffle=True,
            sampling_cfg=sampling_cfg,
            dataloader_cfg=dataloader_cfg,
        )

        val_loader = build_loader(
            csv_path=data_cfg[stage]["valid_csv"],
            embeddings_path=data_cfg["embeddings_path"],
            batch_size=stage_cfg["batch_size"],
            max_len=data_cfg["max_sequence_length"],
            dtype=data_cfg["embedding_dtype"],
            ddp=cfg["top_level_config"]["ddp"]["enabled"],
            shuffle=False,  # No shuffle for validation
            dataloader_cfg=dataloader_cfg,
        )

    return train_loader, val_loader


# Stage execution functions (run_pretrain, run_finetune, run_evaluation)
# are now in src/stages/ for better modularity.


def main(config_path: str) -> None:
    """
    Main orchestrator: parse config, setup environment, run pipeline.

    Args:
        config_path: Path to YAML config file
    """
    # ============================================================
    # 1. Load config
    # ============================================================
    cfg = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # ============================================================
    # 2. Run config setup
    # ============================================================
    run_cfg = extract_keys(cfg, "run_config")
    mode = run_cfg["mode"]
    seed = run_cfg["seed"]

    # Set seed early
    set_seed(seed)

    # Generate run IDs if null
    pretrain_run_id = run_cfg.get("pretrain_run_id") or generate_run_id()
    finetune_run_id = run_cfg.get("finetune_run_id") or generate_run_id()
    eval_run_id = run_cfg.get("eval_run_id") or generate_run_id()

    # Get model name (selector field, not a section)
    model_name = cfg.get("model_config.model")

    # Determine which stages to run based on mode (for future orchestration logic)
    _run_pretrain_stage = mode in ["pretrain_only", "full_pipeline"]
    _run_finetune_stage = mode in [
        "finetune_from_pretrain",
        "train_and_eval",
        "full_pipeline",
    ]
    _run_eval_stage = mode in ["train_and_eval", "eval_only"]

    # ============================================================
    # 3. Device & DDP setup
    # ============================================================
    # Extract top-level config
    top_level_cfg = extract_keys(cfg, "top_level_config")

    # Device selection (delegates to utils/device.py)
    device = get_device(top_level_cfg["device"])
    logging.info(f"Using device: {device}")
    enable_perf_optimizations(device)

    # DDP initialization (delegates to utils/distributed.py)
    ddp_cfg = top_level_cfg["ddp"]
    ddp_enabled = init_if_enabled(ddp_cfg, device)

    if ddp_enabled:
        rank = get_rank()
        logging.info(
            f"DDP initialized: rank {rank}, backend={ddp_cfg.get('backend', 'auto')}"
        )
    else:
        logging.info("Running in single-process mode (DDP disabled)")

    # ============================================================
    # 4. Build model
    # ============================================================
    model, trainable_parameter_count = build_model(cfg)
    model.to(device)
    logging.info(f"Model moved to device: {device}")

    # Wrap with DDP if enabled
    if ddp_enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP

        ddp_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            device_ids = [device.index] if device.index is not None else None
            if device_ids is not None:
                ddp_kwargs["device_ids"] = device_ids
                ddp_kwargs["output_device"] = device.index

        finetune_cfg = cfg.get("finetune_config", {}) or {}
        strategy_cfg = (
            finetune_cfg.get("strategy", {}) if isinstance(finetune_cfg, dict) else {}
        )
        strategy_type = (
            strategy_cfg.get("type") if isinstance(strategy_cfg, dict) else None
        )

        find_unused = ddp_cfg.get("find_unused_parameters")
        if find_unused is None:
            # Default to False; staged unfreeze uses requires_grad toggles but does not
            # drop parameters from the forward graph.
            find_unused = False

        force_find_unused = bool(ddp_cfg.get("force_find_unused_parameters", False))
        has_freeze_steps = False
        if strategy_type == "staged_unfreeze" and isinstance(strategy_cfg, dict):
            schedule = strategy_cfg.get("schedule") or []
            has_freeze_steps = any(
                isinstance(step, dict) and (step.get("freeze") or step.get("unfreeze"))
                for step in schedule
            )

        if find_unused and not has_freeze_steps and not force_find_unused:
            if get_rank() == 0:
                logging.info(
                    "Disabling find_unused_parameters for DDP because no staged_unfreeze "
                    "freeze/unfreeze steps are configured."
                )
            find_unused = False

        if not find_unused and has_freeze_steps:
            if get_rank() == 0:
                logging.info(
                    "Enabling find_unused_parameters for DDP because staged_unfreeze "
                    "may freeze parameters during finetune."
                )
            find_unused = True

        ddp_kwargs["find_unused_parameters"] = bool(find_unused)

        model = DDP(model, **ddp_kwargs)
        logging.info(
            "Model wrapped with DistributedDataParallel (find_unused_parameters=%s)",
            ddp_kwargs["find_unused_parameters"],
        )

    # ============================================================
    # 5. Mode-specific pipeline execution
    # ============================================================

    if mode == "pretrain_only":
        # Create directories for pretrain only
        log_dir, checkpoint_dir = create_run_directories(
            model_name, "pretrain", pretrain_run_id
        )
        setup_logging(pretrain_run_id, "pretrain", model_name, log_dir)
        logging.info("Trainable parameters: %s", f"{trainable_parameter_count:,}")

        # Build dataloaders
        train_loader, val_loader = build_loaders(cfg, "pretrain", device)

        # Run pretrain
        run_pretrain(
            cfg,
            model,
            train_loader,
            val_loader,
            device,
            log_dir,
            checkpoint_dir,
        )

    elif mode == "finetune_from_pretrain":
        # Require explicit checkpoint path
        checkpoint_path = run_cfg.get("load_checkpoint_path")
        if not checkpoint_path:
            raise ValueError(
                "finetune_from_pretrain requires load_checkpoint_path in config"
            )

        # Create directories for finetune only
        log_dir, checkpoint_dir = create_run_directories(
            model_name, "finetune", finetune_run_id
        )
        setup_logging(finetune_run_id, "finetune", model_name, log_dir)
        logging.info("Trainable parameters: %s", f"{trainable_parameter_count:,}")

        # Build dataloaders
        train_loader, val_loader = build_loaders(cfg, "finetune", device)

        # Run finetune
        run_finetune(
            cfg,
            model,
            train_loader,
            val_loader,
            device,
            log_dir,
            checkpoint_dir,
            load_checkpoint_path=checkpoint_path,
        )

    elif mode == "train_and_eval":
        # Finetune from scratch, then evaluate using best finetune checkpoint.
        checkpoint_path = run_cfg.get("load_checkpoint_path")
        if checkpoint_path:
            logging.info(
                "train_and_eval mode ignores load_checkpoint_path; training from scratch"
            )

        log_dir, checkpoint_dir = create_run_directories(
            model_name, "finetune", finetune_run_id
        )
        setup_logging(finetune_run_id, "finetune", model_name, log_dir)
        logging.info("Trainable parameters: %s", f"{trainable_parameter_count:,}")

        train_loader, val_loader = build_loaders(cfg, "finetune", device)
        run_finetune(
            cfg,
            model,
            train_loader,
            val_loader,
            device,
            log_dir,
            checkpoint_dir,
            load_checkpoint_path=None,
        )

        finetune_best_checkpoint = checkpoint_dir / "best_model.pth"
        logging.info(
            "train_and_eval: will load %s for evaluation",
            finetune_best_checkpoint,
        )

        log_dir_evaluate, _ = create_run_directories(
            model_name, "evaluate", eval_run_id
        )
        setup_logging(eval_run_id, "evaluate", model_name, log_dir_evaluate)

        run_evaluation(
            cfg,
            model,
            device,
            log_dir_evaluate,
            load_checkpoint_path=str(finetune_best_checkpoint),
        )

    elif mode == "full_pipeline":
        # Run pretrain first
        log_dir_pretrain, checkpoint_dir_pretrain = create_run_directories(
            model_name, "pretrain", pretrain_run_id
        )
        setup_logging(pretrain_run_id, "pretrain", model_name, log_dir_pretrain)
        logging.info("Trainable parameters: %s", f"{trainable_parameter_count:,}")

        train_loader_pretrain, val_loader_pretrain = build_loaders(
            cfg, "pretrain", device
        )
        run_pretrain(
            cfg,
            model,
            train_loader_pretrain,
            val_loader_pretrain,
            device,
            log_dir_pretrain,
            checkpoint_dir_pretrain,
        )

        # Derive checkpoint path from pretrain
        pretrain_best_checkpoint = checkpoint_dir_pretrain / "best_model.pth"
        logging.info(
            f"Full pipeline: will load {pretrain_best_checkpoint} for finetune"
        )

        # Run finetune with auto-derived checkpoint
        log_dir_finetune, checkpoint_dir_finetune = create_run_directories(
            model_name, "finetune", finetune_run_id
        )
        # Reconfigure logging for finetune stage
        setup_logging(finetune_run_id, "finetune", model_name, log_dir_finetune)

        train_loader_finetune, val_loader_finetune = build_loaders(
            cfg, "finetune", device
        )
        run_finetune(
            cfg,
            model,
            train_loader_finetune,
            val_loader_finetune,
            device,
            log_dir_finetune,
            checkpoint_dir_finetune,
            load_checkpoint_path=str(pretrain_best_checkpoint),
        )

        # Derive checkpoint path from finetune for evaluation
        finetune_best_checkpoint = checkpoint_dir_finetune / "best_model.pth"
        logging.info(
            f"Full pipeline: will load {finetune_best_checkpoint} for evaluation"
        )

        # Run evaluation with best finetuned checkpoint
        log_dir_evaluate, _ = create_run_directories(
            model_name, "evaluate", eval_run_id
        )
        # Reconfigure logging for evaluation stage
        setup_logging(eval_run_id, "evaluate", model_name, log_dir_evaluate)

        run_evaluation(
            cfg,
            model,
            device,
            log_dir_evaluate,
            load_checkpoint_path=str(finetune_best_checkpoint),
        )

    elif mode == "eval_only":
        # Require explicit checkpoint path
        checkpoint_path = run_cfg.get("load_checkpoint_path")
        if not checkpoint_path:
            raise ValueError("eval_only requires load_checkpoint_path in config")

        # Create directories for eval
        log_dir, _ = create_run_directories(model_name, "evaluate", eval_run_id)
        setup_logging(eval_run_id, "evaluate", model_name, log_dir)
        logging.info("Trainable parameters: %s", f"{trainable_parameter_count:,}")

        # Run evaluation
        run_evaluation(
            cfg,
            model,
            device,
            log_dir,
            load_checkpoint_path=checkpoint_path,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ============================================================
    # 6. Cleanup
    # ============================================================
    # Cleanup distributed if enabled
    if ddp_enabled:
        barrier()  # Ensure all processes finish before cleanup
        cleanup()
        logging.info("DDP cleaned up successfully")

    logging.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    # Use 'spawn' to avoid OOM during fork() when memory usage is high (e.g. large mmaps)
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"Warning: Could not set start method to spawn: {e}")

    # Parse config path from command line
    if len(sys.argv) < 2:
        config_path = "configs/v3.yaml"
        print(f"No config provided, using default: {config_path}")
    else:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Run the pipeline
    main(config_path)
