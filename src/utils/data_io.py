"""
Data loading utilities for DIPPI pipeline.

This module provides:
- ProteinPairDataset: Custom PyTorch Dataset for protein pair embeddings
- build_loader: Public API to create DataLoader instances for train/val/test
- load_ml_features: Load mean-pooled features for classical ML models

Responsibilities:
- Load protein pairs from CSV (uniprotID_A, uniprotID_B, isInteraction)
- Fetch pre-computed embeddings from pickle file
- Apply truncation/padding to max_len
- Clean CLS/EOS tokens from ESM embeddings
- Return batched tensors compatible with model forward pass

Does NOT:
- Perform data augmentation or transforms
- Handle model-specific preprocessing beyond truncation/padding
- Manage distributed training internals (delegates to DistributedSampler)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from src.utils.distributed import is_main_process


# Module-level cache to avoid reloading large embeddings file
_EMBEDDINGS_CACHE: Optional[Dict[str, Any]] = None
_EMBEDDINGS_PATH_CACHE: Optional[str] = None


class MemmapEmbeddingStore:
    """
    Memory-mapped embedding store for large npz files.

    Uses numpy memory-mapping to access embeddings on-demand without
    loading the entire file into RAM. This is critical for DDP training
    where each process would otherwise load the full embeddings file.
    """

    def __init__(self, ids_array: np.ndarray, embeddings_mmap: np.ndarray):
        """
        Initialize memory-mapped embedding store.

        Args:
            ids_array: Array of protein IDs (loaded into memory, small)
            embeddings_mmap: Memory-mapped embeddings array (stays on disk)
        """
        # Build ID -> index mapping (small memory footprint)
        self._id_to_idx: Dict[str, int] = {}
        for idx, protein_id in enumerate(ids_array):
            if isinstance(protein_id, bytes):
                protein_id = protein_id.decode("utf-8")
            elif isinstance(protein_id, np.str_):
                protein_id = str(protein_id)
            self._id_to_idx[protein_id] = idx

        # Keep reference to memory-mapped array (does NOT load into RAM)
        self._embeddings = embeddings_mmap

    def __contains__(self, protein_id: str) -> bool:
        return protein_id in self._id_to_idx

    def __getitem__(self, protein_id: str) -> Dict[str, np.ndarray]:
        if protein_id not in self._id_to_idx:
            raise KeyError(f"Protein ID '{protein_id}' not found in embeddings")
        idx = self._id_to_idx[protein_id]
        # Access embedding on-demand from memory-mapped array
        return {"embeddings": self._embeddings[idx]}

    def __len__(self) -> int:
        return len(self._id_to_idx)

    def keys(self):
        return self._id_to_idx.keys()


def _load_embeddings(embeddings_path: str) -> Dict[str, Any]:
    """
    Load embeddings from pickle or npz file with module-level caching.

    Args:
        embeddings_path: Path to embeddings file (.pkl, .pickle, or .npz)

    Returns:
        Dictionary-like object mapping protein IDs to embedding data

    Note:
        Uses module-level cache to avoid reloading the same file.
        For large .npz files with structured format (ids + embeddings arrays),
        uses memory-mapping to avoid loading the entire file into RAM.
    """
    global _EMBEDDINGS_CACHE, _EMBEDDINGS_PATH_CACHE

    # Return cached version if already loaded
    if _EMBEDDINGS_CACHE is not None and _EMBEDDINGS_PATH_CACHE == embeddings_path:
        return _EMBEDDINGS_CACHE

    # Load embeddings
    embeddings_path_obj = Path(embeddings_path)
    if not embeddings_path_obj.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if is_main_process():
        logging.info(
            f"Loading embeddings from {embeddings_path} (this may take a while...)"
        )

    suffix = embeddings_path_obj.suffix.lower()
    if suffix == ".npz":
        # First, check the format without memory mapping
        npz_probe = np.load(embeddings_path, allow_pickle=True)
        has_structured_format = (
            "ids" in npz_probe.files and "embeddings" in npz_probe.files
        )
        npz_probe.close()

        if has_structured_format:
            # Use memory-mapped loading for large structured npz files
            # This keeps the embeddings on disk and loads on-demand
            npz_data = np.load(embeddings_path, allow_pickle=True, mmap_mode="r")
            ids_array = np.array(npz_data["ids"])  # Small, load into memory
            embeddings_mmap = npz_data["embeddings"]  # Large, keep memory-mapped

            embeddings_dict = MemmapEmbeddingStore(ids_array, embeddings_mmap)

            if is_main_process():
                logging.info(
                    f"Memory-mapped structured .npz format: {len(embeddings_dict)} proteins "
                    f"(embeddings stay on disk, loaded on-demand)"
                )
        else:
            # Legacy format: load into memory (typically smaller files)
            npz_data = np.load(embeddings_path, allow_pickle=True)
            embeddings_dict = {}
            for key in npz_data.files:
                val = npz_data[key]
                if isinstance(val, dict):
                    embeddings_dict[key] = val
                else:
                    embeddings_dict[key] = {"embeddings": val}

            if is_main_process():
                logging.info(
                    f"Loaded legacy .npz format: {len(embeddings_dict)} proteins into memory"
                )
    elif suffix in (".pkl", ".pickle"):
        with open(embeddings_path, "rb") as f:
            embeddings_dict = pickle.load(f)
    else:
        raise ValueError(
            f"Unsupported embeddings format: '{suffix}'. "
            "Supported formats: .npz, .pkl, .pickle"
        )

    if is_main_process():
        logging.info(f"Loaded {len(embeddings_dict)} protein embeddings")

    # Cache for future use
    _EMBEDDINGS_CACHE = embeddings_dict
    _EMBEDDINGS_PATH_CACHE = embeddings_path

    return embeddings_dict


def _clean_tokens(
    embeddings: torch.Tensor, lengths: torch.Tensor, strip_cls_eos: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clean protein embeddings by optionally removing CLS/EOS tokens.

    Args:
        embeddings: (B, L, D) protein embeddings
        lengths: (B,) sequence lengths (without CLS/EOS if they exist)
        strip_cls_eos: Whether to remove first/last tokens

    Returns:
        cleaned_embeddings: (B, L', D) cleaned embeddings
        effective_lengths: (B,) adjusted lengths

    Example:
        # Remove CLS/EOS tokens from ESM-3 embeddings
        clean_emb, clean_len = _clean_tokens(emb, lengths, strip_cls_eos=True)
        # Keep embeddings as-is
        clean_emb, clean_len = _clean_tokens(emb, lengths, strip_cls_eos=False)
    """
    device = embeddings.device
    lengths = lengths.to(device)

    if not strip_cls_eos:
        # No cleaning - clamp lengths to actual embedding size
        effective_lengths = torch.clamp(lengths, max=embeddings.size(1))
        return embeddings, effective_lengths

    # Heuristic: if padded length includes +2 over max length, assume CLS/EOS exist
    has_cls_eos = embeddings.size(1) >= (lengths.max().item() + 2)

    if has_cls_eos and embeddings.size(1) > 2:
        # Remove first and last tokens (CLS and EOS)
        cleaned = embeddings[:, 1:-1, :]
        effective_lengths = lengths
    else:
        # Use as-is but clamp lengths to actual size
        cleaned = embeddings
        effective_lengths = torch.clamp(lengths, max=embeddings.size(1))

    return cleaned, effective_lengths


def _collate_protein_pairs(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to batch protein pair samples.

    Args:
        batch: List of dicts from ProteinPairDataset.__getitem__

    Returns:
        Batched dict with keys: emb_a, emb_b, len_a, len_b, label
    """
    # Stack embeddings: (B, max_len, D)
    emb_a = torch.stack([item["emb_a"] for item in batch], dim=0)
    emb_b = torch.stack([item["emb_b"] for item in batch], dim=0)

    # Stack lengths: (B,)
    len_a = torch.tensor([item["len_a"] for item in batch], dtype=torch.long)
    len_b = torch.tensor([item["len_b"] for item in batch], dtype=torch.long)

    # Stack labels: (B, 1) for BCE loss compatibility
    labels = torch.tensor([[item["label"]] for item in batch], dtype=torch.float32)

    return {
        "emb_a": emb_a,
        "emb_b": emb_b,
        "len_a": len_a,
        "len_b": len_b,
        "label": labels,
    }


class ProteinPairDataset(Dataset):
    """
    PyTorch Dataset for protein-protein interaction pairs.

    Loads protein pair metadata from CSV and fetches pre-computed embeddings
    from a pickle file. Applies truncation/padding to max_len.

    CSV format:
        uniprotID_A,uniprotID_B,isInteraction

    Embeddings dict format:
        {protein_id: {"embeddings": np.ndarray[1, L, D], ...}, ...}
    """

    def __init__(
        self,
        csv_path: str,
        embeddings_dict: Dict[str, Any],
        max_len: int,
        dtype: str,
        strip_cls_eos: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV with protein pairs
            embeddings_dict: Pre-loaded embeddings dictionary
            max_len: Maximum sequence length (truncate/pad to this)
            dtype: Embedding dtype string ("bf16", "fp32", "fp16")
            strip_cls_eos: Whether to remove CLS/EOS tokens from embeddings
        """
        super().__init__()

        self.embeddings_dict = embeddings_dict
        self.max_len = max_len
        self.strip_cls_eos = strip_cls_eos

        # Map dtype string to torch dtype
        self.dtype_map = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
        }
        if dtype not in self.dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be one of {list(self.dtype_map.keys())}"
            )
        self.torch_dtype = self.dtype_map[dtype]

        # Load CSV
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Validate required columns
        required_cols = ["uniprotID_A", "uniprotID_B", "isInteraction"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        # Validate that all protein IDs in CSV exist in embeddings
        all_protein_ids = set(self.df["uniprotID_A"].unique()) | set(
            self.df["uniprotID_B"].unique()
        )
        missing_ids = all_protein_ids - set(self.embeddings_dict.keys())
        if missing_ids:
            sample_missing = list(missing_ids)[:10]
            if is_main_process():
                logging.warning(
                    f"Found {len(missing_ids)} protein IDs in CSV that are missing from embeddings. "
                    f"Sample missing IDs: {sample_missing}. "
                    f"These will cause KeyError during data loading. "
                    f"Consider filtering the CSV or updating embeddings."
                )
            # Optionally raise error instead of warning
            # raise ValueError(
            #     f"Missing embeddings for {len(missing_ids)} proteins. "
            #     f"Sample: {sample_missing}"
            # )

        # Log dataset stats
        n_samples = len(self.df)
        n_positive = int(self.df["isInteraction"].sum())
        if is_main_process():
            logging.info(
                f"Dataset: {n_samples} pairs ({n_positive} pos, {n_samples - n_positive} neg)"
            )

    def __len__(self) -> int:
        """Return number of protein pairs."""
        return len(self.df)

    def _process_embedding(self, protein_id: str) -> tuple[torch.Tensor, int]:
        """
        Load and process single protein embedding.

        Args:
            protein_id: UniProt ID

        Returns:
            embedding: Tensor[max_len, D] (truncated/padded)
            actual_length: Original sequence length (before padding)

        Raises:
            KeyError: If protein_id is not found in embeddings_dict
        """
        # Fetch embedding from dict with better error message
        if protein_id not in self.embeddings_dict:
            available_keys = list(self.embeddings_dict.keys())[:10]
            raise KeyError(
                f"Protein ID '{protein_id}' not found in embeddings. "
                f"Total embeddings: {len(self.embeddings_dict)}. "
                f"Sample keys: {available_keys}. "
                f"Check that your CSV protein IDs match the embedding keys."
            )

        protein_data = self.embeddings_dict[protein_id]

        # Handle both nested dict format {"embeddings": ...} and direct array format
        if isinstance(protein_data, dict) and "embeddings" in protein_data:
            embedding = protein_data["embeddings"]  # Shape: (1, L, D) or (L, D)
        elif isinstance(protein_data, (np.ndarray, torch.Tensor)):
            # Direct array format
            embedding = protein_data
        else:
            raise ValueError(
                f"Unexpected embedding format for protein '{protein_id}'. "
                f"Expected dict with 'embeddings' key or array, got {type(protein_data)}"
            )

        # Convert to torch and ensure 3D: (1, L, D)
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)

        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)  # (L, D) -> (1, L, D)

        # Centralized CLS/EOS handling via _clean_tokens
        length_before = embedding.size(1)
        lengths_tensor = torch.tensor(
            [
                length_before - 2
                if (self.strip_cls_eos and length_before > 2)
                else length_before
            ],
            dtype=torch.long,
            device=embedding.device,
        )
        cleaned, effective_lengths = _clean_tokens(
            embedding, lengths_tensor, strip_cls_eos=self.strip_cls_eos
        )
        embedding = cleaned

        # Squeeze batch dimension: (1, L, D) -> (L, D)
        embedding = embedding.squeeze(0)
        actual_length = int(effective_lengths.item())

        # Truncate if necessary
        if actual_length > self.max_len:
            embedding = embedding[: self.max_len, :]
            actual_length = self.max_len

        # Pad if necessary
        if actual_length < self.max_len:
            padding = torch.zeros(
                self.max_len - actual_length,
                embedding.size(1),
                dtype=embedding.dtype,
            )
            embedding = torch.cat([embedding, padding], dim=0)

        # Convert to target dtype
        embedding = embedding.to(dtype=self.torch_dtype)

        return embedding, actual_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single protein pair sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - emb_a: Tensor[max_len, D]
                - emb_b: Tensor[max_len, D]
                - len_a: int (actual length before padding)
                - len_b: int (actual length before padding)
                - label: float (0.0 or 1.0)
        """
        row = self.df.iloc[idx]

        protein_a = row["uniprotID_A"]
        protein_b = row["uniprotID_B"]
        label = float(row["isInteraction"])

        # Process both proteins
        emb_a, len_a = self._process_embedding(protein_a)
        emb_b, len_b = self._process_embedding(protein_b)

        return {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": len_a,
            "len_b": len_b,
            "label": label,
        }


def build_loader(
    csv_path: str,
    embeddings_path: str,
    batch_size: int,
    max_len: int,
    dtype: str,
    ddp: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    strip_cls_eos: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for protein pair data.

    This is the main public API called by run.py to create train/val/test loaders.

    Args:
        csv_path: Path to CSV with protein pairs
        embeddings_path: Path to pickle file with embeddings
        batch_size: Batch size
        max_len: Maximum sequence length (truncate/pad to this)
        dtype: Embedding dtype string ("bf16", "fp32", "fp16")
        ddp: Whether DDP is enabled (will use DistributedSampler)
        shuffle: Whether to shuffle data (ignored if ddp=True; sampler handles it)
        num_workers: Number of data loading workers (0 for MVP)
        pin_memory: Whether to pin memory for faster GPU transfer
        strip_cls_eos: Whether to remove CLS/EOS tokens from embeddings

    Returns:
        DataLoader instance

    Example:
        train_loader = build_loader(
            csv_path="data/splits/train.csv",
            embeddings_path="data/embeddings.pkl",
            batch_size=32,
            max_len=1024,
            dtype="bf16",
            ddp=False,
            shuffle=True,
        )
    """
    # Load embeddings (cached if already loaded)
    embeddings_dict = _load_embeddings(embeddings_path)

    # Create dataset
    dataset = ProteinPairDataset(
        csv_path=csv_path,
        embeddings_dict=embeddings_dict,
        max_len=max_len,
        dtype=dtype,
        strip_cls_eos=strip_cls_eos,
    )

    # Configure sampler for DDP
    sampler = None
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_protein_pairs,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,  # Keep all samples for validation/test
    )

    return loader


# =============================================================================
# ML Feature Loading (for classical ML models like RandomForest, XGBoost)
# =============================================================================

# Module-level cache for ML embeddings (separate from DL embeddings)
_ML_EMBEDDINGS_CACHE: Optional[Dict[str, Any]] = None
_ML_EMBEDDINGS_PATH_CACHE: Optional[str] = None


def _load_ml_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """
    Load mean-pooled embeddings from .pt or .pkl file with caching.

    Expected format:
        - .pt: Dict[protein_id, tensor of shape (1, seq_len, embed_dim) or (seq_len, embed_dim)]
        - .pkl: Dict[protein_id, dict with 'embeddings' key containing ndarray]

    Args:
        embeddings_path: Path to embeddings file (.pt or .pkl).

    Returns:
        Dictionary mapping protein IDs to 1D embedding vectors.
    """
    global _ML_EMBEDDINGS_CACHE, _ML_EMBEDDINGS_PATH_CACHE

    if (
        _ML_EMBEDDINGS_CACHE is not None
        and _ML_EMBEDDINGS_PATH_CACHE == embeddings_path
    ):
        return _ML_EMBEDDINGS_CACHE

    embeddings_path_obj = Path(embeddings_path)
    if not embeddings_path_obj.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if is_main_process():
        logging.info(f"Loading ML embeddings from {embeddings_path}...")

    # Load based on file extension
    if embeddings_path.endswith(".pkl"):
        import pickle

        with open(embeddings_path, "rb") as f:
            raw_dict = pickle.load(f)
        # Handle nested dict format: {protein_id: {'embeddings': ndarray, ...}}
        embeddings_dict = {}
        for protein_id, val in raw_dict.items():
            if isinstance(val, dict) and "embeddings" in val:
                embeddings_dict[protein_id] = val["embeddings"]
            else:
                embeddings_dict[protein_id] = val
    else:
        # Load from PyTorch .pt file
        embeddings_dict = torch.load(
            embeddings_path, map_location="cpu", weights_only=False
        )

    # Convert tensors to numpy if needed
    processed: Dict[str, np.ndarray] = {}
    for protein_id, emb in embeddings_dict.items():
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()
        # Ensure 1D (mean-pooled)
        if emb.ndim == 3:
            # Shape (1, seq_len, embed_dim) -> squeeze batch -> mean over seq
            emb = emb.squeeze(0).mean(axis=0)
        elif emb.ndim == 2:
            # Shape (seq_len, embed_dim) -> mean over seq
            emb = emb.mean(axis=0)
        processed[protein_id] = emb.astype(np.float32)

    if is_main_process():
        logging.info(f"Loaded {len(processed)} protein embeddings")

    _ML_EMBEDDINGS_CACHE = processed
    _ML_EMBEDDINGS_PATH_CACHE = embeddings_path

    return processed


def load_ml_features(
    csv_path: str,
    embeddings_path: str,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels for classical ML models.

    For each protein pair (A, B), creates feature vector:
        [embedding_A, embedding_B] → (2 * embedding_dim,)

    Args:
        csv_path: Path to CSV with columns [uniprotID_A, uniprotID_B, isInteraction].
        embeddings_path: Path to .pt file with mean-pooled embeddings.
        pooling: Pooling strategy (currently only "mean" is supported).

    Returns:
        X: Feature matrix of shape (n_samples, 2 * embedding_dim).
        y: Labels of shape (n_samples,).

    Raises:
        FileNotFoundError: If CSV or embeddings file not found.
        KeyError: If a protein ID is missing from embeddings.
    """
    if pooling != "mean":
        raise ValueError(
            f"Unsupported pooling strategy: {pooling}. Only 'mean' is supported."
        )

    # Load embeddings
    embeddings = _load_ml_embeddings(embeddings_path)

    # Load CSV
    csv_path_obj = Path(csv_path)
    if not csv_path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate columns
    required_cols = ["uniprotID_A", "uniprotID_B", "isInteraction"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    n_samples = len(df)
    embedding_dim = next(iter(embeddings.values())).shape[0]
    feature_dim = 2 * embedding_dim

    if is_main_process():
        logging.info(
            f"Loading ML features: {n_samples} pairs, feature_dim={feature_dim}"
        )

    # Preallocate arrays
    X = np.zeros((n_samples, feature_dim), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    missing_proteins: set = set()

    for i, row in df.iterrows():
        protein_a = row["uniprotID_A"]
        protein_b = row["uniprotID_B"]
        label = int(row["isInteraction"])

        # Get embeddings
        if protein_a not in embeddings:
            missing_proteins.add(protein_a)
            continue
        if protein_b not in embeddings:
            missing_proteins.add(protein_b)
            continue

        emb_a = embeddings[protein_a]
        emb_b = embeddings[protein_b]

        # Concatenate: [emb_a, emb_b]
        X[i] = np.concatenate([emb_a, emb_b])
        y[i] = label

    if missing_proteins:
        logging.warning(
            f"Skipped {len(missing_proteins)} proteins not found in embeddings. "
            f"First 5: {list(missing_proteins)[:5]}"
        )
        # Filter out rows with missing proteins
        valid_mask = ~np.all(X == 0, axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        logging.info(f"After filtering: {len(y)} valid samples")

    # Log class distribution
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if is_main_process():
        logging.info(f"Class distribution: {n_pos} positive, {n_neg} negative")

    return X, y
