"""Embedding cache management and ESM3 embedding generation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.distributed as dist

from src.utils.config import ConfigDict, as_str, get_section

LOGGER = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
INDEX_FILENAME = "index.json"
METADATA_FILENAME = "metadata.json"
EMBEDDINGS_SUBDIR = "embeddings"
DEFAULT_ESM3_MODEL_NAME = "esm3_sm_open_v1"
DEFAULT_EMBEDDING_DEVICE = "auto"
CSV_SUFFIXES = (".csv",)
FASTA_SUFFIXES = (".fasta", ".fa", ".faa", ".fna", ".fas")
DEFAULT_ID_COLUMNS = (
    "uniprot_id",
    "uniprotid",
    "protein_id",
    "protein",
    "id",
    "accession",
)
DEFAULT_SEQUENCE_COLUMNS = ("sequence", "seq", "protein_sequence")
FASTA_UNIPROT_PATTERN = re.compile(r"\|([^|]+)\|")
EMBED_PROGRESS_LOGGER_NAME = "relic.embed.progress"


@dataclass(frozen=True)
class EmbeddingCacheManifest:
    """Resolved embedding cache details.

    Attributes:
        cache_dir: Embedding cache root directory.
        index: Mapping from protein ID to relative embedding file path.
        required_ids: IDs required by configured split files.
    """

    cache_dir: Path
    index: dict[str, str]
    required_ids: frozenset[str]


@dataclass(frozen=True)
class _EmbeddingSettings:
    """Runtime embedding settings parsed from config."""

    source: str
    cache_dir: Path
    model_name: str
    device: str
    sequence_file: Path | None
    id_column: str | None
    sequence_column: str | None


class _Esm3ModelProtocol(Protocol):
    """Protocol for minimal ESM3 model interactions."""

    def to(self, device: str) -> _Esm3ModelProtocol:
        """Move model to target device."""

    def eval(self) -> _Esm3ModelProtocol:
        """Switch model to eval mode."""

    def encode(self, protein: object) -> object:
        """Encode an ESM protein object."""

    def logits(self, protein_tensor: object, logits_config: object) -> _LogitsOutputProtocol:
        """Run logits call returning embeddings."""


class _Esm3ClassProtocol(Protocol):
    """Protocol for ESM3 class constructor."""

    @classmethod
    def from_pretrained(cls, model_name: str) -> _Esm3ModelProtocol:
        """Instantiate a pretrained ESM3 model."""


class _ProteinCtorProtocol(Protocol):
    """Protocol for ESM protein constructor."""

    def __call__(self, *, sequence: str) -> object:
        """Build protein object from sequence."""


class _LogitsConfigCtorProtocol(Protocol):
    """Protocol for logits-config constructor."""

    def __call__(self, *, sequence: bool, return_embeddings: bool) -> object:
        """Build logits config."""


class _LogitsOutputProtocol(Protocol):
    """Protocol for ESM logits output."""

    embeddings: torch.Tensor | None


def _optional_string(value: object) -> str | None:
    """Return a stripped string when provided, else ``None``."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _parse_embedding_settings(config: ConfigDict) -> _EmbeddingSettings:
    """Parse ``data_config.embeddings`` runtime settings."""
    data_cfg = get_section(config, "data_config")
    embeddings_cfg = get_section(data_cfg, "embeddings")

    source = as_str(embeddings_cfg.get("source", ""), "data_config.embeddings.source").lower()
    cache_dir_raw = _optional_string(embeddings_cfg.get("cache_dir"))
    if cache_dir_raw is None:
        raise ValueError("data_config.embeddings.cache_dir must be a non-empty path")

    model_name = _optional_string(embeddings_cfg.get("model_name")) or DEFAULT_ESM3_MODEL_NAME
    device = _optional_string(embeddings_cfg.get("device")) or DEFAULT_EMBEDDING_DEVICE
    sequence_file_raw = _optional_string(embeddings_cfg.get("sequence_file"))
    id_column = _optional_string(embeddings_cfg.get("id_column"))
    sequence_column = _optional_string(embeddings_cfg.get("sequence_column"))

    return _EmbeddingSettings(
        source=source,
        cache_dir=Path(cache_dir_raw),
        model_name=model_name,
        device=device,
        sequence_file=Path(sequence_file_raw) if sequence_file_raw is not None else None,
        id_column=id_column,
        sequence_column=sequence_column,
    )


def _collect_required_protein_ids(
    split_paths: Sequence[Path],
) -> set[str]:
    """Collect required protein IDs from configured split files."""
    required_ids: set[str] = set()
    for split_path in split_paths:
        if not split_path.exists():
            raise FileNotFoundError(f"Split dataset path not found: {split_path}")

        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [part.strip() for part in line.strip().split("\t")]
                if len(parts) < 2:
                    continue
                if not parts[0] or not parts[1]:
                    continue

                required_ids.add(parts[0])
                required_ids.add(parts[1])

    if not required_ids:
        raise ValueError("No protein IDs found in configured split files")
    return required_ids


def _resolve_sequence_search_roots(config: ConfigDict, split_paths: Sequence[Path]) -> list[Path]:
    """Build ordered sequence-discovery roots from config and split paths."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")

    candidate_roots: list[Path] = []
    seen: set[str] = set()

    def _add_if_new(path: Path | None) -> None:
        if path is None:
            return
        normalized = str(path.resolve())
        if normalized in seen:
            return
        seen.add(normalized)
        candidate_roots.append(path)

    root_dir_raw = _optional_string(benchmark_cfg.get("root_dir"))
    processed_dir_raw = _optional_string(benchmark_cfg.get("processed_dir"))
    _add_if_new(Path(processed_dir_raw) if processed_dir_raw is not None else None)
    _add_if_new(Path(root_dir_raw) if root_dir_raw is not None else None)

    for split_path in split_paths:
        _add_if_new(split_path.parent)
        _add_if_new(split_path.parent.parent)

    return [path for path in candidate_roots if path.exists()]


def _collect_candidate_files(search_roots: Sequence[Path], suffixes: Sequence[str]) -> list[Path]:
    """Collect candidate files under search roots for the specified suffixes."""
    candidates: list[Path] = []
    seen: set[str] = set()

    for root in search_roots:
        for suffix in suffixes:
            for path in root.rglob(f"*{suffix}"):
                if not path.is_file():
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(path)

    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates


def _clean_protein_sequence(sequence: str) -> str:
    """Canonicalize a protein sequence string."""
    cleaned = sequence.strip().upper()
    cleaned = cleaned.replace("-", "").replace(".", "").replace("*", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def _resolve_column_name(
    fieldnames: Sequence[str],
    override: str | None,
    candidates: Sequence[str],
    column_kind: str,
    csv_path: Path,
) -> str | None:
    """Resolve a CSV column name by override or known candidates."""
    lowercase_to_actual = {name.lower(): name for name in fieldnames}
    if override is not None:
        resolved = lowercase_to_actual.get(override.lower())
        if resolved is None:
            raise ValueError(
                f"CSV file {csv_path} missing requested {column_kind} column '{override}'"
            )
        return resolved

    for candidate in candidates:
        resolved = lowercase_to_actual.get(candidate.lower())
        if resolved is not None:
            return resolved
    return None


def _load_sequences_from_csv(
    csv_path: Path,
    required_ids: set[str],
    sequences: dict[str, str],
    id_column_override: str | None,
    sequence_column_override: str | None,
) -> None:
    """Load matching protein sequences from a CSV file into ``sequences``."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return

        id_column = _resolve_column_name(
            fieldnames=reader.fieldnames,
            override=id_column_override,
            candidates=DEFAULT_ID_COLUMNS,
            column_kind="ID",
            csv_path=csv_path,
        )
        sequence_column = _resolve_column_name(
            fieldnames=reader.fieldnames,
            override=sequence_column_override,
            candidates=DEFAULT_SEQUENCE_COLUMNS,
            column_kind="sequence",
            csv_path=csv_path,
        )
        if id_column is None or sequence_column is None:
            return

        for row in reader:
            protein_id = row.get(id_column, "").strip()
            if protein_id not in required_ids or protein_id in sequences:
                continue
            sequence_value = _clean_protein_sequence(row.get(sequence_column, ""))
            if sequence_value:
                sequences[protein_id] = sequence_value
            if len(sequences) >= len(required_ids):
                return


def _extract_protein_id_from_header(header: str) -> str:
    """Extract protein ID from FASTA header."""
    match = FASTA_UNIPROT_PATTERN.search(header)
    if match is not None:
        return match.group(1).strip()
    token = header.split(maxsplit=1)[0]
    return token.strip()


def _load_sequences_from_fasta(
    fasta_path: Path,
    required_ids: set[str],
    sequences: dict[str, str],
) -> None:
    """Load matching protein sequences from a FASTA file into ``sequences``."""
    current_id: str | None = None
    current_chunks: list[str] = []

    def _flush_current() -> None:
        nonlocal current_id
        nonlocal current_chunks
        if current_id is None or current_id in sequences or current_id not in required_ids:
            current_id = None
            current_chunks = []
            return
        sequence_value = _clean_protein_sequence("".join(current_chunks))
        if sequence_value:
            sequences[current_id] = sequence_value
        current_id = None
        current_chunks = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                _flush_current()
                current_id = _extract_protein_id_from_header(stripped[1:])
                continue
            if current_id is not None:
                current_chunks.append(stripped)
        _flush_current()


def _discover_sequences(
    required_ids: set[str],
    search_roots: Sequence[Path],
    explicit_sequence_file: Path | None,
    id_column_override: str | None,
    sequence_column_override: str | None,
) -> dict[str, str]:
    """Discover protein sequences using CSV-first then FASTA fallback."""
    sequences: dict[str, str] = {}

    def _load_from_path(path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in CSV_SUFFIXES:
            _load_sequences_from_csv(
                csv_path=path,
                required_ids=required_ids,
                sequences=sequences,
                id_column_override=id_column_override,
                sequence_column_override=sequence_column_override,
            )
            return
        if suffix in FASTA_SUFFIXES:
            _load_sequences_from_fasta(
                fasta_path=path,
                required_ids=required_ids,
                sequences=sequences,
            )
            return
        raise ValueError(f"Unsupported sequence_file extension: {path}")

    if explicit_sequence_file is not None:
        if not explicit_sequence_file.exists():
            raise FileNotFoundError(f"Configured sequence file not found: {explicit_sequence_file}")
        _load_from_path(explicit_sequence_file)

    if len(sequences) < len(required_ids):
        csv_files = _collect_candidate_files(search_roots=search_roots, suffixes=CSV_SUFFIXES)
        for csv_path in csv_files:
            if (
                explicit_sequence_file is not None
                and csv_path.resolve() == explicit_sequence_file.resolve()
            ):
                continue
            _load_sequences_from_csv(
                csv_path=csv_path,
                required_ids=required_ids,
                sequences=sequences,
                id_column_override=id_column_override,
                sequence_column_override=sequence_column_override,
            )
            if len(sequences) >= len(required_ids):
                break

    if len(sequences) < len(required_ids):
        fasta_files = _collect_candidate_files(search_roots=search_roots, suffixes=FASTA_SUFFIXES)
        for fasta_path in fasta_files:
            if (
                explicit_sequence_file is not None
                and fasta_path.resolve() == explicit_sequence_file.resolve()
            ):
                continue
            _load_sequences_from_fasta(
                fasta_path=fasta_path,
                required_ids=required_ids,
                sequences=sequences,
            )
            if len(sequences) >= len(required_ids):
                break

    missing_ids = sorted(required_ids - set(sequences))
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise FileNotFoundError(
            "Unable to discover sequences for required proteins: "
            f"{preview} (missing={len(missing_ids)})"
        )
    return sequences


def _load_json_mapping(path: Path, description: str) -> dict[str, object]:
    """Load a JSON object mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} at {path} must be a JSON object")
    return cast(dict[str, object], payload)


def _load_index(index_path: Path) -> dict[str, str]:
    """Load and validate embedding index file."""
    if not index_path.exists():
        return {}
    raw_index = _load_json_mapping(index_path, "Embedding index")
    index: dict[str, str] = {}
    for protein_id, rel_path in raw_index.items():
        if not isinstance(protein_id, str) or not protein_id:
            raise ValueError(f"Invalid protein ID key in index: {protein_id!r}")
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError(f"Invalid embedding path for protein '{protein_id}' in index")
        index[protein_id] = rel_path
    return index


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    """Load metadata mapping when present."""
    if not metadata_path.exists():
        return {}
    return _load_json_mapping(metadata_path, "Embedding metadata")


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(path)


def _embedding_relative_path(protein_id: str) -> str:
    """Return deterministic relative cache path for one protein ID."""
    digest = hashlib.sha256(protein_id.encode("utf-8")).hexdigest()
    return f"{EMBEDDINGS_SUBDIR}/{digest}.pt"


def _resolve_embedding_path(cache_dir: Path, relative_path: str) -> Path:
    """Resolve and validate embedding path under cache root."""
    rel_path = Path(relative_path)
    if rel_path.is_absolute():
        raise ValueError(
            f"Embedding index path must be relative, got absolute path: {relative_path}"
        )

    cache_root = cache_dir.resolve()
    absolute_path = (cache_dir / rel_path).resolve()
    if absolute_path != cache_root and cache_root not in absolute_path.parents:
        raise ValueError(f"Embedding index path escapes cache directory: {relative_path}")
    return absolute_path


def _save_tensor_atomic(path: Path, tensor: torch.Tensor) -> None:
    """Atomically save an embedding tensor to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    torch.save(tensor, temp_path)
    temp_path.replace(path)


def load_cached_embedding(
    cache_dir: Path,
    index: Mapping[str, str],
    protein_id: str,
    expected_input_dim: int | None = None,
    max_sequence_length: int | None = None,
) -> torch.Tensor:
    """Load one embedding tensor from cache with schema validation.

    Args:
        cache_dir: Embedding cache root directory.
        index: Protein-to-relative-path mapping.
        protein_id: Protein identifier to load.
        expected_input_dim: Optional expected embedding width.
        max_sequence_length: Optional maximum allowed sequence length.

    Returns:
        Tensor of shape ``(seq_len, input_dim)`` on CPU.

    Raises:
        FileNotFoundError: If index entry or file path is missing.
        ValueError: If tensor shape/type validation fails.
    """
    relative_path = index.get(protein_id)
    if relative_path is None:
        raise FileNotFoundError(f"Protein '{protein_id}' is missing from embedding index")

    embedding_path = _resolve_embedding_path(cache_dir=cache_dir, relative_path=relative_path)
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding file missing for protein '{protein_id}': {embedding_path}"
        )

    tensor_object: object = torch.load(embedding_path, map_location="cpu")
    if not isinstance(tensor_object, torch.Tensor):
        raise ValueError(f"Embedding file for protein '{protein_id}' does not contain a tensor")

    tensor = tensor_object.detach().to(dtype=torch.float32)
    if tensor.dim() == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(
            f"Embedding for protein '{protein_id}' must be 2D, got shape {tuple(tensor.shape)}"
        )
    if tensor.size(0) <= 0:
        raise ValueError(f"Embedding for protein '{protein_id}' has empty sequence length")
    if expected_input_dim is not None and tensor.size(1) != expected_input_dim:
        raise ValueError(
            f"Embedding dim mismatch for protein '{protein_id}': "
            f"expected {expected_input_dim}, got {tensor.size(1)}"
        )
    if max_sequence_length is not None and tensor.size(0) > max_sequence_length:
        raise ValueError(
            f"Embedding length exceeds max_sequence_length for protein '{protein_id}': "
            f"{tensor.size(0)} > {max_sequence_length}"
        )

    return tensor.contiguous()


def _find_missing_or_invalid_ids(
    required_ids: set[str],
    cache_dir: Path,
    index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
) -> tuple[set[str], dict[str, str]]:
    """Return missing and invalid IDs after cache validation."""
    missing_ids: set[str] = set()
    invalid_ids: dict[str, str] = {}

    for protein_id in sorted(required_ids):
        if protein_id not in index:
            missing_ids.add(protein_id)
            continue
        try:
            load_cached_embedding(
                cache_dir=cache_dir,
                index=index,
                protein_id=protein_id,
                expected_input_dim=input_dim,
                max_sequence_length=max_sequence_length,
            )
        except FileNotFoundError:
            missing_ids.add(protein_id)
        except (ValueError, RuntimeError, TypeError) as error:
            invalid_ids[protein_id] = str(error)

    return missing_ids, invalid_ids


def _distributed_generation_context(allow_generation: bool) -> tuple[int, int] | None:
    """Return ``(rank, world_size)`` when distributed generation should be used."""
    if not allow_generation:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    if world_size <= 1:
        return None
    return dist.get_rank(), world_size


def _get_embed_progress_logger() -> logging.Logger:
    """Return dedicated logger for distributed embedding progress."""
    logger = logging.getLogger(EMBED_PROGRESS_LOGGER_NAME)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _log_embedding_progress(message: str, *args: object) -> None:
    """Log embedding progress from all ranks."""
    if dist.is_available() and dist.is_initialized():
        progress_logger = _get_embed_progress_logger()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        progress_logger.info("[embed][rank=%d/%d] " + message, rank, world_size, *args)
        return
    LOGGER.info(message, *args)


def _parse_str_dict(payload: object, name: str) -> dict[str, str]:
    """Parse an object payload as a string-to-string mapping."""
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a dictionary")
    parsed: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{name} must contain string keys and values")
        parsed[key] = value
    return parsed


def _parse_str_list(payload: object, name: str) -> list[str]:
    """Parse an object payload as a list of strings."""
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    parsed: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError(f"{name} must contain strings")
        parsed.append(item)
    return parsed


def _shard_ids_for_rank(
    sorted_ids: Sequence[str],
    rank: int,
    world_size: int,
) -> set[str]:
    """Return rank-local shard of sorted protein IDs."""
    shard: set[str] = set()
    for idx, protein_id in enumerate(sorted_ids):
        if idx % world_size == rank:
            shard.add(protein_id)
    return shard


def _expected_metadata(
    source: str,
    model_name: str,
    input_dim: int,
    max_sequence_length: int,
) -> dict[str, object]:
    """Build expected metadata payload for this run configuration."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": source,
        "model_name": model_name,
        "input_dim": input_dim,
        "max_sequence_length": max_sequence_length,
        "format": "torch_pt_per_protein",
    }


def _metadata_matches(current: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    """Return whether current metadata matches expected runtime settings."""
    if not current:
        return False
    return all(current.get(key) == value for key, value in expected.items())


def _resolve_embedding_device(requested_device: str) -> str:
    """Resolve embedding runtime device from config setting."""
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _load_esm3_runtime(
    model_name: str,
    requested_device: str,
) -> tuple[_Esm3ModelProtocol, _ProteinCtorProtocol, _LogitsConfigCtorProtocol, str]:
    """Load ESM3 runtime objects lazily."""
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as error:
        raise RuntimeError("ESM3 runtime is unavailable. Install the 'esm' dependency.") from error

    device = _resolve_embedding_device(requested_device)
    esm3_class = cast(_Esm3ClassProtocol, ESM3)
    protein_ctor = cast(_ProteinCtorProtocol, ESMProtein)
    logits_ctor = cast(_LogitsConfigCtorProtocol, LogitsConfig)
    model = esm3_class.from_pretrained(model_name).to(device)
    model.eval()
    return model, protein_ctor, logits_ctor, device


def _embed_sequence_with_esm3(
    model: _Esm3ModelProtocol,
    protein_ctor: _ProteinCtorProtocol,
    logits_ctor: _LogitsConfigCtorProtocol,
    sequence: str,
) -> torch.Tensor:
    """Embed one protein sequence using ESM3."""
    protein = protein_ctor(sequence=sequence)
    logits_config = logits_ctor(sequence=True, return_embeddings=True)
    with torch.no_grad():
        protein_tensor = model.encode(protein)
        sequence_output = model.logits(protein_tensor, logits_config)
    embeddings = sequence_output.embeddings
    if embeddings is None:
        raise ValueError("ESM3 did not return embeddings")

    embedding_tensor = embeddings.detach().to(dtype=torch.float32).cpu()
    if embedding_tensor.dim() == 3 and embedding_tensor.size(0) == 1:
        embedding_tensor = embedding_tensor.squeeze(0)
    if embedding_tensor.dim() != 2:
        raise ValueError(
            f"ESM3 returned unexpected embedding shape: {tuple(embedding_tensor.shape)}"
        )
    if embedding_tensor.size(0) <= 0:
        raise ValueError("ESM3 returned an empty embedding sequence")
    return embedding_tensor.contiguous()


def _generate_missing_embeddings(
    missing_ids: set[str],
    sequences: Mapping[str, str],
    settings: _EmbeddingSettings,
    cache_dir: Path,
    input_dim: int,
    max_sequence_length: int,
) -> dict[str, str]:
    """Generate and persist missing embeddings for required proteins."""
    if settings.source != "esm3":
        raise ValueError(f"Unsupported embedding source: {settings.source}")

    model, protein_ctor, logits_ctor, resolved_device = _load_esm3_runtime(
        model_name=settings.model_name,
        requested_device=settings.device,
    )
    _log_embedding_progress(
        "Generating embeddings for %d proteins using model=%s device=%s",
        len(missing_ids),
        settings.model_name,
        resolved_device,
    )

    generated_index_updates: dict[str, str] = {}
    total = len(missing_ids)
    for idx, protein_id in enumerate(sorted(missing_ids), start=1):
        sequence = sequences.get(protein_id)
        if sequence is None:
            raise FileNotFoundError(f"Missing sequence for protein '{protein_id}'")

        truncated_sequence = sequence[:max_sequence_length]
        if not truncated_sequence:
            raise ValueError(f"Protein '{protein_id}' has an empty sequence after cleaning")

        embedding_tensor = _embed_sequence_with_esm3(
            model=model,
            protein_ctor=protein_ctor,
            logits_ctor=logits_ctor,
            sequence=truncated_sequence,
        )
        if embedding_tensor.size(0) > max_sequence_length:
            embedding_tensor = embedding_tensor[:max_sequence_length]
        if embedding_tensor.size(1) != input_dim:
            raise ValueError(
                f"Embedding dim mismatch for protein '{protein_id}': "
                f"expected {input_dim}, got {embedding_tensor.size(1)}"
            )

        relative_path = _embedding_relative_path(protein_id)
        output_path = cache_dir / relative_path
        _save_tensor_atomic(path=output_path, tensor=embedding_tensor)
        generated_index_updates[protein_id] = relative_path

        if idx == total or idx % 100 == 0:
            _log_embedding_progress("Embedded %d/%d proteins", idx, total)

    return generated_index_updates


def _build_missing_ids_error_message(missing_ids: set[str]) -> str:
    """Build concise missing-ID message."""
    preview = ", ".join(sorted(missing_ids)[:10])
    return f"Missing embeddings for {len(missing_ids)} proteins: {preview}"


def _build_invalid_ids_error_message(invalid_ids: Mapping[str, str]) -> str:
    """Build concise invalid-ID message."""
    sampled_items = sorted(invalid_ids.items())[:5]
    details = "; ".join(f"{protein_id}: {reason}" for protein_id, reason in sampled_items)
    return f"Invalid embeddings detected for {len(invalid_ids)} proteins ({details})"


def ensure_embeddings_ready(
    config: ConfigDict,
    split_paths: Sequence[Path],
    input_dim: int,
    max_sequence_length: int,
    allow_generation: bool = True,
) -> EmbeddingCacheManifest:
    """Ensure all required embeddings exist and are valid before training/eval.

    Args:
        config: Root runtime configuration dictionary.
        split_paths: Train/valid/test split file paths.
        input_dim: Expected embedding dimension.
        max_sequence_length: Maximum sequence length.
        allow_generation: Whether missing/invalid embeddings may be regenerated.

    Returns:
        Embedding cache manifest containing index and required IDs.

    Raises:
        FileNotFoundError: If required embeddings are unavailable.
        ValueError: If source/config or embedding tensors are invalid.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")

    settings = _parse_embedding_settings(config)
    required_ids = _collect_required_protein_ids(
        split_paths=split_paths,
    )
    distributed_context = _distributed_generation_context(allow_generation=allow_generation)
    distributed_rank: int | None = None
    distributed_world_size: int | None = None
    if distributed_context is not None:
        distributed_rank, distributed_world_size = distributed_context

    cache_dir = settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / EMBEDDINGS_SUBDIR).mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / INDEX_FILENAME
    metadata_path = cache_dir / METADATA_FILENAME

    index = _load_index(index_path=index_path)
    metadata = _load_metadata(metadata_path=metadata_path)
    expected_metadata = _expected_metadata(
        source=settings.source,
        model_name=settings.model_name,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )
    metadata_matches = _metadata_matches(current=metadata, expected=expected_metadata)

    if metadata and not metadata_matches:
        LOGGER.warning(
            "Embedding cache metadata mismatch at %s; required IDs will be re-embedded",
            metadata_path,
        )

    if metadata_matches:
        missing_ids, invalid_ids = _find_missing_or_invalid_ids(
            required_ids=required_ids,
            cache_dir=cache_dir,
            index=index,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        )
        ids_to_generate = set(missing_ids) | set(invalid_ids)
    else:
        missing_ids = set(required_ids)
        invalid_ids = {}
        ids_to_generate = set(required_ids)

    if distributed_context is not None:
        if distributed_rank is None or distributed_world_size is None:
            raise RuntimeError("Distributed generation context must include rank and world size")

        index_payload_list: list[object] = [dict(index) if distributed_rank == 0 else {}]
        ids_payload_list: list[object] = [sorted(ids_to_generate) if distributed_rank == 0 else []]
        dist.broadcast_object_list(index_payload_list, src=0)
        dist.broadcast_object_list(ids_payload_list, src=0)

        index = _parse_str_dict(index_payload_list[0], "broadcast embedding index")
        ids_to_generate = set(_parse_str_list(ids_payload_list[0], "broadcast ids_to_generate"))

    if ids_to_generate:
        if not allow_generation:
            if invalid_ids:
                raise ValueError(_build_invalid_ids_error_message(invalid_ids))
            raise FileNotFoundError(_build_missing_ids_error_message(ids_to_generate))

        search_roots = _resolve_sequence_search_roots(config=config, split_paths=split_paths)
        if distributed_context is None:
            discovered_sequences = _discover_sequences(
                required_ids=ids_to_generate,
                search_roots=search_roots,
                explicit_sequence_file=settings.sequence_file,
                id_column_override=settings.id_column,
                sequence_column_override=settings.sequence_column,
            )
            generated_updates = _generate_missing_embeddings(
                missing_ids=ids_to_generate,
                sequences=discovered_sequences,
                settings=settings,
                cache_dir=cache_dir,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
            )
            index.update(generated_updates)
            _write_json_atomic(path=index_path, payload=index)
            _write_json_atomic(path=metadata_path, payload=expected_metadata)
        else:
            if distributed_rank is None or distributed_world_size is None:
                raise RuntimeError(
                    "Distributed generation context must include rank and world size"
                )
            sorted_ids_to_generate = sorted(ids_to_generate)
            local_ids = _shard_ids_for_rank(
                sorted_ids=sorted_ids_to_generate,
                rank=distributed_rank,
                world_size=distributed_world_size,
            )
            _log_embedding_progress("Assigned %d proteins to this rank", len(local_ids))
            local_updates: dict[str, str] = {}
            local_error: str | None = None
            try:
                if local_ids:
                    discovered_sequences = _discover_sequences(
                        required_ids=local_ids,
                        search_roots=search_roots,
                        explicit_sequence_file=settings.sequence_file,
                        id_column_override=settings.id_column,
                        sequence_column_override=settings.sequence_column,
                    )
                    local_updates = _generate_missing_embeddings(
                        missing_ids=local_ids,
                        sequences=discovered_sequences,
                        settings=settings,
                        cache_dir=cache_dir,
                        input_dim=input_dim,
                        max_sequence_length=max_sequence_length,
                    )
            except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
                local_error = f"rank {distributed_rank}: {error}"

            error_reports: list[object] = [None for _ in range(distributed_world_size)]
            dist.all_gather_object(error_reports, local_error)
            errors = [report for report in error_reports if isinstance(report, str) and report]
            if errors:
                raise RuntimeError("Distributed embedding generation failed: " + " | ".join(errors))

            if distributed_rank == 0:
                gathered_updates: list[object] = [None for _ in range(distributed_world_size)]
                dist.gather_object(local_updates, gathered_updates, dst=0)
                for rank_updates_payload in gathered_updates:
                    index.update(_parse_str_dict(rank_updates_payload, "gathered index updates"))
                _write_json_atomic(path=index_path, payload=index)
                _write_json_atomic(path=metadata_path, payload=expected_metadata)
            else:
                dist.gather_object(local_updates, None, dst=0)

            dist.barrier()
            index = _load_index(index_path=index_path)

    final_missing_ids, final_invalid_ids = _find_missing_or_invalid_ids(
        required_ids=required_ids,
        cache_dir=cache_dir,
        index=index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )
    if final_missing_ids:
        raise FileNotFoundError(_build_missing_ids_error_message(final_missing_ids))
    if final_invalid_ids:
        raise ValueError(_build_invalid_ids_error_message(final_invalid_ids))

    return EmbeddingCacheManifest(
        cache_dir=cache_dir,
        index=index,
        required_ids=frozenset(required_ids),
    )
