"""Thin CLI wrapper around the embed pipelines."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict

from ..config import default_config
from ..io import load_sequences, save_results
from ..pipelines import SequenceEmbedder
from ..pipelines.multimodal import MultimodalEmbedder


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the embed pipeline (sequence-only or multimodal)"
    )
    parser.add_argument("input", type=Path, help="Path to FASTA or JSON input file")
    parser.add_argument("output", type=Path, help="Where to write NPZ results")
    parser.add_argument(
        "--mode",
        choices=["sequence", "multimodal"],
        default="sequence",
        help="Pipeline mode: sequence-only or sequence+structure (default: sequence)",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "fasta", "json", "csv"],
        default="auto",
        help="Input format (default: infer from file suffix)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Data root directory (default: EMBED_DATA_ROOT env or data/embed)",
    )
    parser.add_argument(
        "--csv-id-column",
        default="uniprotID",
        help="Column name containing identifiers when --input-format=csv (default: uniprotID)",
    )
    parser.add_argument(
        "--csv-sequence-column",
        default="sequence",
        help="Column name containing sequences when --input-format=csv (default: sequence)",
    )
    parser.add_argument(
        "--truncate-long-sequences",
        action="store_true",
        help=(
            "Truncate sequences longer than EMBED_MAX_SEQUENCE_LENGTH instead of failing "
            "(sequence-only pipeline, default: disabled)"
        ),
    )
    parser.add_argument(
        "--retry-truncate-errors",
        action="store_true",
        help=(
            "After the primary run, retry failed sequences longer than the max length "
            "by truncating them (sequence mode only)."
        ),
    )
    parser.add_argument(
        "--truncate-retry-length",
        type=int,
        default=None,
        help=(
            "Override the max length used during the truncation retry "
            "(default: EMBED_MAX_SEQUENCE_LENGTH)."
        ),
    )
    parser.add_argument(
        "--truncate-retry-output",
        type=Path,
        help=(
            "Optional NPZ output path for truncated retries "
            "(default: <output>_truncated_<length>.npz)."
        ),
    )
    parser.add_argument(
        "--truncate-retry-json",
        type=Path,
        help=(
            "Optional JSON path to record sequences retried with truncation "
            "(default: <output>_errors_retry.json)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve data_root with precedence: CLI > env > default
    if args.data_root is not None:
        data_root = args.data_root
        logger.info(f"Using data_root from CLI: {data_root}")
    elif "EMBED_DATA_ROOT" in os.environ:
        data_root = Path(os.environ["EMBED_DATA_ROOT"])
        logger.info(f"Using data_root from EMBED_DATA_ROOT env: {data_root}")
    else:
        # Get default from config
        default_cfg = default_config()
        data_root = default_cfg.data_root
        logger.info(f"Using default data_root: {data_root}")

    # Validate data_root exists for multimodal mode
    if args.mode == "multimodal":
        if not data_root.exists():
            parser.error(
                f"Data root directory does not exist: {data_root}\n"
                f"Multimodal mode requires structure data under data_root.\n"
                f"Specify --data-root or set EMBED_DATA_ROOT environment variable."
            )
        logger.info(f"Multimodal mode: validated data_root exists at {data_root}")

    # Prepare output path
    output_path = args.output
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".npz")
    elif output_path.suffix.lower() != ".npz":
        parser.error("Output file must have .npz extension")

    # Load sequences
    try:
        sequences = load_sequences(
            args.input,
            input_format=args.input_format,
            csv_id_column=args.csv_id_column,
            csv_sequence_column=args.csv_sequence_column,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Create config with resolved data_root
    config = default_config()
    config.data_root = data_root
    if args.truncate_long_sequences:
        config.truncate_long_sequences = True

    # Instantiate embedder based on mode
    if args.mode == "sequence":
        logger.info("Running sequence-only pipeline")
        embedder = SequenceEmbedder(config)
    else:
        logger.info("Running multimodal pipeline (sequence+structure)")
        embedder = MultimodalEmbedder(config)

    # Process and save
    results = embedder.embed_many(sequences.items())
    save_results(results, output_path)
    logger.info(f"Results saved to {output_path}")
    if args.retry_truncate_errors:
        _retry_truncate_failures(args, config, output_path)
    return 0


app = main


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover - CLI entrypoint


def _retry_truncate_failures(
    args: argparse.Namespace, config, output_path: Path
) -> None:
    """Retry failed sequences by truncating them."""

    if args.mode != "sequence":
        logger.info(
            "Truncation retry requested but mode=%s; skipping (sequence-only).",
            args.mode,
        )
        return

    error_path = output_path.with_suffix(output_path.suffix + ".errors.json")
    if not error_path.exists():
        logger.info("No error file at %s; skipping truncation retry.", error_path)
        return

    try:
        errors = json.loads(error_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse %s: %s", error_path, exc)
        return

    retry_length = args.truncate_retry_length or config.max_sequence_length
    if retry_length is None or retry_length <= 0:
        logger.warning(
            "Invalid truncate retry length (%s); skipping truncation retry.",
            retry_length,
        )
        return

    filtered: Dict[str, str] = {}
    for entry in errors:
        identifier = entry.get("uniprot_id")
        metadata: dict = entry.get("metadata") or {}
        sequence = metadata.get("original_sequence")
        if not identifier or not sequence:
            continue
        if len(sequence) <= retry_length:
            continue
        filtered.setdefault(identifier, sequence)

    if not filtered:
        logger.info(
            "No sequences exceeded length %d; skipping truncation retry.",
            retry_length,
        )
        return

    retry_json = args.truncate_retry_json
    if retry_json is None:
        retry_json = output_path.with_name(f"{output_path.stem}_errors_retry.json")
    retry_json.write_text(
        json.dumps(
            [
                {"id": identifier, "sequence": sequence}
                for identifier, sequence in filtered.items()
            ],
            indent=2,
        )
    )
    logger.info(
        "Prepared %d sequences for truncation retry (saved to %s).",
        len(filtered),
        retry_json,
    )

    retry_config = config.with_updates(
        max_sequence_length=retry_length,
        truncate_long_sequences=True,
    )
    retry_embedder = SequenceEmbedder(retry_config)
    retry_results = retry_embedder.embed_many(filtered.items())

    retry_output = args.truncate_retry_output
    if retry_output is None:
        retry_output = output_path.with_name(
            f"{output_path.stem}_truncated_{retry_length}{output_path.suffix}"
        )

    save_results(retry_results, retry_output)
    logger.info("Truncated retry results saved to %s", retry_output)
