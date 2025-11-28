"""CLI tests for the embed sequence pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from src.embed.core.types import EmbeddingConfig, EmbeddingResult
from src.embed.cli import main as cli_main


class StubEmbedder:
    """Minimal stub in place of the heavy SequenceEmbedder."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config
        self.seen: list[tuple[str, str]] = []
        self.invocations: list[dict[str, Any]] = []

    def embed_many(self, items: Iterable[tuple[str, str]]) -> list[EmbeddingResult]:
        items_list = list(items)
        self.invocations.append({"config": self.config, "items": items_list.copy()})
        results = []
        max_len = (
            self.config.max_sequence_length
            if (self.config and self.config.max_sequence_length > 0)
            else None
        )
        for uniprot_id, sequence in items_list:
            self.seen.append((uniprot_id, sequence))
            should_fail = (
                self.config is not None
                and max_len is not None
                and len(sequence) > max_len
                and not self.config.truncate_long_sequences
            )
            if should_fail:
                results.append(
                    EmbeddingResult(
                        uniprot_id=uniprot_id,
                        embedding=None,
                        metadata={
                            "source": "stub",
                            "original_sequence": sequence,
                        },
                        error="invalid protein sequence",
                    )
                )
                continue

            processed = sequence
            if (
                self.config is not None
                and self.config.truncate_long_sequences
                and max_len is not None
                and len(sequence) > max_len
            ):
                processed = sequence[:max_len]
            embedding = np.full(
                (1, len(processed), 2), fill_value=1.0, dtype=np.float32
            )
            results.append(
                EmbeddingResult(
                    uniprot_id=uniprot_id,
                    embedding=embedding,
                    metadata={
                        "source": "stub",
                        "original_sequence": sequence,
                        "cleaned_sequence": processed,
                    },
                )
            )
        return results


class StubMultimodalEmbedder(StubEmbedder):
    """Minimal stub for MultimodalEmbedder."""

    def embed_many(self, items: Iterable[tuple[str, str]]) -> list[EmbeddingResult]:
        items_list = list(items)
        self.invocations.append({"config": self.config, "items": items_list.copy()})
        results = []
        for uniprot_id, sequence in items_list:
            self.seen.append((uniprot_id, sequence))
            embedding = np.full((1, len(sequence), 2), fill_value=2.0, dtype=np.float32)
            results.append(
                EmbeddingResult(
                    uniprot_id=uniprot_id,
                    embedding=embedding,
                    metadata={
                        "source": "stub_multimodal",
                        "pipeline": "multimodal",
                        "original_sequence": sequence,
                        "cleaned_sequence": sequence,
                    },
                )
            )
        return results


@pytest.fixture
def stub_embedders(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fixture providing both sequence and multimodal stub embedders."""
    sequence_stub = StubEmbedder()
    multimodal_stub = StubMultimodalEmbedder()

    def sequence_factory(config: EmbeddingConfig | None = None) -> StubEmbedder:
        sequence_stub.config = config
        return sequence_stub

    def multimodal_factory(
        config: EmbeddingConfig | None = None,
    ) -> StubMultimodalEmbedder:
        multimodal_stub.config = config
        return multimodal_stub

    monkeypatch.setattr("src.embed.cli.main.SequenceEmbedder", sequence_factory)
    monkeypatch.setattr("src.embed.cli.main.MultimodalEmbedder", multimodal_factory)

    return {
        "sequence": sequence_stub,
        "multimodal": multimodal_stub,
    }


@pytest.fixture(autouse=True)
def stub_sequence_embedder(stub_embedders: dict[str, Any]) -> StubEmbedder:
    """Backward compatibility fixture for existing tests."""
    return stub_embedders["sequence"]


def test_cli_runs_with_fasta_input(
    tmp_path: Path, stub_sequence_embedder: StubEmbedder
) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n>P22222\nGHK\n")
    output = tmp_path / "out"

    exit_code = cli_main.main([str(fasta), str(output)])

    assert exit_code == 0
    with np.load(output.with_suffix(".npz"), allow_pickle=True) as data:
        assert set(data["ids"].tolist()) == {"P11111", "P22222"}
        assert all(meta.get("source") == "stub" for meta in data["metadata"].tolist())
    assert stub_sequence_embedder.seen == [("P11111", "ACD"), ("P22222", "GHK")]


def test_cli_runs_with_json_input(
    tmp_path: Path, stub_sequence_embedder: StubEmbedder
) -> None:
    records = [{"id": "Q1", "sequence": "AC"}, {"id": "Q2", "sequence": "GH"}]
    json_path = tmp_path / "seqs.json"
    json_path.write_text(json.dumps(records))
    output = tmp_path / "out.npz"

    exit_code = cli_main.main([str(json_path), str(output), "--input-format", "json"])

    assert exit_code == 0
    with np.load(output, allow_pickle=True) as payload:
        assert set(payload["ids"].tolist()) == {"Q1", "Q2"}
    assert stub_sequence_embedder.seen == [("Q1", "AC"), ("Q2", "GH")]


def test_cli_errors_on_unknown_format(tmp_path: Path) -> None:
    bogus = tmp_path / "seqs.bin"
    bogus.write_text("dummy")
    output = tmp_path / "out.npz"

    with pytest.raises(SystemExit):
        cli_main.main([str(bogus), str(output)])


def test_cli_multimodal_mode(tmp_path: Path, stub_embedders: dict[str, Any]) -> None:
    """Test that multimodal mode uses MultimodalEmbedder."""
    data_root = tmp_path / "data_root"
    data_root.mkdir()

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main(
        [str(fasta), str(output), "--mode", "multimodal", "--data-root", str(data_root)]
    )

    assert exit_code == 0
    multimodal_stub = stub_embedders["multimodal"]
    assert multimodal_stub.seen == [("P11111", "ACD")]
    assert multimodal_stub.config is not None
    assert multimodal_stub.config.data_root == data_root

    with np.load(output, allow_pickle=True) as data:
        assert "P11111" in data["ids"].tolist()
        metadata = data["metadata"].tolist()[0]
        assert metadata.get("source") == "stub_multimodal"


def test_cli_sequence_mode_explicit(
    tmp_path: Path, stub_embedders: dict[str, Any]
) -> None:
    """Test that explicitly specifying sequence mode works."""
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main([str(fasta), str(output), "--mode", "sequence"])

    assert exit_code == 0
    sequence_stub = stub_embedders["sequence"]
    assert sequence_stub.seen == [("P11111", "ACD")]


def test_cli_defaults_to_sequence_mode(
    tmp_path: Path, stub_embedders: dict[str, Any]
) -> None:
    """Test that mode defaults to sequence when not specified."""
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main([str(fasta), str(output)])

    assert exit_code == 0
    sequence_stub = stub_embedders["sequence"]
    assert sequence_stub.seen == [("P11111", "ACD")]
    # Multimodal should not have been called
    multimodal_stub = stub_embedders["multimodal"]
    assert multimodal_stub.seen == []


def test_cli_enables_truncation_flag(
    tmp_path: Path, stub_embedders: dict[str, Any]
) -> None:
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main([str(fasta), str(output), "--truncate-long-sequences"])

    assert exit_code == 0
    sequence_stub = stub_embedders["sequence"]
    assert sequence_stub.config is not None
    assert sequence_stub.config.truncate_long_sequences is True


def test_cli_retries_truncation_errors(
    tmp_path: Path, stub_embedders: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the CLI can retry long sequences with truncation."""
    monkeypatch.setenv("EMBED_MAX_SEQUENCE_LENGTH", "4")
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n>P22222\nACDEFG\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main(
        [
            str(fasta),
            str(output),
            "--retry-truncate-errors",
            "--truncate-retry-length",
            "4",
        ]
    )

    assert exit_code == 0
    sequence_stub = stub_embedders["sequence"]
    assert len(sequence_stub.invocations) == 2
    retry_config = sequence_stub.invocations[-1]["config"]
    assert retry_config is not None
    assert retry_config.truncate_long_sequences is True
    assert retry_config.max_sequence_length == 4

    retry_output = output.with_name("out_truncated_4.npz")
    assert retry_output.exists()
    retry_json = output.with_name("out_errors_retry.json")
    assert json.loads(retry_json.read_text()) == [
        {"id": "P22222", "sequence": "ACDEFG"}
    ]


def test_cli_data_root_from_cli_flag(
    tmp_path: Path, stub_embedders: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that CLI --data-root flag takes precedence."""
    cli_data_root = tmp_path / "cli_root"
    cli_data_root.mkdir()
    env_data_root = tmp_path / "env_root"
    env_data_root.mkdir()

    # Set env var that should be overridden
    monkeypatch.setenv("EMBED_DATA_ROOT", str(env_data_root))

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main(
        [str(fasta), str(output), "--data-root", str(cli_data_root)]
    )

    assert exit_code == 0
    sequence_stub = stub_embedders["sequence"]
    assert sequence_stub.config is not None
    assert sequence_stub.config.data_root == cli_data_root


def test_cli_data_root_from_env(
    tmp_path: Path, stub_embedders: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that EMBED_DATA_ROOT env var is used when CLI flag not provided."""
    env_data_root = tmp_path / "env_root"
    env_data_root.mkdir()

    monkeypatch.setenv("EMBED_DATA_ROOT", str(env_data_root))

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main([str(fasta), str(output)])

    assert exit_code == 0
    sequence_stub = stub_embedders["sequence"]
    assert sequence_stub.config is not None
    assert sequence_stub.config.data_root == env_data_root


def test_cli_data_root_uses_default(
    tmp_path: Path, stub_embedders: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that default data_root is used when neither CLI nor env provided."""
    # Ensure env var is not set
    monkeypatch.delenv("EMBED_DATA_ROOT", raising=False)

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    exit_code = cli_main.main([str(fasta), str(output)])

    assert exit_code == 0
    sequence_stub = stub_embedders["sequence"]
    assert sequence_stub.config is not None
    # Should have some default value
    assert sequence_stub.config.data_root is not None


def test_cli_multimodal_requires_existing_data_root(tmp_path: Path) -> None:
    """Test that multimodal mode validates data_root exists."""
    nonexistent_root = tmp_path / "does_not_exist"

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    with pytest.raises(SystemExit):
        cli_main.main(
            [
                str(fasta),
                str(output),
                "--mode",
                "multimodal",
                "--data-root",
                str(nonexistent_root),
            ]
        )


def test_cli_sequence_mode_no_data_root_validation(
    tmp_path: Path, stub_embedders: dict[str, Any]
) -> None:
    """Test that sequence mode doesn't require data_root to exist."""
    nonexistent_root = tmp_path / "does_not_exist"

    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.npz"

    # Should succeed even though data_root doesn't exist
    exit_code = cli_main.main(
        [
            str(fasta),
            str(output),
            "--mode",
            "sequence",
            "--data-root",
            str(nonexistent_root),
        ]
    )

    assert exit_code == 0


def test_cli_output_npz_extension_added(
    tmp_path: Path, stub_sequence_embedder: StubEmbedder
) -> None:
    """Test that .npz extension is automatically added to output path."""
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out"  # No extension

    exit_code = cli_main.main([str(fasta), str(output)])

    assert exit_code == 0
    assert (output.with_suffix(".npz")).exists()
    assert not output.exists()  # Original path without extension shouldn't exist


def test_cli_output_wrong_extension_errors(tmp_path: Path) -> None:
    """Test that wrong extension causes an error."""
    fasta = tmp_path / "proteins.fasta"
    fasta.write_text(">P11111\nACD\n")
    output = tmp_path / "out.pkl"

    with pytest.raises(SystemExit):
        cli_main.main([str(fasta), str(output)])
