"""Tests for MultimodalEmbedder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.embed.core.types import (
    BackboneCoordinateSet,
    EmbeddingConfig,
    EmbeddingResult,
    MultimodalStructurePayload,
    ResidueMask,
)
from src.embed.io.structure import StructureDataNotFoundError
from src.embed.pipelines.multimodal import MultimodalEmbedder


@pytest.fixture()
def embed_config(tmp_path: Path) -> EmbeddingConfig:
    return EmbeddingConfig().with_updates(data_root=tmp_path)


def _payload() -> MultimodalStructurePayload:
    coords = BackboneCoordinateSet(np.zeros((1, 4, 3), dtype=np.float32))
    mask = ResidueMask(np.array([True]))
    return MultimodalStructurePayload(
        uniprot_id="P12345",
        sequence="A",
        coordinates=coords,
        mask=mask,
        metadata={"pdb_id": "1abc", "coordinate_file": "coords.cif"},
    )


class _StubBackend:
    def __init__(self, tracks: dict[str, dict[str, np.ndarray | None]]):
        self._tracks = tracks

    def embed(self, payload: MultimodalStructurePayload):
        return self._tracks


class _StubSequenceEmbedder:
    def __init__(self, embedding: np.ndarray):
        self.embedding = embedding

    def embed_one(self, uniprot_id: str, sequence: str) -> EmbeddingResult:
        return EmbeddingResult(
            uniprot_id=uniprot_id,
            embedding=self.embedding,
            metadata={"pipeline": "sequence"},
        )


def test_multimodal_embedder_produces_embeddings(embed_config: EmbeddingConfig) -> None:
    captured: dict = {}

    def loader(data_root: Path, uniprot_id: str, **kwargs):
        captured["data_root"] = data_root
        captured["uniprot_id"] = uniprot_id
        captured.update(kwargs)
        return _payload()

    embedding = np.ones((1, 1, 3), dtype=np.float32)
    mean_embedding = np.zeros((1, 3), dtype=np.float32)
    backend = _StubBackend(
        {
            "sequence_structure": {"per_residue": embedding, "mean": mean_embedding},
            "structure_only": {"per_residue": embedding * 2, "mean": None},
        }
    )

    embedder = MultimodalEmbedder(
        config=embed_config,
        structure_loader=loader,
        backend_factory=lambda model: backend,
        sequence_fallback=_StubSequenceEmbedder(np.zeros(1, dtype=np.float32)),
    )
    embedder._backend = backend  # bypass model loading for test

    result = embedder.embed_one("P12345", "A")

    assert captured["data_root"] == embed_config.data_root
    assert captured["sequence"] == "A"
    assert result.error is None
    assert np.allclose(result.embedding, embedding.squeeze())
    assert result.metadata["mask_complete"] is True
    assert result.metadata["pdb_id"] == "1abc"
    assert result.metadata["mean_embedding"] == mean_embedding.squeeze().tolist()
    tracks_meta = result.metadata["multimodal_tracks"]
    assert tracks_meta["sequence_structure"]["per_residue_shape"] == (1, 1, 3)
    assert tracks_meta["sequence_structure"]["has_mean"] is True
    assert tracks_meta["structure_only"]["has_mean"] is False


def test_multimodal_embedder_handles_missing_structure(
    embed_config: EmbeddingConfig,
) -> None:
    def loader(data_root: Path, uniprot_id: str, **kwargs):
        raise StructureDataNotFoundError("no coordinates")

    fallback = _StubSequenceEmbedder(np.array([0.5], dtype=np.float32))
    embedder = MultimodalEmbedder(
        config=embed_config,
        structure_loader=loader,
        sequence_fallback=fallback,
    )

    result = embedder.embed_one("P12345", "A")

    assert result.error is None
    assert result.metadata["pipeline"] == "sequence"
    assert result.metadata.get("fallback_reason") == "no coordinates"


def test_multimodal_embedder_validates_sequence(embed_config: EmbeddingConfig) -> None:
    embedder = MultimodalEmbedder(
        config=embed_config,
        structure_loader=lambda *a, **k: _payload(),
        sequence_fallback=_StubSequenceEmbedder(np.zeros(1, dtype=np.float32)),
    )

    result = embedder.embed_one("P12345", "")

    assert result.error == "invalid protein sequence"


# ===== Deterministic Tests with Synthetic Fixtures =====


@pytest.fixture()
def synthetic_structure_data(tmp_path: Path) -> Path:
    """Create minimal synthetic structure data for deterministic testing.

    Mimics the structure of data/embed/ but with tiny, hand-crafted proteins.
    """
    import csv

    root = tmp_path / "synthetic_data"
    consolidated = root / "consolidated"
    consolidated.mkdir(parents=True)
    masks_dir = root / "structures" / "masks"
    masks_dir.mkdir(parents=True)

    # Create synthetic backbone coordinates (3 residues, perfect geometry)
    coords_npz = consolidated / "backbone_coordinates_test.npz"

    # Protein 1: 3-residue helix with complete coordinates
    residues_complete = np.empty(3, dtype=object)
    for i in range(3):
        residues_complete[i] = {
            "N": [float(i), 0.0, 0.0],
            "CA": [float(i), 1.0, 0.0],
            "C": [float(i), 2.0, 0.0],
            "O": [float(i), 3.0, 0.0],
        }

    # Protein 2: 2-residue with one missing backbone atom
    residues_partial = np.empty(2, dtype=object)
    residues_partial[0] = {
        "N": [0.0, 0.0, 0.0],
        "CA": [1.0, 1.0, 1.0],
        "C": [2.0, 2.0, 2.0],
        "O": [3.0, 3.0, 3.0],
    }
    residues_partial[1] = {
        "N": [0.5, 0.5, 0.5],
        "CA": [1.5, 1.5, 1.5],
        "C": [2.5, 2.5, 2.5],
        "O": [np.nan, np.nan, np.nan],  # Missing O atom
    }

    np.savez_compressed(
        coords_npz,
        SYNTH001=residues_complete,
        SYNTH002=residues_partial,
    )

    # Create metadata CSV
    metadata_csv = consolidated / "esm3_ready_proteins_test.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uniprot_id",
                "pdb_id",
                "original_sequence",
                "structure_sequence",
                "coordinate_file",
                "chain_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uniprot_id": "SYNTH001",
                "pdb_id": "SYN1",
                "original_sequence": "ACK",
                "structure_sequence": "ACK",
                "coordinate_file": "synthetic/synth001.cif.gz",
                "chain_id": "A",
            }
        )
        writer.writerow(
            {
                "uniprot_id": "SYNTH002",
                "pdb_id": "SYN2",
                "original_sequence": "MG",
                "structure_sequence": "MG",
                "coordinate_file": "synthetic/synth002.cif.gz",
                "chain_id": "B",
            }
        )

    # Create residue masks
    np.save(masks_dir / "SYNTH001.npy", np.array([True, True, True]))
    np.save(masks_dir / "SYNTH002.npy", np.array([True, False]))

    return root


def test_multimodal_embedder_with_synthetic_complete_structure(
    synthetic_structure_data: Path,
) -> None:
    """Test multimodal embedder with complete synthetic structure data."""
    config = EmbeddingConfig().with_updates(data_root=synthetic_structure_data)

    # Create embedder with mocked backend
    def mock_loader(data_root, uniprot_id, **kwargs):
        from src.embed.io.structure import load_multimodal_structure

        return load_multimodal_structure(data_root, uniprot_id, **kwargs)

    # Mock backend that returns deterministic embeddings
    class MockBackend:
        def __init__(self, model):
            pass

        def embed(self, payload):
            seq_len = len(payload.sequence)
            # Return simple linear embeddings
            per_residue = np.arange(seq_len * 4, dtype=np.float32).reshape(
                1, seq_len, 4
            )
            mean_emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 4)
            return {
                "sequence_structure": {"per_residue": per_residue, "mean": mean_emb},
                "structure_only": {"per_residue": per_residue * 0.5, "mean": None},
            }

    embedder = MultimodalEmbedder(
        config=config,
        structure_loader=mock_loader,
        backend_factory=lambda model: MockBackend(model),
        sequence_fallback=_StubSequenceEmbedder(np.zeros(1, dtype=np.float32)),
    )
    embedder._backend = MockBackend(None)  # Bypass model loading

    result = embedder.embed_one("SYNTH001", "ACK")

    # Validate results
    assert result.error is None
    assert result.embedding.shape == (3, 4)  # 3 residues, 4 features
    assert result.metadata["pipeline"] == "multimodal"
    assert result.metadata["sequence_length"] == 3
    assert result.metadata["residue_count"] == 3
    assert result.metadata["mask_complete"] is True
    assert result.metadata["structure_coverage"] == 1.0
    assert "multimodal_tracks" in result.metadata
    assert "mean_embedding" in result.metadata


def test_multimodal_embedder_with_synthetic_partial_structure(
    synthetic_structure_data: Path,
) -> None:
    """Test multimodal embedder with partial synthetic structure data."""
    config = EmbeddingConfig().with_updates(data_root=synthetic_structure_data)

    def mock_loader(data_root, uniprot_id, **kwargs):
        from src.embed.io.structure import load_multimodal_structure

        return load_multimodal_structure(data_root, uniprot_id, **kwargs)

    class MockBackend:
        def __init__(self, model):
            pass

        def embed(self, payload):
            seq_len = len(payload.sequence)
            per_residue = np.arange(seq_len * 4, dtype=np.float32).reshape(
                1, seq_len, 4
            )
            return {
                "sequence_structure": {"per_residue": per_residue, "mean": None},
                "structure_only": {"per_residue": per_residue * 0.5, "mean": None},
            }

    embedder = MultimodalEmbedder(
        config=config,
        structure_loader=mock_loader,
        backend_factory=lambda model: MockBackend(model),
        sequence_fallback=_StubSequenceEmbedder(np.zeros(1, dtype=np.float32)),
    )
    embedder._backend = MockBackend(None)

    result = embedder.embed_one("SYNTH002", "MG")

    # Validate results with partial structure
    assert result.error is None
    assert result.embedding.shape == (2, 4)  # 2 residues
    assert result.metadata["pipeline"] == "multimodal"
    assert result.metadata["sequence_length"] == 2
    assert result.metadata["mask_complete"] is False  # One residue has missing atoms
    assert result.metadata["structure_coverage"] == 0.5  # 1 out of 2 residues complete


# ===== Backend Error Handling Tests =====


def test_multimodal_embedder_handles_backend_initialization_error(
    embed_config: EmbeddingConfig,
) -> None:
    """Test fallback when backend initialization fails."""
    from src.embed.pipelines.multimodal import LocalModelBackendError
    import src.embed.pipelines.multimodal as mm_module

    def mock_load_model(model_name, device):
        raise LocalModelBackendError("Model not found")

    # Patch _load_local_model to avoid real model loading
    original_load = mm_module._load_local_model
    mm_module._load_local_model = mock_load_model

    try:
        fallback = _StubSequenceEmbedder(np.array([0.7], dtype=np.float32))
        embedder = MultimodalEmbedder(
            config=embed_config,
            structure_loader=lambda *a, **k: _payload(),
            sequence_fallback=fallback,
        )

        result = embedder.embed_one("P12345", "A")

        assert result.error is None
        assert result.metadata["pipeline"] == "sequence"
        assert "fallback_reason" in result.metadata
        assert "Model not found" in result.metadata["fallback_reason"]
    finally:
        mm_module._load_local_model = original_load


def test_multimodal_embedder_handles_embedding_runtime_error(
    embed_config: EmbeddingConfig,
) -> None:
    """Test fallback when embedding execution fails."""

    class FailingBackend:
        def embed(self, payload):
            raise RuntimeError("CUDA out of memory")

    fallback = _StubSequenceEmbedder(np.array([0.8], dtype=np.float32))
    embedder = MultimodalEmbedder(
        config=embed_config,
        structure_loader=lambda *a, **k: _payload(),
        backend_factory=lambda model: FailingBackend(),
        sequence_fallback=fallback,
    )
    embedder._backend = FailingBackend()

    result = embedder.embed_one("P12345", "A")

    assert result.error is None
    assert result.metadata["pipeline"] == "sequence"
    assert "fallback_reason" in result.metadata
    assert "CUDA out of memory" in result.metadata["fallback_reason"]


def test_multimodal_embedder_handles_multimodal_backend_error(
    embed_config: EmbeddingConfig,
) -> None:
    """Test fallback when backend returns no embeddings."""
    from src.embed.pipelines.multimodal import MultimodalBackendError

    class BackendWithoutEmbeddings:
        def embed(self, payload):
            raise MultimodalBackendError("backend did not produce embeddings")

    fallback = _StubSequenceEmbedder(np.array([0.9], dtype=np.float32))
    embedder = MultimodalEmbedder(
        config=embed_config,
        structure_loader=lambda *a, **k: _payload(),
        backend_factory=lambda model: BackendWithoutEmbeddings(),
        sequence_fallback=fallback,
    )
    embedder._backend = BackendWithoutEmbeddings()

    result = embedder.embed_one("P12345", "A")

    assert result.error is None
    assert result.metadata["pipeline"] == "sequence"
    assert "fallback_reason" in result.metadata


def test_multimodal_embedder_handles_general_structure_loading_error(
    embed_config: EmbeddingConfig,
) -> None:
    """Test fallback when structure loading raises unexpected exception."""

    def failing_loader(*args, **kwargs):
        raise ValueError("Unexpected file format error")

    fallback = _StubSequenceEmbedder(np.array([0.6], dtype=np.float32))
    embedder = MultimodalEmbedder(
        config=embed_config,
        structure_loader=failing_loader,
        sequence_fallback=fallback,
    )

    result = embedder.embed_one("P12345", "A")

    assert result.error is None
    assert result.metadata["pipeline"] == "sequence"
    assert "fallback_reason" in result.metadata


def test_multimodal_embedder_creates_fallback_when_none(
    embed_config: EmbeddingConfig,
) -> None:
    """Test that fallback embedder is created on-the-fly when None."""
    from src.embed.pipelines.sequence import SequenceEmbedder

    def loader(*args, **kwargs):
        raise StructureDataNotFoundError("no structure")

    # Mock SequenceEmbedder to avoid real model loading
    class MockSequenceEmbedder:
        def __init__(self, config):
            self.config = config
            self.call_count = 0

        def embed_one(self, uniprot_id: str, sequence: str) -> EmbeddingResult:
            self.call_count += 1
            return EmbeddingResult(
                uniprot_id=uniprot_id,
                embedding=np.array([0.5] * len(sequence), dtype=np.float32),
                metadata={"pipeline": "sequence"},
            )

    import src.embed.pipelines.multimodal as mm_module

    original_seq_embedder = mm_module.SequenceEmbedder
    mm_module.SequenceEmbedder = MockSequenceEmbedder

    try:
        embedder = MultimodalEmbedder(
            config=embed_config,
            structure_loader=loader,
            sequence_fallback=None,  # No fallback initially
        )

        # First call should create the fallback
        assert embedder._sequence_fallback is None
        result = embedder.embed_one("P12345", "A")

        # Fallback should now be created
        assert embedder._sequence_fallback is not None
        assert result.metadata.get("pipeline") == "sequence"

        # Second call should reuse the same fallback
        fallback_instance = embedder._sequence_fallback
        result2 = embedder.embed_one("P67890", "MK")
        assert result2.metadata.get("pipeline") == "sequence"
        assert embedder._sequence_fallback is fallback_instance  # Same instance
    finally:
        mm_module.SequenceEmbedder = original_seq_embedder


def test_multimodal_embedder_require_backend_initialization(
    embed_config: EmbeddingConfig,
) -> None:
    """Test backend initialization through _require_backend."""
    from src.embed.pipelines.multimodal import LocalMultimodalBackend

    class MockModel:
        pass

    initialized_models = []

    def mock_load_model(model_name, device):
        model = MockModel()
        initialized_models.append((model_name, device))
        return model

    # Patch _load_local_model
    import src.embed.pipelines.multimodal as mm_module

    original_load = mm_module._load_local_model
    mm_module._load_local_model = mock_load_model

    try:
        embedder = MultimodalEmbedder(
            config=embed_config,
            structure_loader=lambda *a, **k: _payload(),
        )

        # Backend should be None initially
        assert embedder._backend is None

        # Call _require_backend
        backend = embedder._require_backend()

        # Backend should now be initialized
        assert embedder._backend is not None
        assert isinstance(embedder._backend, LocalMultimodalBackend)
        assert len(initialized_models) == 1
        assert initialized_models[0][0] == embed_config.model_name

        # Calling again should return the same backend
        backend2 = embedder._require_backend()
        assert backend2 is backend
        assert len(initialized_models) == 1  # No new model loaded

    finally:
        mm_module._load_local_model = original_load


# ===== LocalMultimodalBackend Tests =====


def test_local_multimodal_backend_missing_dependencies() -> None:
    """Test that backend raises error when dependencies are missing."""
    from src.embed.pipelines.multimodal import (
        LocalMultimodalBackend,
        MultimodalBackendError,
    )
    import src.embed.pipelines.multimodal as mm_module

    # Temporarily remove torch/ESMProtein
    original_torch = mm_module.torch
    original_esm = mm_module.ESMProtein
    original_logits = mm_module.LogitsConfig
    original_residue = mm_module.residue_constants

    try:
        mm_module.torch = None
        mm_module.ESMProtein = None
        mm_module.LogitsConfig = None
        mm_module.residue_constants = None

        backend = LocalMultimodalBackend(model=None)

        with pytest.raises(MultimodalBackendError) as exc_info:
            backend.embed(_payload())

        assert "esm3 package with torch support is required" in str(exc_info.value)

    finally:
        mm_module.torch = original_torch
        mm_module.ESMProtein = original_esm
        mm_module.LogitsConfig = original_logits
        mm_module.residue_constants = original_residue


def test_local_multimodal_backend_embed_with_mock_model() -> None:
    """Test LocalMultimodalBackend.embed() with a mock model."""
    from src.embed.pipelines.multimodal import LocalMultimodalBackend
    import src.embed.pipelines.multimodal as mm_module

    # Skip test if torch is not available
    if mm_module.torch is None:
        pytest.skip("torch not available")

    class MockTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    class MockLogitsOutput:
        def __init__(self, with_mean=True):
            self.embeddings = MockTensor()
            if with_mean:
                self.mean_embedding = MockTensor()

    class MockModel:
        def encode(self, protein):
            return "encoded_protein_tensor"

        def logits(self, protein_tensor, config):
            # Return with mean for sequence_structure, without for structure_only
            has_mean = config.sequence
            return MockLogitsOutput(with_mean=has_mean)

    backend = LocalMultimodalBackend(model=MockModel())
    result = backend.embed(_payload())

    # Validate outputs
    assert "sequence_structure" in result
    assert "structure_only" in result
    assert result["sequence_structure"]["per_residue"] is not None
    assert result["sequence_structure"]["mean"] is not None
    assert result["structure_only"]["per_residue"] is not None
    assert result["structure_only"]["mean"] is None


def test_local_multimodal_backend_run_logits_no_embeddings() -> None:
    """Test _run_logits raises error when backend doesn't produce embeddings."""
    from src.embed.pipelines.multimodal import (
        LocalMultimodalBackend,
        MultimodalBackendError,
    )
    import src.embed.pipelines.multimodal as mm_module

    if mm_module.torch is None:
        pytest.skip("torch not available")

    class MockLogitsOutputNoEmbeddings:
        pass  # No embeddings attribute

    class MockModel:
        def logits(self, protein_tensor, config):
            return MockLogitsOutputNoEmbeddings()

    backend = LocalMultimodalBackend(model=MockModel())

    with pytest.raises(MultimodalBackendError) as exc_info:
        backend._run_logits("protein_tensor", None)

    assert "backend did not produce embeddings" in str(exc_info.value)


def test_local_multimodal_backend_build_protein() -> None:
    """Test _build_protein constructs ESMProtein correctly."""
    from src.embed.pipelines.multimodal import LocalMultimodalBackend
    import src.embed.pipelines.multimodal as mm_module

    if mm_module.torch is None or mm_module.ESMProtein is None:
        pytest.skip("torch or ESMProtein not available")

    # Create a more complex payload with multiple residues
    coords_array = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
            [
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [np.nan, np.nan, np.nan],
                [3.0, 2.0, 0.0],
            ],  # Missing C
        ],
        dtype=np.float32,
    )
    coords = BackboneCoordinateSet(coords_array)
    mask = ResidueMask(np.array([True, True, False]))  # Third residue masked out
    payload = MultimodalStructurePayload(
        uniprot_id="TEST",
        sequence="ACK",
        coordinates=coords,
        mask=mask,
        metadata={},
    )

    backend = LocalMultimodalBackend(model=None)
    protein = backend._build_protein(payload)

    # Validate protein
    assert protein.sequence == "ACK"
    assert protein.coordinates is not None
    assert protein.coordinates.shape[0] == 3  # 3 residues

    # Check that masked residue (index 2) has NaN coordinates
    assert mm_module.torch.isnan(protein.coordinates[2]).all()

    # Check that valid residues have actual coordinates
    assert not mm_module.torch.isnan(protein.coordinates[0, 0]).any()  # N atom
    assert not mm_module.torch.isnan(protein.coordinates[1, 0]).any()


def test_local_multimodal_backend_build_protein_with_edge_cases() -> None:
    """Test _build_protein with unknown atoms and NaN coordinates."""
    from src.embed.pipelines.multimodal import LocalMultimodalBackend
    import src.embed.pipelines.multimodal as mm_module

    if mm_module.torch is None or mm_module.ESMProtein is None:
        pytest.skip("torch or ESMProtein not available")

    # Create payload with edge cases:
    # - Unknown atom type that won't be in residue_constants.atom_order
    # - NaN coordinates that should be skipped
    class CustomCoordinateSet:
        def __init__(self):
            self.residue_count = 2
            # Use custom atom order including an unknown atom "XX"
            self.atom_order = ["N", "CA", "XX", "C"]  # XX is unknown
            # First residue has NaN in one coordinate
            self.values = np.array(
                [
                    [
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0],
                        [np.nan, np.nan, np.nan],
                    ],
                    [
                        [4.0, 4.0, 4.0],
                        [5.0, 5.0, 5.0],
                        [6.0, 6.0, 6.0],
                        [7.0, 7.0, 7.0],
                    ],
                ],
                dtype=np.float32,
            )
            self.source = None

    coords = CustomCoordinateSet()
    mask = ResidueMask(np.array([True, True]))
    payload = MultimodalStructurePayload(
        uniprot_id="EDGE",
        sequence="MK",
        coordinates=coords,
        mask=mask,
        metadata={},
    )

    backend = LocalMultimodalBackend(model=None)
    protein = backend._build_protein(payload)

    # Validate protein was built
    assert protein.sequence == "MK"
    assert protein.coordinates is not None
    assert protein.coordinates.shape[0] == 2

    # Check that known atoms were set
    assert not mm_module.torch.isnan(protein.coordinates[0, 0]).any()  # N atom
    assert not mm_module.torch.isnan(protein.coordinates[1, 0]).any()  # N atom

    # The coordinate at position with NaN should remain as NaN
    # (fourth atom in first residue, which is C)
    c_atom_index = mm_module.residue_constants.atom_order.get("C")
    if c_atom_index is not None:
        assert mm_module.torch.isnan(protein.coordinates[0, c_atom_index]).all()
