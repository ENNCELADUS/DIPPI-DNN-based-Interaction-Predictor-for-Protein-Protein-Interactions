# PRING Dataset

Processed protein-protein interaction (PPI) dataset from [PRING](https://huggingface.co/datasets/piaolaidangqu/PRING).

**Species:** Human, Yeast, Arabidopsis (arath), E. coli

## Directory Structure

```
PRING/
├── human/           # Main species with train/val/test splits
│   ├── BFS/         # Breadth-First Search split
│   ├── DFS/         # Depth-First Search split
│   └── RANDOM_WALK/ # Random Walk split
├── yeast/           # Cross-species evaluation
├── arath/           # Cross-species evaluation
└── ecoli/           # Cross-species evaluation
```

## File Formats

| File | Description |
|------|-------------|
| `*_ppi.txt` | PPI pairs: `ProteinA ProteinB Label` (1=positive, 0=negative) |
| `*_simple.fasta` | Protein sequences with UniProt IDs |
| `*.fasta` | Full sequences with complete metadata |
| `*_protein_id.csv` | Mapping: UniProt ID, organism, sequence, length |
| `*_graph.pkl` | NetworkX graph of PPI network |
| `*_sampled_nodes.pkl` | Subgraph samples for graph-level testing |

## Human Splits (BFS/DFS/RANDOM_WALK)

| File | Purpose |
|------|---------|
| `human_train_ppi.txt` | Training pairs |
| `human_val_ppi.txt` | Validation pairs |
| `human_test_ppi.txt` | Test pairs (binary classification) |
| `all_test_ppi.txt` | All-against-all pairs for ranking |

## Preprocessing

Sequences filtered by:
- Length: 50–1000 amino acids
- Sequence similarity: <40% (MMseqs2)
- Function similarity removed across species
