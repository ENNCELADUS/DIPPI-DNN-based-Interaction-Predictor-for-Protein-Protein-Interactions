## TMP Dataset Summary

| File | Positive | Negative | Total | Neg:Pos Ratio | Unique Proteins |
|------|----------|----------|-------|---------------|-----------------|
| `raw/pretrain.csv` | 719,740 | 18,403,866 | 19,123,606 | 25.6:1 | 18,177 |
| `processed/pretrain.csv` | 719,740 | 18,403,866 | 19,123,606 | 25.6:1 | 18,177 |
| `pretrain_train.csv` | generated from `processed/pretrain.csv` via TPPNI | generated | config-driven | emergent from TPPNI pool | generated |
| `pretrain_val.csv` | generated from `processed/pretrain.csv` via TPPNI | generated | config-driven | emergent from TPPNI pool | generated |
| `raw/finetune.csv` | 65,544 | 2,751,612 | 2,817,156 | 42.0:1 | 9,352 |
| `processed/finetune.csv` | 65,544 | 2,751,612 | 2,817,156 | 42.0:1 | 9,352 |
| `finetune_train.csv` | generated from `processed/finetune.csv` via TPPNI | generated | config-driven | emergent from TPPNI pool | generated |
| `finetune_val.csv` | generated from `processed/finetune.csv` via TPPNI | generated | config-driven | emergent from TPPNI pool | generated |
| `test.csv` | 4,788 | 183,909 | 188,697 | 38.4:1 | 1,282 |

> [!NOTE]
> - `processed/pretrain.csv` and `processed/finetune.csv` are raw stage corpora, not balanced training files.
> - Canonical train/validation files are produced by `python -m src.data_preprocess.prepare_tppni_datasets --config <config>`.
> - Both pretrain and finetune outputs use the full post-CL3 TPPNI set for each split.
> - Output class ratios are emergent from the paper pipeline, not forced to `1:1` and not copied from the raw corpus.

---

## TMP vs membrane_protein Comparison

| Dataset | TMP Total | membrane Total | Size Ratio | TMP Neg:Pos | membrane Neg:Pos |
|---------|-----------|----------------|------------|-------------|------------------|
| **Pretrain** | 19,123,606 | 266,356 | 72x | 25.6:1 | 3.0:1 |
| **Finetune** | 567,895 | 56,040 | 10x | 7.7:1 | 2.0:1 |
| **Test** | 188,697 | 9,576 | 20x | 38.4:1 | 1:1 |

## Unique Proteins

| Dataset | TMP | membrane_protein | Overlap |
|---------|-----|------------------|---------|
| Pretrain | 18,177 | 8,742 | 6,730 (77%) |
| Finetune | 4,235 | 2,306 | 2,306 (100%) |
| Test | 1,282 | 1,282 | 1,282 (identical) |

---

## Data Splits

**Pretrain source corpus (raw stage input):**
| Split | Total | Pos | Neg | Ratio |
|-------|-------|-----|-----|-------|
| Full | 19,123,606 | 719,740 | 18,403,866 | 1:25.6 |

**Pretrain generated datasets:**
| Split | Total | Pos | Neg | Ratio |
|-------|-------|-----|-----|-------|
| Train | generated from config-driven split | generated | generated | emergent from TPPNI pool |
| Val | generated from config-driven split | generated | generated | emergent from TPPNI pool |

**Finetune generated datasets:**
| Split | Total | Pos | Neg | Ratio |
|-------|-------|-----|-----|-------|
| Train | generated from config-driven split | generated | generated | emergent from TPPNI pool |
| Val | generated from config-driven split | generated | generated | emergent from TPPNI pool |

## Unique Proteins

| Dataset | TMP/processed | membrane_protein | Overlap |
|---------|---------------|------------------|---------|
| Pretrain | 18,177 | 8,742 | 6,730 (77% of membrane) |
| Finetune | 4,235 | 2,306 | 2,306 (100% of membrane) |
| Test | 1,282 | 1,282 | **1,282 (identical)** |

## Protein Sequences

### `all_proteins.fasta`

FASTA format file containing all unique protein sequences in the TMP dataset.

**File Structure:**
- Format: Standard FASTA format
- Header: `>UniProtID`
- Sequence: One line per sequence (no line wrapping)

**Content:**
- Total proteins: 21,394 unique sequences
- Corresponds to all proteins from `all_proteins.csv`
- Generated via `convert_to_fasta.py`
