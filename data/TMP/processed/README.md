## TMP Dataset Summary

| File | Positive | Negative | Total | Neg:Pos Ratio | Unique Proteins |
|------|----------|----------|-------|---------------|-----------------|
| `raw/pretrain.csv` | 719,740 | 18,403,866 | 19,123,606 | 25.6:1 | 18,177 |
| `processed/pretrain.csv` | 719,740 | 719,740 | 1,439,480 | 1:1 | 17,137 |
| `pretrain_train_balanced.csv` | 683,752 | 683,752 | 1,367,504 | 1:1 | 17,022 |
| `pretrain_val_balanced.csv` | 35,987 | 35,987 | 71,974 | 1:1 | 10,879 |
| `raw/finetune.csv` | 65,544 | 2,751,612 | 2,817,156 | 42.0:1 | 9,352 |
| `processed/finetune.csv` | 65,544 | 502,351 | 567,895 | 7.7:1 | 4,235 |
| `finetune_train.csv` | 58,989 | 452,115 | 511,104 | 7.7:1 | 4,189 |
| `finetune_val.csv` | 6,555 | 50,236 | 56,791 | 7.7:1 | 2,778 |
| `test.csv` | 4,788 | 183,909 | 188,697 | 38.4:1 | 1,282 |

> [!NOTE]
> - `raw/finetune.csv` was downsampled from 2.8M (42:1) → 567K (7.7:1) to reduce class imbalance
> - Balanced pretrain files use DDB negative sampling (1:1 ratio). See `docs/roadmap/12negative_sampling.md`

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

**Pretrain (95/5) - Original (unbalanced):**
| Split | Total | Pos | Neg | Ratio |
|-------|-------|-----|-----|-------|
| Train | 18,167,425 | 683,753 | 17,483,672 | 1:25.6 |
| Val | 956,181 | 35,987 | 920,194 | 1:25.6 |

**Pretrain (95/5) - Balanced (DDB negative sampling):**
| Split | Total | Pos | Neg | Ratio |
|-------|-------|-----|-----|-------|
| Train | 1,367,504 | 683,752 | 683,752 | 1:1 |
| Val | 71,974 | 35,987 | 35,987 | 1:1 |

**Finetune (90/10):**
| Split | Total | Pos | Neg | Ratio |
|-------|-------|-----|-----|-------|
| Train | 511,104 | 58,989 | 452,115 | 1:7.7 |
| Val | 56,791 | 6,555 | 50,236 | 1:7.7 |

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
