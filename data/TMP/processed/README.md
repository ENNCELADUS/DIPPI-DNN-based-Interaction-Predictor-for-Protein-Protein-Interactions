## TMP Dataset Summary

| File | Positive | Negative | Total | Neg:Pos Ratio | Unique Proteins |
|------|----------|----------|-------|---------------|-----------------|
| `raw/pretrain.csv` | 719,740 | 18,403,866 | 19,123,606 | 25.6:1 | 18,177 |
| `processed/pretrain.csv` | 719,740 | 18,403,866 | 19,123,606 | 25.6:1 | 18,177 |
| `pretrain_train.csv` | generated from `processed/pretrain.csv` via global-TPPNI then protein split | generated | config-driven | fixed to 1:1 | generated |
| `pretrain_val.csv` | generated from `processed/pretrain.csv` via global-TPPNI then protein split | generated | config-driven | fixed to 1:1 | generated |
| `raw/finetune.csv` | 65,544 | 2,751,612 | 2,817,156 | 42.0:1 | 9,352 |
| `processed/finetune.csv` | 65,544 | 2,751,612 | 2,817,156 | 42.0:1 | 9,352 |
| `finetune_train.csv` | generated from `processed/finetune.csv` via global-TPPNI then protein split | generated | config-driven | normalized to match val ratio | generated |
| `finetune_val.csv` | generated from `processed/finetune.csv` via global-TPPNI then protein split | generated | config-driven | normalized to match train ratio | generated |
| `test.csv` | 4,788 | 183,909 | 188,697 | 38.4:1 | 1,282 |

> [!NOTE]
> - `processed/pretrain.csv` and `processed/finetune.csv` are raw stage corpora, not balanced training files.
> - Canonical train/validation files are produced by `python -m src.data_preprocess.prepare_tppni_datasets --config <config>`.
> - The preprocessing order is: clean raw positives -> build one global TPPNI pool per stage -> protein-level train/val split -> induce split datasets.
> - `test.csv` is fixed and not rewritten by TPPNI preprocessing.
> - `pretrain_train.csv` and `pretrain_val.csv` are score-downsampled to exact `1:1`.
> - `finetune_train.csv` and `finetune_val.csv` are score-downsampled to the same `neg:pos` ratio, using the smaller split-induced ratio as the target.

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
| Train | generated from global TPPNI + protein split | generated | generated | 1:1 |
| Val | generated from global TPPNI + protein split | generated | generated | 1:1 |

**Finetune generated datasets:**
| Split | Total | Pos | Neg | Ratio |
|-------|-------|-----|-----|-------|
| Train | generated from global TPPNI + protein split | generated | generated | same as val after normalization |
| Val | generated from global TPPNI + protein split | generated | generated | same as train after normalization |

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
