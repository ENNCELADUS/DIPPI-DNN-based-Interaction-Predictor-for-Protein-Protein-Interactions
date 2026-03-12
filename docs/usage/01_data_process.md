# Data preprocess usage

## Config-driven TPPNI preprocessing
```bash
cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate esm
python -m src.data_preprocess.prepare_tppni_datasets --config configs/v3.yaml
```

The CLI reads `data_config.preprocessing.tppni` from the config file.

- When `enabled: true`, it cleans the raw pretrain and finetune corpora, rebuilds
  TPPNI negatives, and writes the canonical split files consumed by training.
- Cleaning policy is enforced in code: drop missing IDs, drop self-loops,
  canonicalize undirected pairs, deduplicate within label, and error on
  conflicting labels.
- Train and validation splits use the full TPPNI set produced by the paper's
  pipeline for that split: bottom-`N` configuration-model nonedges, followed by
  CL3 filtering to keep only zero-L3 pairs. The resulting negative count is
  emergent, not user-specified.
- When `enabled: false`, the command is a no-op.
- When `force_rebuild: false`, the manifest skips redundant rebuilds.

## Finetune sampling behavior

Sampling config is resolved per stage by merging the global
`data_config.dataloader.sampling` block with an optional
`data_config.dataloader.finetune_sampling` override. In the current project
configs there is no `finetune_sampling` override, so finetune inherits the
global sampling settings.

- If finetune uses `strategy: "none"`, the loader is a normal shuffled
  `DataLoader`. There is no fixed per-batch class ratio. Batch composition
  follows the generated finetune dataset distribution in expectation.
- If finetune uses `strategy: "ohem"`, there are two stages:
  - Warmup stage: the trainer does not perform OHEM selection yet, but the
    dataloader still uses `StagedOHEMBatchSampler` warmup batches. If
    `warmup_pos_neg_ratio` is not set, it defaults to the generated train
    split's natural ratio `neg_count / pos_count`.
  - Mining stage: the dataloader yields candidate pools of size
    `pool_multiplier * batch_size`, approximately preserving the dataset class
    fraction. The trainer then scores that pool and keeps the hardest
    `batch_size` samples, subject only to the `cap_protein` constraint. The
    final optimized minibatch does not enforce a fixed class ratio.
- `training_config.finetune.loss.pos_weight` is applied only during warmup or
  when finetune sampling uses `strategy: "none"`. During OHEM mining, the
  trainer computes both selection loss and final optimized loss with
  `pos_weight = 1.0`, so the active finetune configs now keep `pos_weight: 1.0`
  for consistency with the training logic.

## Paper ratio note

The TPPNI paper does not recommend a universal fixed negative ratio such as
`7:1`. Its main control is the negative construction method, not a single class
ratio heuristic.

- In the paper's reported training corpus, there are `706,244` PPIs and
  `3,063,605` PPNIs, which is approximately `4.34:1` negatives to positives.
- For the random-negative ablation, the paper explicitly keeps the number of
  random negatives the same as the number of TPPNI samples, so the comparison
  isolates negative quality rather than changing the class count.
- Therefore, the current preprocessing does not force `1:1`, does not preserve
  the raw finetune ratio, and does not prescribe `7:1`. It uses the full TPPNI
  set that emerges from the configuration-model plus CL3 pipeline.

## Embed Usage

```bash
python src/embed/embed.py --input "$TMP_CSV" --output "$TMP_NPZ"
```
