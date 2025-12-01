# Data preprocess usage

## Train/Val split
```bash
cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions && python src/data_preprocess/split_data.py --input data/TMP/processed/pretrain.csv --train-ratio 0.95 --seed 42
```

```bash
cd /public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions && python src/data_preprocess/split_data.py --input data/TMP/processed/finetune.csv --train-ratio 0.90 --seed 42
```

## Embed Usage

```bash
python src/embed/embed.py --input "$TMP_CSV" --output "$TMP_NPZ"
```