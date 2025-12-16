# Analysis: Deep Learning vs Classical ML Performance

## Executive Summary

On the provided PPI prediction task, **XGBoost significantly outperforms the Deep Learning (DL) models (v2, v3, v4)**.
*   **XGBoost**: Test AUROC **0.798**, Test AUPRC **0.196**
*   **Best DL (v3)**: Test AUROC **0.769**, Test AUPRC **0.103**

This indicates that the DL models are suffering from **poor generalization**, likely due to overfitting to sequence-specific details that do not carry over to the test set, whereas the mean-pooled ML models capture more robust, global interaction signals.

## 1. Metric Comparison

All models use the **exact same data splits** (`finetune_train.csv`, `finetune_val.csv`, `test.csv`), ensuring a fair comparison.

| Metric | XGBoost | Random Forest | v3 (DL Best) | v4 (DL) | v2 (DL Ablation) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test AUROC** | **0.798** | 0.765 | 0.769 | 0.763 | 0.748 |
| **Test AUPRC** | **0.196** | 0.165 | 0.103 | 0.098 | 0.085 |
| **Val AUROC** | 0.963 | 0.936 | ~0.96 | ~0.96 | ~0.89 |
| **Gen. Gap (AUROC)** | -0.165 | -0.171 | **-0.191** | **-0.197**| -0.142 |

*   **Generalization Gap**: All models show a massive drop from Validation (~0.96) to Test (~0.77-0.80). This confirms a **strong distribution shift** or **data leakage** in the train/val splitting strategy (likely random split vs. distinct proteins/clusters in test).
*   **AUPRC Discrepancy**: XGBoost achieves nearly **2x the AUPRC** of DL models. Given the low positive prevalence (~2.5%) in the test set, XGBoost provides much better "lift" (ranking positives at the very top).

## 2. Root Cause Analysis

### A. Inductive Bias: Global vs. Local
*   **ML Models (XGB/RF)**: Use **Global Mean Pooling**. They see the *average* properties of Protein A and Protein B. This forces the model to learn simple, robust heuristics (e.g., "Protein A is generally hydrophobic and Protein B is charged").
*   **DL Models (v3)**: Use **Cross-Attention**. They attempt to find specific *residue-residue* interactions.
    *   **Failure Mode**: In a small/medium data regime with a distribution shift, the DL models likely "hallucinate" specific interactions or memorize motif pairings seen in training that simply aren't present or relevant in the test set proteins.
    *   **Evidence**: v2 (which removes the self-attention encoder but keeps cross-attention) performs the worst (0.085 AUPRC), suggesting that *learning* the interaction from scratch via attention is difficult and prone to noise on this dataset.

### B. Distribution Alignment (Evaluation Setting)
*   **Observation**: The DL evaluation logs explicitly state: `Distribution alignment disabled by config; using raw logits`. which is confirmed in `configs/v3.yaml` (`apply_distribution_alignment: false`).
*   **Impact**:
    *   This **does not affect AUROC/AUPRC** (rank metrics).
    *   It **drastically affects F1/Recall/Precision** because the DL models (trained with `pos_weight=1.5` or balanced sampling) output logits calibrated to a ~50% prior, whereas the test set has ~2.5% prevalence.
    *   XGBoost (trained with `scale_pos_weight`) also faces this, but its probability outputs might be naturally better calibrated or its decision boundary is more robust.
    *   *Note*: The poor F1/MCC scores for DL in the table (F1 ~0.15) are largely due to this misalignment, but the AUROC/AUPRC gap is structural.

### C. Overfitting Complexity
*   **v3/v4 Parameters**: ~9M trainable parameters.
*   **Dataset**: Finite number of protein pairs (only ~4.8k positives in test).
*   The transformer architecture is likely **over-parameterized** for the amount of *distinct* interaction patterns available in the training data, leading it to memorize noise. XGBoost's tree-based structure with max depth and feature subsampling acts as a stronger regularizer for tabular-like embeddings.

## 3. Recommendations

1.  **Trust ML Baselines**: For this specific dataset and embedding type (ESM-3), the mean-pooled XGBoost model is the superior production candidate. It is faster, lighter, and more accurate.
2.  **Regularize DL Models**:
    *   If DL is required (e.g., for interpretability or future scaling), increase regularization (dropout > 0.2, weight decay > 0.01).
    *   Try **Mean Pooling** in the DL model (removing cross-attention) to see if it matches XGBoost. This would isolate whether the issue is the *attention mechanism* or the *optimization landscape*.
3.  **Fix Evaluation Calibration**: Enable `apply_distribution_alignment: true` in `configs/v3.yaml` to get meaningful F1/Accuracy numbers, though this won't fix the ranking (AUROC/AUPRC).

##

Highest Leverage (All v3/v4/v5)

  - Make validation match test: finetune_val.csv is ~11.5% positive but
    test.csv is ~2.5%, and test positives are almost all unseen edges; create
    a “realistic”/harder val split (same prevalence + edge-holdout) and point
    data_config.finetune.valid_csv to it.
  - Match XGBoost’s positive weighting: set finetune_config.loss.pos_weight:
    7.66 (≈ n_neg/n_pos from finetune_train.csv) and sweep {3, 5, 7.66, 10}; keep
    label_smoothing small (0–0.01).
  - Regularize/slow finetune: add finetune_config.max_grad_norm:
    1.0, increase model_config.regularization.
    {dropout,token_dropout,cross_attention_dropout,stochastic_depth} (e.g.
    0.2/0.2/0.15/0.15), and use a more conservative unfreeze schedule (freeze
    cross_attention. + encoder. longer).

  v3

  - Reduce capacity (edge-generalization): model_config.d_model: 256, n_heads: 8,
    encoder_layers: 1, cross_attn_layers: 1.
  - Remove “free” head flexibility: give the head weight decay (in
    finetune_config.optimizer.param_groups set head.weight_decay: 0.01 instead of 0.0)
    and consider lowering head.lr slightly.

  v4 / v5

  - Push toward the more “global” inductive bias: try model_config.cross_attn_layers:
    0 (or 1) in configs/v4.yaml/configs/v5.yaml.
  - For v5 specifically (contact-map overfit risk): lower
    data_config.max_sequence_length to 384–512, and shrink model_config.pair_dim/
    cnn_dim (e.g. 64/32), plus freeze map_builder. + contact_cnn. for the first few
    epochs.

  Note: in this repo’s current trainer, loss.type and use_class_weights don’t affect
  training; the knobs that matter are pos_weight, label_smoothing, and l1_lambda in
  finetune_config.loss.
