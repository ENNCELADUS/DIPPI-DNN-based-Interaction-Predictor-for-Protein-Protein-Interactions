## 0. 先帮你捋一下现有架构（基准）

根据你给的 `v3.py` 现在的 interaction 大致是：

```python
# 1) SiameseEncoder: 各自 self-attn 编码
encoded_a = self.encoder(emb_a, lengths_a)  # (B, La, d)
encoded_b = self.encoder(emb_b, lengths_b)  # (B, Lb, d)

# 2) InteractionCrossAttention: L 层 CrossAttentionLayer
#    每一层都做：
#    - a_to_b: A 作为 query，B 作为 key/value，更新 A
#    - b_to_a: B 作为 query，A 作为 key/value，更新 B
#    - concat(A,B) → 让一个 CLS token 去 attend 两条序列，FFN 再更新 CLS
cls = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)  # (B, d)

# 3) MLPHead: 用 cls 表示互作
logits = self.output_head(cls)  # (B, 1)
```

也就是说，现在**互作表征是通过一个 learnable CLS token 来「读」两条序列**。

---

## PLM-Interact Model

How to prepare data (run in conda activate esm):

  mkdir -p data/TMP_protein/processed/pair-no-leakage/plm_interact

  python scripts/convert_pairs_to_sequences.py \
    --input_csv data/TMP_protein/processed/pair-no-leakage/split/finetune_train.csv \
    --seq_csv data/TMP_protein/processed/unique_proteins.csv \
    --output_csv data/TMP_protein/processed/pair-no-leakage/plm_interact/finetune_train.csv

  python scripts/convert_pairs_to_sequences.py \
    --input_csv data/TMP_protein/processed/pair-no-leakage/split/finetune_val.csv \
    --seq_csv data/TMP_protein/processed/unique_proteins.csv \
    --output_csv data/TMP_protein/processed/pair-no-leakage/plm_interact/finetune_val.csv

  python scripts/convert_pairs_to_sequences.py \
    --input_csv data/TMP_protein/processed/pair-no-leakage/test_balanced.csv \
    --seq_csv data/TMP_protein/processed/unique_proteins.csv \
    --output_csv data/TMP_protein/processed/pair-no-leakage/plm_interact/test_balanced.csv

  python scripts/convert_pairs_to_sequences.py \
    --input_csv data/TMP_protein/processed/pair-no-leakage/val_balanced.csv \
    --seq_csv data/TMP_protein/processed/unique_proteins.csv \
    --output_csv data/TMP_protein/processed/pair-no-leakage/plm_interact/val_balanced.csv

  Configure PLM-interact run:

  - Edit configs/plm_interact.yaml:
      - model_config.base_model_path: local ESM base (e.g., your offline facebook/esm2_t33_650M_UR50D).
      - model_config.checkpoint_path: local danliu1226/PLM-interact-650M-humanV12/pytorch_model.bin.
      - Paths under data_config already point to the converted CSVs above.
      - Set run_config.mode to eval_only (just evaluate) or finetune_eval (finetune then eval). For finetune, adjust
        train_config batch sizes/epochs for your A40.

  Run:

  python run_plm_interact.py configs/plm_interact.yaml

  - Logs land in logs/plm_interact/<mode>/<run_id>/.
  - Best checkpoint (if finetuning) in models/plm_interact/<run_id>/best_model.pth.
  - Metrics printed per split (test_balanced/test_realistic) using the Evaluator metrics list in config.

---

## 1. V4: 双向 cross-attention + 双序列 pooling（无 CLS）

这是你直接想到的方案，也是**最常见、最好实现的 baseline**。

### 架构思路

1. 保留现在的 `attn_a_to_b` / `attn_b_to_a`（A↔B 互注意），**只去掉 CLS 分支**；

2. 堆叠几层 cross-attn block 之后，得到：

   * `h_a' ∈ ℝ^{B×La×d}`，`h_b' ∈ ℝ^{B×Lb×d}`；

3. 用 mask 做 pooling：

   ```python
   rep_a_mean = masked_mean(h_a', mask_a)   # (B, d)
   rep_b_mean = masked_mean(h_b', mask_b)   # (B, d)
   rep_a_max  = masked_max(h_a', mask_a)    # (B, d)
   rep_b_max  = masked_max(h_b', mask_b)    # (B, d)
   ```

4. 最常见的组合方式是：

   ```python
   # 经典「匹配」特征组合
   h = torch.cat(
       [
           rep_a_mean,
           rep_b_mean,
           torch.abs(rep_a_mean - rep_b_mean),
           rep_a_mean * rep_b_mean,   # 逐元素乘
       ],
       dim=-1,
   )  # (B, 4d)
   ```

   或者如果你也想用 max-pool，就把 `rep_a_max / rep_b_max` 再拼进去，变成 `6d` 或 `8d` 也行。

5. 对应地，只要把 `MLPHead.input_dim` 改成 `4 * d_model`（或 6d / 8d），其它结构完全不动。

### 优点 / 缺点（相对 CLS）

* 👍 **更「解释得清」**：`rep_a_mean`、`rep_b_mean` 确实是「互作上下文下的蛋白整体表征」，方便后面分析。
* 👍 对极长序列时，比单个 CLS 更不容易「吃不下」信息。
* 👍 实现最简单：只改 interaction 模块 & MLP 输入维度。
* 👎 表征维度变大（4d/6d/8d），MLP 参数略增。

### 在你代码里的改法建议

* 新建一个 `InteractionCrossAttentionNoCLS`：

  * 复制 `CrossAttentionLayer`，**删掉 cls 相关的 norm/attn/ffn 分支**，让它只返回 `(h_a, h_b)`。
  * 新的 `InteractionCrossAttentionNoCLS` 堆叠这些 layer，最后做 pooling → 拼接 → 返回 `interaction_repr`。
* 在 `V3.__init__` 里用一个配置开关，比如（不要改现有 config 就先硬编码也行）：

  ```python
  if self.interaction_type == "cls":
      self.interaction = InteractionCrossAttention(...)
      mlp_input_dim = self.d_model
  elif self.interaction_type == "bi_pool":
      self.interaction = InteractionCrossAttentionNoCLS(...)
      mlp_input_dim = 4 * self.d_model
  ```

---

## 4. Token–token 相似度矩阵 + CNN（MatchPyramid 风格）

这是很多文本匹配、PPI 文本+序列工作会用到的**相互作用矩阵**架构。

### 架构思路

1. 先用 Siamese encoder 得到 `encoded_a`, `encoded_b`（或再加一层轻量 self-attn / cross-attn）；

2. 计算 pairwise 相似度矩阵：

   ```python
   # (B, La, d) · (B, d, Lb) → (B, La, Lb)
   sim = torch.einsum("bid, bjd -> bij", encoded_a, encoded_b)
   # 或者 sim[i,j] = v^T tanh(Wa * a_i + Wb * b_j) 等更复杂的核
   ```

3. 把 `sim` 当成「单通道图像」送进小 CNN / Conv2d：

   ```python
   x = sim.unsqueeze(1)  # (B, 1, La, Lb)
   x = conv_block(x)     # 几层 2D Conv + Pool
   x = torch.flatten(x, 1)  # (B, D')
   logits = MLP(x)
   ```

### 优点 / 缺点

* 👍 非常直观：矩阵中高值区域可以解释为「潜在接触 patch」；
* 👍 对局部 pattern（比如连续若干残基与另一条上的 patch 互作）表达能力强；
* 👎 计算复杂度 O(La*Lb)，长序列时可能很贵，需要截断 / 下采样；
* 👎 和你现在的 transformer pipeline 有点「风格」不同，需要额外 CNN 模块。

个人建议：
可以先在**较短序列的子集**上试一个小 CNN 版本，看看是否有明显增益，如果有，再考虑推广。

