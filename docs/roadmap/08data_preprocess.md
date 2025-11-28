先帮你把现状翻一下：

* 现在有：

  * 69,343 个 **pos**
  * 用 `topo:low_score:random=1:1:1` 做了 refine neg → 得到一套 **pretrain 数据集**
* 另外还有：

  * 一个 **balanced 的纯 tmp-tmp test set**（已经定好，最好当“神圣不可碰”的最终测试集）
  * 一些 **旧的 tmp-related pairs**（至少一端是 tmp）

你现在的问题本质上是：

1. 怎么做 **去泄露（leakage removal）**？
2. 需不需要把“原来那套 tmp-related 数据”和“新从 STRING 抓到的 69k pos”合起来，重新生成 refined neg，让 **所有 tmp 相关的已知正样本都用上**？

我会建议：**是的，要做与 test 的去重；是的，值得合并所有 tmp-related positives 后重新生成 refined negatives**。下面按步骤说怎么搞。

---

## 一步一步怎么处理（推荐流程）

### Step 0：先定一个“总目标图景”

我按你之前设计的 pipeline 来假设目标：

* **Pretrain 阶段**：

  * 用尽可能多的 **tmp-related PPI（pos）** + 高质量 neg
  * 任务可以是“二分类 / 打分”
* **Finetune 阶段**：

  * 可能在更贴近你目标分布的子集上（比如特定子网络、实验数据集）
* **Test 阶段**：

  * 已有的 **balanced 纯 tmp-tmp test set**
  * 希望它对 train/pretrain 是 **完全独立**，防止 inflate 指标

基于这个目标，下面是更细的操作步骤。

---

### Step 0: Standardize Protein IDs to UniProt Format

Since `tmp_ppi_refined.csv` contains Ensembl protein IDs (ENSP...), we need to map them to UniProt IDs first.

**Why this is needed:**
- The `tmp_ppi_refined.csv` file from STRING pipeline contains ~7,454 ENSP IDs (for non-TMP partner proteins)
- The legacy dataset `20250724_train_dataset_at_least_one_membrane.csv` already has UniProt IDs
- We need consistent UniProt IDs for all proteins before merging datasets

**Steps:**

1. **Export ENSP IDs for mapping:**

   ```bash
   python -m src.data_process.export_ensembl_ids \
       --input data/full_TMP/processed/tmp_ppi_refined.csv \
       --output data/full_TMP/processed/ensembl_ids_to_map.txt
   ```

   This will extract 7,454 unique ENSP IDs and save them to a text file (one per line).

2. **Manual mapping via UniProt web interface:**

   - Visit: https://www.uniprot.org/id-mapping
   - Upload: `data/full_TMP/processed/ensembl_ids_to_map.txt`
   - Configure mapping:
     - **From**: `Ensembl_Protein`
     - **To**: `UniProtKB`
   - Submit and wait for results
   - Download results as **TSV format**
   - Save to: `data/full_TMP/processed/ensembl_to_uniprot_mapping.tsv`

3. **Apply mapping to CSV (future script):**

   ```bash
   python -m src.data_process.apply_ensembl_mapping \
       --input data/full_TMP/processed/tmp_ppi_refined.csv \
       --mapping data/full_TMP/processed/ensembl_to_uniprot_mapping.tsv \
       --output data/full_TMP/processed/tmp_ppi_refined_uniprot.csv \
       --cache data/full_TMP/processed/cache/ensembl_to_uniprot_mapping.pkl
   ```

   Note: The `apply_ensembl_mapping` script will be implemented after you obtain the mapping results.

**Expected outcome:**
- All proteins in `tmp_ppi_refined_uniprot.csv` will use UniProt IDs where possible
- Unmapped ENSP IDs will be kept as-is (to avoid data loss)
- Mapping will be cached for future use

---

### Step 1: Build a "Master Positives" Consolidated Set

Merge all positive samples from different sources, deduplicate, and remove leakage.

1. **Load data:**

   * From STRING + refine pipeline: **69,343 pos** (rows where `isInteraction==1` in `tmp_ppi_refined_uniprot.csv`)
   * From legacy dataset: **tmp-related pairs** (at least one TMP) with `label==1` in `data/full_TMP/processed/20250724_train_dataset_at_least_one_membrane.csv`

2. **Standardize format:**

   * Map all IDs to **unified ID type** (UniProt)
   * Canonicalize pairs so `(A,B)` and `(B,A)` are treated as the same:

     ```python
     pair = tuple(sorted([protA, protB]))
     ```

3. **Merge and deduplicate** to create **Master Positive Set**:

   * Each row: one pair + optional metadata (source: `string_highconf` / `legacy`, etc.)

> **Purpose**: All subsequent negative sampling, pretrain/finetune splitting will be based on this master positives set to avoid treating known positives as negatives.

output file: `data/TMP_protein/raw/master_tmp_positives.csv`

---

### Step 1.1: Split out the finetune positive set ✓ DONE

**Implementation**: `src/data_process/build_positives/split_finetune_positives.py`

Splits `master_tmp_positives.csv` into:
- `data/TMP_protein/processed/finetune_positives.csv`: 43,219 pairs where **both** proteins are TMP (from membrane_list.csv)
- `data/TMP_protein/processed/pretrain_positives.csv`: total 144,716 pairs preserved(TMP-TMP + TMP-X)

**Results**:
- ✓ All finetune pairs: 100% both proteins in TMP list
- ✓ All pretrain pairs: 144,716 pairs preserved(TMP-TMP + TMP-X)
- ✓ Total: 144,716 pairs (100% preserved)

---

### Step 2：对照 **纯 tmp-tmp test set** 做去泄露

这里分两个层级：

#### 2.1 Pair-level 去泄露（必须做）

* 从 test set 提取所有 test pairs（规范化成 `sorted(A,B)`）
* 在 **Master Positive Set** 里 **删除所有与 test pair 完全相同的 pair**

这样保证：

* **没有 train/pretrain pair 和 test pair 完全一样**
* 这是最基本的 leakage removal，必须保证。

#### 2.2 Protein-level 去泄露（看你要哪种泛化设置）

这里其实是一个 **实验设定问题**：

1. **Relaxed 设置（更常见）：**

   * 只禁止 pair 重复
   * 允许某个蛋白在 train/test 都出现，但 partner 不同
   * 适合评估“对已知蛋白的新 pair 的泛化能力”

2. **严格 OOD 设置：**

   * test set 里出现的任何蛋白 **在 pretrain+finetune 中都不出现**
   * 适合评估“对完全新蛋白的泛化能力”，但数据利用率会下降很多

我建议你这样玩：

* **主线实验**：

  * 采用 **Pair-level no leakage**（只移除 pair 重叠），
  * 保留蛋白级别交叉（现实中你确实会同时对很多已知蛋白做新 pair 预测）。
* **附加分析 / Ablation**：

  * 可以另外构造一个 “protein-disjoint test” 子集（所有蛋白在 train/val 中强制不出现），
  * 用同一个模型评估下这个 OOD 能力。

所以现在先按 **pair-level** 去泄露就足够。

---

### Step 3：基于“去泄露后的 Master Positives”重新生成 refined negatives

这一点你问得非常关键：**已经生成过一次 refined neg，要不要重来？**

答案：**最好重来一次**，原因：

* 你现在的 refined neg 是基于 “69,343 pos” 那一小块构造的；
* 当你把旧数据中额外的 tmp-related pos 合进来后：

  * 其中一部分 pair 可能原本被你当作 negative 候选（topo / low-score / random）；
  * 如果不重生成，就会出现 “某对 pair 在 A 数据库里是 pos，在 B 数据库里被当作 neg” 的 label noise；
* 重新跑 refine，可以保证：

  * **final negatives 与新的全集 positives 完全不重叠**；
  * neg pool 的拓扑/打分分布对所有 tmp-related 区域都是一致定义的。

推荐做法：

1. 用 **去泄露后的 Master Positive Set**（仍然是 all tmp-related pos）作为 `refine_negatives.py` 的输入：

   * `--input` 指向你新合并去重后的 positives CSV
2. `refine_negatives` 内部逻辑不变：

   * 从 cache 中的 full interactions 重新算 degree；
   * 重建 topo/low-score/random 三个候选池；
   * 根据 `neg_ratio` + 策略权重重新采样。

如果你已经决定偏向 hard neg，可以顺便调整策略配比，比如：

* `topo:low_score:random = 0.5 : 0.4 : 0.1` 或
* `topo:low_score:random = 0.7 : 0.3 : 0.0`（真正只用 topo+low-score 的版本，作为新实验线）

**Implementation**: `src/data_process/string_tmp_fetch/refine_negatives.py`

After running the initial pipeline, you can refine negatives using topology-based sampling for higher quality:

```bash
python -m src.data_proccess.string_tmp_fetch.refine_negatives \
    --input data/full_TMP/processed/tmp_ppi_train.csv \
    --cache-dir data/full_TMP/processed/cache \
    --output data/full_TMP/processed/tmp_ppi_refined.csv \
    --neg-ratio 10 \
    --degree-percentile 75 \
    --low-score-range 0.15 0.2 \
    --seed 42
```

**This generates three types of negatives** (1:1:1 ratio by default):
1. **Topology-based**: High-degree hub proteins (≥75th percentile connectivity)
2. **Low-score**: STRING interactions with score 0.15-0.2
3. **Random**: Standard Cartesian product sampling

**Fallback**: If topology/low-score insufficient, random is upsampled to ≥50% of total.

**Output**:
- `tmp_ppi_refined.csv`: Refined training dataset
- `tmp_ppi_negative_stats.txt`: Detailed statistics report

---

### Step 4：再做一次稳定的 **train/val 划分**: refer to https://arxiv.org/abs/2103.16370

**existing files**:
- pair-level no overlap: data/TMP_protein/processed/pair-no-leakage/finetune.csv
data/TMP_protein/processed/pair-no-leakage/pretrain.csv
data/TMP_protein/processed/pair-no-leakage/test_balanced.csv

- protein-level no overlap: data/TMP_protein/processed/protein-no-leakage/finetune.csv
data/TMP_protein/processed/protein-no-leakage/pretrain.csv
data/TMP_protein/processed/protein-no-leakage/test_balanced.csv

## 1. 我们先约定几个数据集合

结合你现在的数据情况，定义 3 个“池子”：

1. **$D_\text{pre}^\text{all}$**：(`pretrain.csv`)

   * 所有 **含 TMP** 的 pairs（TMP–TMP + TMP–nonTMP），pos/neg 都要
   * 这是 **pretrain 用的大池子**，对应论文里第一阶段的 “imperfect long-tail training set”

2. **$D_\text{fine}^\text{all}$**：(`finetune.csv`)

   * 所有 **纯 TMP–TMP** 的 pairs（你现在的 23336 pos + 184352 neg）
   * 这是专门给 finetune（第二阶段 classifier 调整）的池子

3. **$D_\text{test}^{\text{TMP-TMP}}$**：(`test_balanced.csv`)

   * 你已经准备好的 **1:1 balanced TMP–TMP 测试集**
   * 只用于最终报告结果，不参与任何训练 / 校准

## 0. 现在这套数据的关键统计（作为后面公式和代码的基线）

根据你 `README` 里的说明：

**Pretrain（pretrain.csv）**

* Pos：144,678
* Neg：875,061
* Total：1,019,739
* 自然先验
  [
  \pi_\text{pre}^{\text{nat}} \approx \frac{144678}{144678+875061} \approx 0.142
  ]
* Neg 由 topo / low-score / random 三种策略各约 1/3 组成。

**Finetune（finetune.csv）**

* Pos：43,181
* Neg：263,895
* Total：307,076
* 自然先验
  [
  \pi_\text{fine}^{\text{nat}} \approx \frac{43181}{43181+263895} \approx 0.141
  ]
* Neg 的策略分布：topo 约 50%，low-score 约 40%，random 约 10%，偏向“hard neg”。

**Test（test_balanced.csv）**

* Pos：9,576
* 当前文件中只有正例（未来你评测 1:1 时会再配建 neg 池）。

重要的是：**这三份 csv 已经做过 pair 级去泄漏**，test 里的 pair 不会出现在 pretrain/finetune 中。

---

## 1️⃣ Stage 1：Pretrain（用 pretrain.csv 学通用 TMP 域表征）

### 1.1 训练前：从 pretrain.csv 划分 train / val

目标：在这个大约 102 万条的长尾数据上做“Stage1 representation learning”，对齐 DisAlign/Decoupling 那种先单独把表征学好、再调 classifier 的思路。([arXiv][2])

具体步骤（跟前版一样，只是数字换成现在的）：

```python
from sklearn.model_selection import train_test_split

pairs_pre, labels_pre = load_pretrain_csv("pretrain.csv")  # 144678 pos, 875061 neg
# 这里的 pair 已经不包含 test 里的 pair，无需再过滤

pre_pairs_train, pre_pairs_val, \
pre_y_train, pre_y_val = train_test_split(
    pairs_pre,
    labels_pre,
    test_size=0.1,            # 90% / 10%
    stratify=labels_pre,
    random_state=seed
)
```

大概的数量级：

* `D_pre_train`：约 918k pair，pos ≈ 130k，neg ≈ 788k
* `D_pre_val`：约 102k pair，pos ≈ 14k，neg ≈ 87k

记录一下 “自然先验”（只是 log 用，不改变方法）：

```python
N_pre_pos = (pre_y_train == 1).sum()
N_pre_neg = (pre_y_train == 0).sum()
pi_pre_nat = N_pre_pos / (N_pre_pos + N_pre_neg)   # ≈ 0.142
```

### 1.2 每个 epoch/batch 的采样逻辑（保持 1:3 pos:neg）

延续前面的方法论：
Stage1 仍然用一个 **BalancedBatchSampler** 做“轻度重采样”，让每个 batch 内的 pos 比例提升到 **1 : 3**（自然先验是 1 : 6 左右），梯度更稳定，但不是极端 re-weight。

正负索引从 `D_pre_train` 中分出来：

```python
pretrain_dataset = PretrainDataset(pre_pairs_train, pre_y_train, esm3_embeddings)

pos_idx_pre = [i for i, y in enumerate(pre_y_train) if y == 1]
neg_idx_pre = [i for i, y in enumerate(pre_y_train) if y == 0]
```

BalancedBatchSampler（和之前完全同一个类，只是现在你心里知道它背后是 14 万 vs 87.5 万的真实规模）：

```python
import math
import numpy as np
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_size, neg_pos_ratio=3):
        self.pos_indices = np.array(pos_indices)
        self.neg_indices = np.array(neg_indices)
        self.batch_size = batch_size
        self.neg_pos_ratio = neg_pos_ratio

        self.pos_per_batch = max(1, batch_size // (1 + neg_pos_ratio))
        self.num_batches = math.ceil(len(self.pos_indices) / self.pos_per_batch)

    def __iter__(self):
        rng = np.random.default_rng()
        pos_perm = rng.permutation(self.pos_indices)
        pos_ptr = 0
        n_pos = len(pos_perm)

        for _ in range(self.num_batches):
            start = pos_ptr
            end = min(start + self.pos_per_batch, n_pos)
            batch_pos = pos_perm[start:end]
            pos_ptr = end
            if len(batch_pos) == 0:
                pos_perm = rng.permutation(self.pos_indices)
                pos_ptr = 0
                start = 0
                end = min(self.pos_per_batch, len(pos_perm))
                batch_pos = pos_perm[start:end]
                pos_ptr = end

            n_neg_needed = self.neg_pos_ratio * len(batch_pos)
            batch_neg = rng.choice(self.neg_indices, size=n_neg_needed, replace=True)

            batch_indices = np.concatenate([batch_pos, batch_neg])
            rng.shuffle(batch_indices)
            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches
```

构建 pretrain DataLoader（可以 batch 大一点，比如 512）：

```python
from torch.utils.data import DataLoader

pretrain_sampler = BalancedBatchSampler(
    pos_idx_pre,
    neg_idx_pre,
    batch_size=512,
    neg_pos_ratio=3      # 1:3，方法不变，只是自然先验从 1:极大→1:6 换成现在的 1:6
)

pretrain_loader = DataLoader(
    pretrain_dataset,
    batch_sampler=pretrain_sampler,
    num_workers=...,
    pin_memory=True
)
```

每个 pretrain epoch 的训练 loop：

```python
for epoch in range(num_epochs_pretrain):
    model.train()
    for batch in pretrain_loader:
        emb_a, emb_b, labels = batch  # ESM3 embedding + label
        logits = model(emb_a, emb_b)
        loss = bce_with_logits(logits, labels)  # 不加 pos_weight，和之前一样

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # epoch 结束，用 pre_val 做一次 eval（不做 DA，只看 AUROC / PR-AUC 收敛）
```

Pretrain 阶段的目标仍然是：在这 100 万级别的含 TMP 交互（+三种 neg 策略）上，学到一个稳定的 pair-wise 表征和初始 classifier 头，为 Stage2 做铺垫。([arXiv][2])

---

## 2️⃣ Stage 2：Finetune（用 finetune.csv + DA 对齐到 TMP 任务）

这里沿用“decouple rep/head + DA”的方法论：Stage1 的 encoder 冻结，Stage2 只调 head 并做 Distribution Alignment 校准。([arXiv][1])

### 2.1 finetune train/val 划分（在 finetune.csv 上）

从 `finetune.csv` 读取所有 TMP-focused pair：

```python
pairs_fine, labels_fine = load_finetune_csv("finetune.csv")  # 43181 pos, 263895 neg

fine_pairs_train, fine_pairs_val, \
fine_y_train, fine_y_val = train_test_split(
    pairs_fine,
    labels_fine,
    test_size=0.2,        # 80% / 20%
    stratify=labels_fine,
    random_state=seed
)
```

大概数量级：

* `D_fine_train`：约 245k pair，pos ≈ 34k，neg ≈ 211k
* `D_fine_val`：约 61k pair，pos ≈ 9k，neg ≈ 53k

自然先验（TMP-TMP 或 TMP-involving 的 hard-neg 环境）：

```python
N_fine_pos = (fine_y_train == 1).sum()
N_fine_neg = (fine_y_train == 0).sum()
pi_fine_nat = N_fine_pos / (N_fine_pos + N_fine_neg)   # ≈ 0.141
```

这和 pretrain 的 0.142 很接近，说明整个 TMP 域的数据在两个 stage 上先验是一致的，只是 finetune 明显偏向“hard negative”。

### 2.2 finetune 的每个 epoch/batch 采样逻辑（仍然 1:3）

继续用同一个 BalancedBatchSampler，只是现在 index 换成 finetune 的：

```python
fine_dataset = FinetuneDataset(fine_pairs_train, fine_y_train, esm3_embeddings)

pos_idx_fine = [i for i, y in enumerate(fine_y_train) if y == 1]
neg_idx_fine = [i for i, y in enumerate(fine_y_train) if y == 0]

fine_sampler = BalancedBatchSampler(
    pos_idx_fine,
    neg_idx_fine,
    batch_size=256,
    neg_pos_ratio=3    # 1:3，延续前面的方法论
)

fine_loader = DataLoader(
    fine_dataset,
    batch_sampler=fine_sampler,
    num_workers=...,
    pin_memory=True
)
```

这里的逻辑不变：

* 每个 epoch 约等于“让全部 finetune 正例扫一遍 + 重采样相同比例（×3）的 hard neg”；
* 不强制每个 neg 都在每个 epoch 被用到。

### 2.3 冻结 encoder，只训 head

方法论不变：严格 decouple rep vs classifier。([arXiv][2])

```python
# 假设 model = Encoder + Head
for p in model.encoder.parameters():
    p.requires_grad = False

optimizer_head = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr_head
)
```

训练 loop（未加 DA）：

```python
for epoch in range(num_epochs_finetune):
    model.train()
    for batch in fine_loader:
        emb_a, emb_b, labels = batch
        logits = model(emb_a, emb_b)
        loss = bce_with_logits(logits, labels)  # 仍然不加 pos_weight

        optimizer_head.zero_grad()
        loss.backward()
        optimizer_head.step()

    # epoch 末尾再做 DA + eval（下一小节）
```

---

## 3️⃣ Stage 2 的 DA：用经验分布 + target 先验做标量 bias 校准（方法不变）

仍然使用之前那套“估计模型输出的经验先验，再对齐到 target 先验”的简化 DisAlign 方案：([arXiv][1])

### 3.1 从 finetune-val 估计预测先验 q，计算 bias b

```python
import math
import torch

def estimate_da_bias(model, loader, device, pi_target=0.5):
    model.eval()
    n_samples = 0
    sum_p1 = 0.0

    with torch.no_grad():
        for batch in loader:
            emb_a, emb_b, labels = batch
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)

            logits = model(emb_a, emb_b)
            probs = torch.sigmoid(logits).view(-1)

            sum_p1 += probs.sum().item()
            n_samples += probs.numel()

    q1 = sum_p1 / n_samples   # 模型眼里的“正类先验”
    q0 = 1.0 - q1

    eps = 1e-6
    q1 = max(q1, eps)
    q0 = max(q0, eps)

    logit_target = math.log(pi_target / (1 - pi_target))  # target=0.5 → 0
    logit_pred   = math.log(q1 / q0)

    b = logit_target - logit_pred
    return b
```

这里 `pi_target=0.5` 仍然对应你最终希望“在一个 balanced TMP-TMP 评测设置下”决策边界不偏。
注意：我们用了 **模型预测概率的经验分布 q**，而不是直接用 1:3 这个采样比例，这是对 DisAlign 里“从预测分布估计 class prior 再做对齐”的简化。([arXiv][1])

### 3.2 带 bias 的 eval 函数（finetune-val 和 test 都用）

```python
def evaluate_with_da(model, loader, device, bias=0.0):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            emb_a, emb_b, labels = batch
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            labels = labels.to(device)

            logits = model(emb_a, emb_b)
            logits = logits + bias   # DA 核心：统一平移 logit
            probs = torch.sigmoid(logits).view(-1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.view(-1).cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    # 这里用 torchmetrics / sklearn 算 AUROC / PR-AUC / F1 等
    metrics = compute_metrics(all_probs, all_labels)
    return metrics, all_probs, all_labels
```

### 3.3 把 DA 接到 finetune 训练循环里

Finetune 主 loop 更新为：

```python
best_val_metric = -float("inf")
best_state = None
best_bias = 0.0
best_threshold = 0.5   # 之后在 val 上会 search

for epoch in range(num_epochs_finetune):
    model.train()
    for batch in fine_loader:
        ...
        # 同上，训练 head

    # === 每个 epoch 结束：先估计 DA bias ===
    b_da = estimate_da_bias(model, fine_val_loader, device, pi_target=0.5)

    # === 用 b_da 在 finetune-val 上评估 ===
    metrics, probs_val, labels_val = evaluate_with_da(
        model, fine_val_loader, device, bias=b_da
    )

    # 在 probs_val / labels_val 上 grid search 一个最佳阈值
    thr, val_score = find_best_threshold(probs_val, labels_val, metric="F1")

    if val_score > best_val_metric:
        best_val_metric = val_score
        best_state = copy_model_state(model)
        best_bias = b_da
        best_threshold = thr
```

最后，在 **TMP-TMP 的 test 设置** 上：

1. 加载 `best_state`；
2. 仍然用 `best_bias` 校准 logit；
3. 用 `best_threshold` 做最终 0/1 判别；
4. 报告 AUROC / PR-AUC / F1 / balanced accuracy 等指标。

（test 的负样本池如何构造，这属于“评测协议设计”，不改变方法论的话，你可以沿用之前计划的 balanced 1:1 TMP-TMP test 设定。）

---

## 4️⃣ 一句话总览这版“最终 plan”

1. **Stage1 Pretrain（pretrain.csv, 144,678 pos / 875,061 neg, 1:6）：**

   * 90/10 stratified 划分 train/val；
   * 在 train 上用 BalancedBatchSampler 做 batch 内 1:3 pos:neg 采样；
   * 用 BCE（无 pos_weight）训练全模型，val 上只监控 AUROC/PR-AUC/recall。

2. **Stage2 Finetune（finetune.csv, 43,181 pos / 263,895 neg, 1:6.1）：**

   * 80/20 stratified 划分 train/val；
   * 冻结 encoder，只训 head；
   * train 用同样的 1:3 BalancedBatchSampler；
   * 每个 epoch 结束：

     * 在 finetune-val 上估计预测先验 q → 计算 DA bias b；
     * 用 `logits + b` 在 val 上评估，并 grid search 阈值；
     * 选 val 指标最好的模型 + (b, threshold) 作为最终配置。

3. **Test（test_balanced.csv + 构造的 neg 池）：**

   * 固定 Stage2 找到的 best 模型、bias 和 threshold；
   * 在 1:1 TMP-TMP test 设置下报告最终指标。

这样，你就把新的“大号 pretrain/finetune 数据集统计”融合进了之前那套两阶段 + DA 框架里，方法论完全不变，只是所有先验、采样、epoch 规模都换成了现在这三个 csv 的真实数字。

[1]: https://arxiv.org/abs/2103.16370?utm_source=chatgpt.com "Distribution Alignment: A Unified Framework for Long-tail Visual Recognition"
[2]: https://arxiv.org/abs/1910.09217?utm_source=chatgpt.com "Decoupling Representation and Classifier for Long-Tailed Recognition"