## 总体思路

你现在有一串 **TMP 的 UniProt ID 列表**。目标是拿到：

* 所有 **包含至少一个 TMP** 的 PPI（TMP–TMP, TMP–nonTMP 全部要）
* 每条边附带一堆 **score “标签”**：`score, nscore, fscore, pscore, ascore, escore, dscore, tscore`
* 再按 **combined score（score）做置信度过滤**，只保留 “high confidence” 以上，比如 `score ≥ 0.7`（STRING 官方把 0.7 定义为高置信度，0.9 为最高置信度）([STRING][1])

STRING 里刚好有两个关键 API：

1. **`get_string_ids`**：先把 UniProt ID 映射成 STRING 自己的 ID
2. **`interaction_partners`**：返回你的蛋白集合与“所有 STRING 蛋白”的相互作用，这就满足了“至少一个端点在你的 list 里”([STRING][2])

## 整体 pipeline 小结（直接照这个做）

1. **准备 TMP UniProt 列表** `tmp_uniprots`
2. 用 `get_string_ids` 映射为 `tmp_string_ids`（指定 `species`）。([STRING][4])
3. 调用 `interaction_partners`：

   * `identifiers = "%0d".join(tmp_string_ids)`
   * `species = 你的物种 ID`
   * `required_score = 700` 或 `900`
   * 得到 `interactions_df`，包含 `score` 和各通道的分数列。([STRING][2])
4. 在 `interactions_df` 上：

   * 标记 `is_tmp_A`, `is_tmp_B` 和 `pair_type`（tmp-tmp / tmp-non_tmp）
   * 设置 `label = 1`
   * 如有需要再做二次 `score >= high_cutoff` 过滤
5. 单独构造一批 negative pairs（`label = 0`），和上面的 positive 合并，即为你的 **TMP 相关 PPI 训练集**。

[1]: https://string-db.org/cgi/info?utm_source=chatgpt.com "Info - STRING functional protein association networks"
[2]: https://string-db.org/help/api/ "API - STRING Help"
[3]: https://string-db.org/help/api/?utm_source=chatgpt.com "API - STRING Help"
[4]: https://version11.string-db.org/help/api/?utm_source=chatgpt.com "API - STRING Help"
[5]: https://string-db.org/cgi/download?utm_source=chatgpt.com "Downloads - STRING functional protein association networks"

---

## Refined Implementation Plan

**Task Summary**: Create a script to fetch TMP-related PPI training data from STRING database with stratified negative sampling.

**Files to Create/Edit**:
1. Create: `src/data_proccess/string_api.py` - API wrapper with batching & caching
2. Create: `src/data_proccess/fetch_tmp_ppi.py` - Main pipeline script
3. Update: `requirements.txt` - Add `requests` package

**Detailed Workflow**:

### Step 1: Data Loading & Mapping
- Load ~13,660 TMP UniProt IDs from `data/full_TMP/raw/membrane_list.csv`
- **Species handling**: Make it a CLI parameter (default=9606 for human) since we can't infer from UniProt IDs alone
- Map UniProt → STRING IDs in batches (e.g., 500 IDs per request)
- Cache mapping to `data/full_TMP/processed/uniprot_to_string_cache.pkl`

### Step 2: Fetch Positive Interactions
- Call `interaction_partners` API for all TMP STRING IDs
- Use `required_score=700` (confidence ≥ 0.7)
- Store result with all score channels
- Get partner proteins (both TMP and non-TMP)
- These are **positive samples** (isInteraction=1)

### Step 3: Fetch Low-Score Interactions (for hard negatives)
- Call `interaction_partners` with lower threshold (e.g., `required_score=0`)
- Filter for `0.15 <= score < 0.2` locally
- These are candidates for **hard negatives**

### Step 4: Generate Negative Samples
- Build candidate set: all TMP-X pairs where X appeared in any interaction
- Remove all positives (score ≥ 0.7)
- **Random negatives**: Sample 9N pairs from remaining candidates
- **Hard negatives**: Sample N pairs from low-score candidates (0.15 ≤ score < 0.2)

### Step 5: Output Generation
- **Main file**: `data/full_TMP/processed/tmp_ppi_train.csv`
  - Columns: `uniprotID_A,uniprotID_B,isInteraction`
  - Contains: N positives + 9N random negatives + N hard negatives (total 11N rows)
- **Hard negatives file**: `data/full_TMP/processed/tmp_ppi_hard_negatives.csv`
  - Same schema, only the N hard negative samples (for reference/analysis)

**CLI Interface**:
```bash
python -m src.data_proccess.fetch_tmp_ppi \
    --input data/full_TMP/raw/membrane_list.csv \
    --output data/full_TMP/processed/tmp_ppi_train.csv \
    --species 9606 \
    --confidence 0.7 \
    --neg-ratio 10 \
    --hard-neg-ratio 0.1 \
    --batch-size 500
```

**Key Implementation Details**:
- Batching: 500 IDs per API call with 1-2s delay between requests
- Caching: Save intermediate results (STRING ID mapping, raw interactions)
- Error handling: Log unmapped IDs, API failures
- Progress tracking: Use `tqdm` for long operations
- Logging: INFO level for progress, WARNING for issues

**Species parameter**: Since UniProt IDs don't indicate species: Default to human (9606) with optional override

### Step 6: Fetch sequence info for all the proteins in pretrain/finetune/test
- Run a script to get the list of all proteins.
- Use legacy `data/membrane_protein/unique_proteins_with_sequences.csv` to get existing sequence first.
- Use api of STRING to get the rest sequences.