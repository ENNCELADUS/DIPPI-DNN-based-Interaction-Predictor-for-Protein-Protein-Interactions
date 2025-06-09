**Part 1: Beginning to Average/Max Pooling**

* **幻灯片 1 (标题)**
    * “今天我们讨论使用ESM衍生特征 (ESM-derived features) 改进蛋白质-蛋白质相互作用 (Protein-Protein Interaction, PPI) 的预测。”

* **幻灯片 2 (特征提取 Feature Extraction)**
    * “首先，我们来看一下特征提取 (feature extraction)。我们利用ESMC进行蛋白质序列的嵌入 (embedding protein sequences)。”
    * “ESM Cambrian旨在创建能够捕捉蛋白质底层生物学特性的表征 (representations that capture the underlying biology of proteins)。”
    * “这相较于那些依赖生物学注释 (biological annotations) 或仅蛋白质序列本身的旧方法，是一个进步。”
    * “技术上，我们将来自b4ppi数据集 (b4ppi dataset) 的蛋白质序列输入到ESM-C预训练转换器 (ESM-C pre-trained transformer) 中。”
    * “每个蛋白质的输出是一个形状为L+2乘以960的嵌入 (embedding)，其中L代表序列长度 (sequence length)，960是嵌入维度 (embedding dimension)。额外的两个标记 (tokens) 是\[CLS]和\[EOS]。”

* **幻灯片 3 (数据处理 - 池化 (Pooling) 介绍)**
    * “一旦我们获得了这些可变长度的嵌入 (variable-length embeddings)，我们需要将它们转换为固定大小的表征 (fixed-size representations)，以用于下游的分类 (downstream classification)。”
    * “挑战在于将形状为\[L, 960]的嵌入转换为固定的\[960]向量 (vector)。”
    * “为实现这一目标，我们探索了多种池化策略 (pooling strategies)。”

* **幻灯片 4 (初始方法：平均/最大池化 Average/Max Pooling)**
    * “我们的初始方法包括使用平均池化 (average pooling) 或最大池化 (max pooling)。”
    * “具体来说，我们沿序列维度 (L+2) 应用了平均池化 (average pooling)。”
    * “这样做的原因是为了统一处理可变长度的输入 (variable-length inputs)，降低因嵌入矩阵 (embedding matrix) 过大而产生的计算复杂性 (computational complexity)，并保留通道信息 (channel-wise information)。”
    * “这使得每个蛋白质产生了一个压缩后大小为\[1, 960]的嵌入 (compressed embedding)。”
    * “然而，一个显著的缺点是损失了重要的位置和结构信息 (positional and structural information)。”

---

**Part 2: From MLP to Last**

* **Slide 15 (Multi-Layer Perceptron - MLP)** 


#### 🎯 核心目标与输入

* **目标**：预测两条蛋白质是否会发生相互作用
* **输入**：一对蛋白质（A 和 B）的嵌入向量，形状为 `(Batch, Length, 960)`，如 ESM 模型所生成

---

#### 🧩 组件一：增强型蛋白质编码器（Enhanced Protein Encoder）

* **作用**：分别处理蛋白质 A 和 B 的嵌入，生成上下文感知的精炼表示
* **关键模块**：

  * **RoPE 旋转位置编码**：比 v2 的可学习编码更稳定
  * **增强型 Transformer 层**：深度增加至 8–16 层，引入 GELU 与 LayerNorm
  * **层级式注意力池化**：为了将Transformer层可变长度的输出提炼成一个固定大小的向量，v4版本采用了一种‘层级式注意力池化 (Hierarchical Attention Pooling)’策略。该策略结合了全局注意力 (global attention)（以捕捉整体上下文）和局部注意力池化 (local attention pooling)
  * **压缩头**：将每个蛋白质嵌入压缩为 960 维 `refined_emb_a` 和 `refined_emb_b`

---

#### 🔁 组件二：交叉注意力相互作用模块（Cross-Attention Interaction）

* **输入**：`refined_emb_a` 和 `refined_emb_b`
* **双向交叉注意力**：A attends to B, B attends to A
  → 学习蛋白质之间的交互依赖关系（v2 不具备）
* **交叉注意力输出 → 前馈网络 (FFN)**：

  * 含残差连接
  * 输出：`interaction_embedding`（960 维）

当我们获得了蛋白质A和蛋白质B提炼后的嵌入之后，下一个关键步骤是对它们的相互作用进行建模。这由‘交叉注意力相互作用模块 (Cross-Attention Interaction Module)’来处理。

这里的一个关键特性是双向交叉注意力 (Bidirectional Cross-Attention)：蛋白质A的表征关注蛋白质B的表征，同时，蛋白质B的表征也关注蛋白质A的表征。这使得模型能够学习对相互作用至关重要的相互依赖性和上下文信息。这种显式的相互作用建模在v2版本中是没有的。”

“交叉注意力的输出随后由一个相互作用前馈网络 (Interaction Feed-Forward Network, FFN) 处理，该网络也包含残差连接 (residual connections)，最终形成一个单一的interaction_embedding向量（960维度），该向量封装了相互作用对的组合特征。

---

#### 🧠 组件三：增强型 MLP 解码器（Enhanced MLP Decoder）

* **任务**：将 `interaction_embedding` 映射为预测 logit
* **结构**：\[960 → 512 → 256 → 128 → 1]
* **每层包含**：

  * Linear → LayerNorm → GELU → Dropout → Residual
    → 提高训练稳定性与分类性能

这是一个专为二元分类 (binary classification) 设计的残差多层感知器 (residual Multi-Layer Perceptron, MLP)。它将960维的相互作用嵌入映射为一个单一的输出对数几率 (logit)，表明相互作用的可能性。其层结构为 [960 → 512 → 256 → 128 → 1]。”

“这个MLP中的每一层都经过精心构建，包含一系列操作：线性变换 (Linear transformation) → 归一化 (Normalization) → GELU激活函数 → Dropout（用于正则化 regularization）→ 以及一个残差连接 (Residual connection)。这种设计促进了稳定的训练和有效的分类。

---

#### ✅ v4 架构关键进展总结

* **从自监督到监督学习**：明确用于 PPI 分类
* **更强嵌入表示**：采用 RoPE + 深层 Transformer
* **显式交互建模**：双向交叉注意力机制
* **更佳特征提取与分类**：层级池化 + 残差 MLP 解码器
* **动态长度支持**：兼容不同长度蛋白质序列

---

实验结果摘要

文献对比 (Test1 数据集):

文献中的一个基于序列的Siamese网络模型 (Sequence-based Siamese network model) 报告其AUROC为0.68，AUPRC为0.68。

同文献中的XGBoost模型报告其AUROC为0.62，AUPRC为0.65。

我们的XGBoost模型表现:

XGBoost + 平均池化 (Average Pooling):

在Test1 (平衡数据集)上：AUROC为0.6441，AUPRC为0.6420 (根据最终表格则为0.6447)。Test1上随机基线的AUPRC为0.5000。

XGBoost + 掩码自编码器 (Masked Autoencoder, MAE) 嵌入:

在Test1上：AUROC为0.6636，AUPRC为0.6723 (根据最终表格则为AUPRC 0.6771，AUROC 0.6466)。

与XGBoost + 平均池化 (Average Pooling) 相比，MAE方法在两个数据集上的AUPRC均表现出改进。
我们最佳方法总结 (XGBoost + MAE on Test1, 根据最终表格):

AUROC: 0.6466 (注：幻灯片18中报告的XGBoost+MAE在Test1上的AUROC为0.6636)
AUPRC: 0.6771

---

* **Slide 20 (Conclusion & Future Directions)** 

  * "In conclusion, our XGBoost + MAE approach has shown competitive results. It outperforms the literature XGBoost baselines, although it achieves a slightly lower AUROC compared to the sequence-based RNN models from literature." 
  * "Among the traditional machine learning classifiers we tested, XGBoost demonstrated the best performance, effectively capturing nonlinear feature interactions." 
