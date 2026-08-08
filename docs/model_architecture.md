# DINOv2 + INP-Former 模型架构与训练推理原理

## 1. 总体架构

```
                        训练阶段 (无监督)                         推理阶段
                    ┌─────────────────────────┐          ┌─────────────────────────┐
                    │                         │          │                         │
  输入图像          │   DINOv2 ViT-B/14      │          │   DINOv2 ViT-B/14      │
 [B, V=5, 3, H, W] │   (冻结, 不可训练)      │          │   (冻结, 不可训练)      │
        │          │         │               │          │         │               │
        ▼          │         ▼               │          │         ▼               │
 ┌──────────────┐  │  ┌───────────────────┐  │          │  ┌───────────────────┐  │
 │ 5 个视角图像  │──┼─▶│ cls_token [B,V,D] │  │          │  │ cls_token [B,V,D] │  │
 └──────────────┘  │  │ patch  [B,V,N,D]  │  │          │  │ patch  [B,V,N,D]  │  │
                   │  └────────┬──────────┘  │          │  └────────┬──────────┘  │
                   │           ▼              │          │           ▼              │
                   │  ┌───────────────────┐  │          │  ┌───────────────────┐  │
                   │  │ ViewPatchEncoder  │  │          │  │ ViewPatchEncoder  │  │
                   │  │  (Transformer)    │  │          │  │  (Transformer)    │  │
                   │  │  (可训练)         │  │          │  │  (可训练)         │  │
                   │  └────────┬──────────┘  │          │  └────────┬──────────┘  │
                   │           ▼              │          │           ▼              │
                   │  ┌───────────────────┐  │          │  ┌───────────────────┐  │
                   │  │ Normalizing Flow  │  │          │  │ Normalizing Flow  │  │
                   │  │  (可训练)         │  │          │  │  (可训练)         │  │
                   │  └────────┬──────────┘  │          │  └────────┬──────────┘  │
                   │           ▼              │          │           ▼              │
                   │   NLL 损失 → 反向传播    │          │   NLL → 异常得分 [0,1]  │
                   │   patch_score → 丢弃     │          │   patch_score → mask    │
                   └─────────────────────────┘          └─────────────────────────┘
```

## 2. 各模块详解

### 2.1 DINOv2 特征提取器（冻结）

DINOv2（Self-Distillation with No Labels v2）是 Meta 提出的自监督视觉 Transformer，在大规模无标注图像上预训练，具备强大的通用视觉特征表达能力。

**选择 ViT-B/14 的原因**：
- **768 维** patch 特征，兼顾表达能力和计算开销
- **patch_size = 14**，比 ViT-B/16 的空间分辨率更精细（同样 518×518 输入下得到 37×37 = 1369 个 patch）
- 预训练权重大小约 330MB

**提取的特征**：

```
输入: [B, 3, 518, 518]
  ↓
patch_embed: Conv2d(3 → 768, kernel=14, stride=14)
  ↓
[CLS] + 1369 patches + pos_embed
  ↓
Transformer Blocks (12 层, 12 heads)
  ↓
输出:
  cls_token:    [B, 768]          全局语义
  patch_tokens: [B, 1369, 768]    局部空间语义 (37×37 grid)
```

**多视角处理**：将 V=5 个视角展平到 batch 维度，一次前向提取所有视角特征，再恢复视角维度：

```
cls_tokens:     [B, V, 768]
patch_features: [B, V, 1369, 768]
```

**关键点**：DINOv2 权重完全冻结（`requires_grad=False`），不参与梯度计算和参数更新，仅作为固定的特征提取器。

### 2.2 ViewPatchEncoder（多视角 Transformer 编码器）

核心作用是**建模视角间的对应关系**和**视角内的空间结构**。

```
输入:
  patch_features: [B, V, N, 768]  (DINOv2 输出)
  cls_tokens:     [B, V, 768]

处理流程:
  1. 线性投影: 768 → 256 (d_model)
  2. 添加位置编码: [1, 1, N, 256] (可学习)
  3. 添加视角嵌入: [V, 256] (可学习, 区分不同视角)
  4. 展平: [B, V*N, 256]
  5. 拼接 CLS token: [B, 1+V*N, 256]
  6. Transformer Encoder (4 层, 8 heads)
  7. 提取输出:
     - cls_out:     [B, 256]          全局上下文
     - view_tokens: [B, V, 256]       每个视角的汇总表示
     - encoded:     [B, 1+V*N, 256]   完整编码（含 patch）
```

**为什么需要视角嵌入？**
同一零件的 5 个视角拍摄角度不同，模型需要知道每个 patch 来自哪个视角。视角嵌入（learnable embedding）让 Transformer 的 self-attention 能区分"同一视角内的空间邻居"和"不同视角间的对应区域"。

**可训练参数量**：约 4.85M（含 Normalizing Flow）

### 2.3 Normalizing Flow（可逆概率流）

Normalizing Flow 是本方案的**核心创新点**，用于学习正常特征的精确概率分布。

#### 基本思想

Normalizing Flow 通过一系列**可逆变换**将复杂分布（正常样本的编码特征）映射到简单的标准高斯分布 N(0, I)：

```
正常样本编码特征 x ──f₁──▶ ──f₂──▶ ... ──fₖ──▶ z ~ N(0, I)
                          (每层都是可逆的)
```

训练完成后：
- **正常样本** → 变换后的 z 接近标准高斯 → 高似然 → 低异常得分
- **异常样本** → 变换后的 z 偏离标准高斯 → 低似然 → 高异常得分

#### Flow 层结构

每层 Flow 由三个子模块组成，共堆叠 6 层：

```
┌─────────────────────────────────────────────────┐
│ ActNorm: 可逆仿射归一化                          │
│   y = (x - μ) / σ                               │
│   首个 batch 用数据均值/方差初始化 μ, σ            │
├─────────────────────────────────────────────────┤
│ Permute: 随机排列特征维度（增加表达能力）          │
│   y = x[perm]                                    │
├─────────────────────────────────────────────────┤
│ AffineCoupling: 仿射耦合变换                      │
│   x = [x₁, x₂]  (前半 / 后半)                   │
│   log_s, t = MLP(x₁)                             │
│   y₂ = x₂ * exp(log_s) + t                       │
│   y = [x₁, y₂]                                   │
└─────────────────────────────────────────────────┘
```

**数值稳定性措施**：
- `log_scale` 参数化（保证 scale > 0），clamp 到 [-3, 3]
- AffineCoupling 的 `tanh(scale) * 1.0` 限制（exp(1.0)≈2.7，更保守）
- `shift` clamp 到 [-3, 3]
- Coupling 层最后一层权重初始化为零（初始变换接近恒等映射）
- Flow 输入前加 LayerNorm 归一化
- **log_det 正则化惩罚**：防止雅可比行列式过大导致"体积膨胀"作弊

#### 两个独立 Flow

模型使用两个独立的 Normalizing Flow：

| Flow | 输入 | 作用 |
|------|------|------|
| `flow_cls` | CLS token [B, 256] | 建模全局正常分布 |
| `flow_view` | 视角 token [B*V, 256] | 建模视角一致性分布 |

### 2.4 Patch-level 异常定位

用于生成像素级的异常热力图（448×448 mask）：

```
1. patch_head: 线性层 + GELU 映射 patch 编码特征
2. 计算每个 patch 与 CLS 特征的余弦距离 → patch_score [B, V, N]
3. reshape 为 2D grid: [B, V, 1, n_h, n_w]
4. 双线性上采样到 448×448
5. 逐视角归一化到 [0, 255] → uint8 灰度 mask
```

**为什么用余弦距离？**
- 正常 patch 的特征方向与全局 CLS 一致 → 余弦相似度高 → 距离小
- 异常区域的特征方向偏离全局 → 余弦相似度低 → 距离大

## 3. 训练流程

### 3.1 训练目标

**无监督训练**：训练集（Train）由比赛方预先筛选，仅包含 50 个已见类别的正常样本（无异常标签）。模型无需区分正常/异常，只需学习"正常样本的特征分布是什么样的"。

> **注意**：Train 目录下所有样本均为正常零件，不需要也无法利用异常标签。这正是"无监督"的含义——模型只看到正常数据，不知道异常长什么样，通过拟合正常分布来间接检测异常。

```
正常样本 → DINOv2 → ViewPatchEncoder → Normalizing Flow → z ~ N(0, I)
                                                              │
                                              最大化 log p(z) + log |det J|
```

### 3.2 损失函数

```
L = λ_cls · NLL_cls + λ_view · NLL_view + λ_reg · Reg + λ_det · DetReg

其中:
  NLL_cls  = -E[log p(z_cls) + log |det(dz/dx)|]     CLS 的负对数似然
  NLL_view = -E[log p(z_view) + log |det(dz/dx)|]     视角的负对数似然
  Reg      = E[‖z_cls‖² + ‖z_view‖²]                 高斯先验正则化
  DetReg   = E[log_det_cls² + log_det_view²]          雅可比行列式正则化

默认权重: λ_cls = 1.0, λ_view = 0.5, λ_reg = 0.01, λ_det = 0.01
```

**损失函数直觉**：
- NLL 最小化 → Flow 将正常特征映射到标准高斯附近 → 正常样本获得低 NLL
- Reg 正则化防止 z 飘散到无穷远 → 保持数值稳定
- **DetReg 正则化防止"体积膨胀"作弊**：惩罚过大的雅可比行列式，避免模型通过扩大变换体积来降低 loss

> **为什么需要 DetReg？** 纯 NLL 训练中，模型可能学会让 log|det J| 变得很大，使 NLL = -(log p(z) + log|det J|) 变为负数。这并没有真正学到正常分布，而是"作弊"。DetReg 通过惩罚 log_det² 来限制这种现象。

### 3.3 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Batch Size | 8 | 每个 batch 的样本数 |
| Learning Rate | 1e-4 | AdamW 优化器 |
| Epochs | 100 | 训练轮数 |
| Scheduler | Cosine | 学习率衰减策略 |
| Grad Clip | 1.0 | 梯度裁剪阈值 |
| AMP | 开启 (GPU) | 混合精度训练加速 ~30% |

### 3.4 训练过程示意

```
Epoch 1:  Flow 初始化 (ActNorm 用首个 batch 的均值/方差)
          模型开始学习正常特征的分布边界
          loss 较高且波动大

Epoch 10~30: Flow 逐渐收敛
             正常特征被稳定映射到高斯分布
             loss 持续下降

Epoch 50~100: 模型充分学习正常分布
              loss 趋于平稳
              异常样本（如果输入）会产生高 NLL
```

## 4. 推理流程

### 4.1 图像级异常分类

```
测试样本 → DINOv2 → ViewPatchEncoder → LayerNorm → Normalizing Flow → z
                                                                          │
                 image_score = ‖z_cls‖ + mean(‖z_view‖)   (z空间L2距离)
                                                                          │
                            Min-Max 归一化到 [0, 1] → anomaly_score
```

**评分逻辑**（z 空间 L2 距离，比 NLL 更鲁棒）：
- 正常样本 → z 接近 N(0, I) 原点 → ‖z‖ 小 → score 低
- 异常样本 → z 偏离原点 → ‖z‖ 大 → score 高

> **为什么不用 NLL 评分？** NLL 依赖 log_det，可能因雅可比行列式波动导致异常样本得分反而低于正常样本。z 空间 L2 距离是更稳定的异常度量。

### 4.2 像素级异常分割

```
patch_map [B, V, N, 256]
    ↓
计算与 CLS 的余弦距离 → patch_score [B, V, N]
    ↓
reshape 为 [B, V, n_h, n_w]
    ↓
bilinear 上采样 → [B, V, 448, 448]
    ↓
归一化到 [0, 255] → uint8 灰度 mask
    ↓
保存为 {0~4}_mask.png
```

### 4.3 零样本泛化（B 榜未见类别）

由于 DINOv2 在大规模无标注数据上预训练，其特征提取能力具有跨类别的泛化性。对于未见类别：

1. **DINOv2 特征**仍然有效（预训练覆盖了广泛的视觉概念）
2. **ViewPatchEncoder** 通过 Transformer 的通用注意力机制适配新类别的空间结构
3. **Normalizing Flow** 在新类别上的分布偏移会反映为更高的 NLL，但仍能区分正常/异常

**局限**：未见类别的正常分布与训练集差异较大时，Flow 的分布建模能力会下降。

## 5. 提交格式

```
submission.zip
├── submission.csv
│   group_folder,anomaly_score
│   3_adapter/S0001,0.0321
│   battery/S0001,0.8765
│
└── predicted_masks/
    └── 3_adapter/
        └── S0001/
            ├── 0_mask.png  (448×448, 单通道灰度)
            ├── 1_mask.png
            ├── 2_mask.png
            ├── 3_mask.png
            └── 4_mask.png
```

## 6. 评估指标

```
S = 100 × (0.3 × S_cls + 0.5 × S_seg + 0.2 × S_zs)

S_cls (30%): 已见类别图像级 I-AUROC + I-AUPR 宏平均
S_seg (50%): 已见类别像素级 P-AUROC + P-AUPR + P-F1max 宏平均
S_zs  (20%): 未见类别综合指标
```

**关键洞察**：分割得分占 50%，说明像素级定位能力是最重要的评分维度。patch 级异常热力图的质量直接决定最终排名。
