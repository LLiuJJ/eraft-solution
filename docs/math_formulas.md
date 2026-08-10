# DINOv2 + INP-Former 数学公式推导

> 本文档整理模型各模块用到的全部数学基础公式，按数据流顺序组织。

---

## 1. DINOv2 ViT-B/14 特征提取

### 1.1 Patch Embedding

输入图像 $X \in \mathbb{R}^{3 \times 518 \times 518}$，用步长为 14 的卷积切分为非重叠 patch：

$$
\text{patches} = \text{Conv2d}_{k=14, s=14}(X) \in \mathbb{R}^{768 \times 37 \times 37}
$$

每个 patch 覆盖 $14 \times 14$ 像素区域，共 $N = 37 \times 37 = 1369$ 个 patch。展平后：

$$
\mathbf{x}_{\text{patch}} \in \mathbb{R}^{N \times D}, \quad N=1369, \; D=768
$$

### 1.2 位置编码

将可学习的位置编码 $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{N \times D}$ 加到 patch token 上：

$$
\mathbf{H}^{(0)} = \mathbf{x}_{\text{patch}} + \mathbf{E}_{\text{pos}}
$$

### 1.3 多头自注意力（MHSA）

每层 Transformer 包含多头注意力。对于 $h$ 个头（$h=12$），每个头维度 $d_k = D/h = 64$：

**Query / Key / Value 投影**：

$$
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V
$$

**缩放点积注意力**（scale factor $= d_k^{-1/2} = 1/\sqrt{64}$）：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

**多头拼接**：

$$
\text{MHSA}(\mathbf{H}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O
$$

### 1.4 Transformer Block（Pre-Norm 结构）

每层包含 LayerNorm → MHSA → 残差连接 → LayerNorm → MLP → 残差连接：

$$
\mathbf{H}' = \mathbf{H} + \text{MHSA}(\text{LN}(\mathbf{H}))
$$

$$
\mathbf{H}^{(l+1)} = \mathbf{H}' + \text{MLP}(\text{LN}(\mathbf{H}'))
$$

其中 MLP 为两层 GELU 前馈网络：

$$
\text{MLP}(\mathbf{x}) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(\mathbf{x})))
$$

**GELU 激活函数**：

$$
\text{GELU}(x) = x \cdot \Phi(x) = \frac{x}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

### 1.5 LayerNorm

对特征维度 $D$ 做归一化：

$$
\text{LN}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu = \frac{1}{D}\sum_{i=1}^{D} x_i$，$\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2$，$\gamma, \beta$ 为可学习参数。

### 1.6 多尺度特征输出

从 12 层 Transformer 中提取第 8、9、10、11 层的 patch 输出，组成多尺度特征列表：

$$
\mathcal{M} = \left[\mathbf{H}^{(8)}, \mathbf{H}^{(9)}, \mathbf{H}^{(10)}, \mathbf{H}^{(11)}\right], \quad \text{each} \in \mathbb{R}^{N \times D}
$$

---

## 2. ViewPatchEncoder（多视角 Transformer）

### 2.1 多尺度特征拼接

取 DINOv2 倒数第二层和最后一层拼接，融合浅层纹理与深层语义：

$$
\mathbf{F}_{\text{ms}} = [\mathbf{H}^{(9)} \; \| \; \mathbf{H}^{(11)}] \in \mathbb{R}^{V \times N \times 2D}
$$

其中 $\|$ 表示维度拼接，$V=5$ 为视角数。

### 2.2 线性投影 + LayerNorm + GELU

将 $2D=1536$ 维投影到 $d=256$ 维：

$$
\mathbf{X}_0 = \text{GELU}(\text{LN}(\text{Linear}(\mathbf{F}_{\text{ms}}))) \in \mathbb{R}^{V \times N \times d}
$$

### 2.3 位置编码 + 视角嵌入

$$
\mathbf{X}_1 = \mathbf{X}_0 + \mathbf{E}_{\text{pos}} + \mathbf{E}_{\text{view}}
$$

其中：
- $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{N \times d}$：可学习位置编码（编码 patch 在图像中的空间位置）
- $\mathbf{E}_{\text{view}} \in \mathbb{R}^{V \times d}$：可学习视角嵌入（编码每个 patch 属于哪个视角）

### 2.4 序列构建

展平视角和 patch 维度，前置 CLS token：

$$
\mathbf{S} = [\text{CLS} \; \| \; \text{flatten}(\mathbf{X}_1)] \in \mathbb{R}^{(1 + VN) \times d}
$$

序列长度 $L = 1 + 5 \times 1369 = 6846$。

### 2.5 Transformer 编码

经 4 层 Transformer（8 头注意力，pre-norm 结构），输出：

$$
\mathbf{S}' = \text{TransformerEncoder}^{(4)}(\mathbf{S})
$$

### 2.6 输出提取

- **CLS 输出**（全局上下文）：$\mathbf{c} = \mathbf{S}'_0 \in \mathbb{R}^d$
- **视角 token**（每视角汇总）：$\mathbf{v}_j = \frac{1}{N}\sum_{n=1}^{N} \mathbf{S}'_{1+jN+n}$，对 $j=1,\dots,V$
- **Patch map**（patch 级编码）：$\mathbf{P} = \text{GELU}(\text{Linear}_2(\text{Linear}_1(\mathbf{S}'_{1:}))) \in \mathbb{R}^{V \times N \times d}$

### 2.7 LayerNorm 稳定化

Flow 输入前做 LayerNorm 归一化：

$$
\hat{\mathbf{c}} = \text{LN}(\mathbf{c}), \quad \hat{\mathbf{v}}_j = \text{LN}(\mathbf{v}_j)
$$

---

## 3. Normalizing Flow（可逆概率流）

### 3.1 变量代换与似然

Normalizing Flow 通过可逆变换 $f$ 将输入 $\mathbf{x}$ 映射到隐变量 $\mathbf{z}$，利用变量代换公式计算 $\mathbf{x}$ 的似然：

$$
\log p(\mathbf{x}) = \log p(\mathbf{z}) + \log \left|\det\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right|
$$

其中 $\mathbf{z} = f(\mathbf{x})$，$\log p(\mathbf{z})$ 为标准高斯密度：

$$
\log p(\mathbf{z}) = -\frac{1}{2}\|\mathbf{z}\|^2 - \frac{d}{2}\log(2\pi)
$$

### 3.2 Flow 层结构

每层 Flow 由 3 个可逆子模块串联：$\text{ActNorm} \to \text{Invertible1x1Conv} \to \text{AffineCoupling}$，共 8 层。

#### 3.2.1 ActNorm（激活归一化）

逐维度的可逆仿射变换：

$$
y_i = \frac{x_i - \mu_i}{\exp(\log s_i)}
$$

其中 $\mu_i, \log s_i$ 为可学习参数。初始化时用首个 batch 的均值和标准差：

$$
\mu_i = \text{mean}(x_i), \quad \log s_i = \log(\text{std}(x_i))
$$

**雅可比行列式**（对角阵）：

$$
\log\left|\det\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right| = -\sum_{i=1}^{d} \log s_i
$$

数值安全：$\log s_i$ 被 clamp 到 $[-3, 3]$，即 $s_i \in [0.05, 20]$。

#### 3.2.2 Invertible1×1 Conv（可学习旋转混合）

用 $d/2$ 个 2×2 旋转矩阵参数化，每个作用于一对维度 $(x_{2i}, x_{2i+1})$：

$$
\begin{pmatrix} y_{2i} \\ y_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

其中 $\theta_i$ 为可学习旋转角度。

**雅可比行列式**：旋转矩阵的行列式恒为 1：

$$
\det(\mathbf{R}(\theta)) = \cos^2\theta + \sin^2\theta = 1 \implies \log|\det| = 0
$$

> **设计动机**：用旋转矩阵替代一般可逆矩阵，行列式恒为 1，无需计算 `slogdet`，避免 CUDA 上的 `slogdet` hang 和 AMP 下数值不稳定。

#### 3.2.3 AffineCoupling（仿射耦合层）

将输入分为两半 $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$，$\mathbf{x}_1 \in \mathbb{R}^{d/2}$：

$$
[\log \mathbf{s}, \mathbf{t}] = \text{MLP}(\mathbf{x}_1)
$$

$$
\log s = \tanh(\log \mathbf{s}) \times 1.0, \quad \mathbf{t} = \text{clamp}(\mathbf{t}, -3, 3)
$$

$$
\mathbf{y}_2 = \mathbf{x}_2 \odot \exp(\log \mathbf{s}) + \mathbf{t}
$$

$$
\mathbf{y} = [\mathbf{x}_1, \mathbf{y}_2]
$$

**雅可比行列式**（下三角阵）：

$$
\log\left|\det\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right| = \sum_{i=1}^{d/2} \log s_i
$$

MLP 结构：$\text{Linear}(d/2, 256) \to \text{GELU} \to \text{Linear}(256, 256) \to \text{GELU} \to \text{Linear}(256, d)$

> **初始化技巧**：最后一层权重和偏置初始化为 0，使初始变换接近恒等映射（$\log s \approx 0, t \approx 0$），保证训练初期的数值稳定性。

### 3.3 复合 Flow 的雅可比行列式

8 层 Flow 的总雅可比行列式为各层之积：

$$
\log\left|\det\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right| = \sum_{l=1}^{8} \log\left|\det\frac{\partial f_l}{\partial f_{l-1}}\right|
$$

### 3.4 双路 Flow

| Flow | 输入 | 维度 | 作用 |
|:---|:---|:---|:---|
| $f_{\text{cls}}$ | $\hat{\mathbf{c}} \in \mathbb{R}^{B \times d}$ | $d=256$ | 全局异常分布建模 |
| $f_{\text{view}}$ | $\hat{\mathbf{v}} \in \mathbb{R}^{BV \times d}$ | $d=256$ | 视角一致性分布建模 |

---

## 4. 训练损失函数

### 4.1 z 空间距离损失（主项）

训练信号为 z 到原点的 L2 距离，不含 $\log\det$ 项（防止 Flow 通过膨胀雅可比行列式作弊）：

$$
\mathcal{L}_{\text{cls}} = \frac{1}{B}\sum_{b=1}^{B} \frac{1}{2}\|\mathbf{z}_{\text{cls}}^{(b)}\|^2 = \frac{1}{B}\sum_{b=1}^{B} \frac{1}{2}\sum_{i=1}^{d} z_{\text{cls},i}^{(b)\,2}
$$

$$
\mathcal{L}_{\text{view}} = \frac{1}{BV}\sum_{b=1}^{B}\sum_{j=1}^{V} \frac{1}{2}\|\mathbf{z}_{\text{view}}^{(b,j)}\|^2
$$

> **直觉**：正常样本经 Flow 映射后 $\mathbf{z} \approx \mathbf{0}$，$\|\mathbf{z}\|^2$ 小。推理时异常样本偏离原点，$\|\mathbf{z}\|^2$ 大。

### 4.2 方差正则化（反塌缩项）

防止所有 z 塌缩到 0（z collapse），强制每个维度的批内方差接近 1（标准高斯）：

$$
\mathcal{L}_{\text{var}} = \frac{1}{d}\sum_{i=1}^{d}\left(\text{Var}(z_{\text{cls},i}) - 1\right)^2 + \frac{1}{d}\sum_{i=1}^{d}\left(\text{Var}(z_{\text{view},i}) - 1\right)^2
$$

其中 $\text{Var}$ 为 batch 维度的方差：

$$
\text{Var}(z_i) = \frac{1}{B-1}\sum_{b=1}^{B}\left(z_i^{(b)} - \bar{z}_i\right)^2
$$

### 4.3 Margin Hinge 损失（反塌缩项）

强制 batch 内 z 的平均 L2 范数不低于 margin $m=1.0$：

$$
\mathcal{L}_{\text{margin}} = \text{ReLU}(m - \bar{\|\mathbf{z}_{\text{cls}}\|}) + \text{ReLU}(m - \bar{\|\mathbf{z}_{\text{view}}\|})
$$

其中 $\bar{\|\mathbf{z}\|} = \frac{1}{B}\sum_{b=1}^{B}\|\mathbf{z}^{(b)}\|_2$。

> **作用**：当 $\|\mathbf{z}\| < 1.0$ 时给予惩罚，当 $\|\mathbf{z}\| \geq 1.0$ 时惩罚为 0。配合方差正则共同防止 z 塌缩。

### 4.4 总损失

$$
\boxed{
\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{view}} \mathcal{L}_{\text{view}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}} + \lambda_{\text{margin}} \mathcal{L}_{\text{margin}}
}
$$

默认权重：$\lambda_{\text{cls}}=1.0, \; \lambda_{\text{view}}=0.5, \; \lambda_{\text{var}}=1.0, \; \lambda_{\text{margin}}=0.5$

---

## 5. 推理评分公式

### 5.1 图像级异常得分

### 5.1.1 Flow z 空间距离（当前 z 塌缩后已禁用）

$$
S_{\text{flow}} = \|\mathbf{z}_{\text{cls}}\|_2 + \frac{1}{V}\sum_{j=1}^{V}\|\mathbf{z}_{\text{view},j}\|_2
$$

### 5.1.2 k-NN Memory Bank 余弦距离

#### Step 1: L2 归一化

对测试 patch 特征和 Memory Bank 特征分别做 L2 归一化，使余弦相似度等价于内积：

$$
\hat{\mathbf{p}} = \frac{\mathbf{p}}{\|\mathbf{p}\|_2}, \quad \hat{\mathbf{b}}_i = \frac{\mathbf{b}_i}{\|\mathbf{b}_i\|_2}
$$

其中 $\mathbf{p} \in \mathbb{R}^{D}$ 为单个测试 patch 的多尺度拼接特征（$D = 2 \times 768 = 1536$），$\mathbf{b}_i \in \mathcal{B}_c$ 为类别 $c$ 的 Memory Bank 中的第 $i$ 个参考特征。

#### Step 2: 余弦相似度矩阵

将 $VN$ 个测试 patch 与 Memory Bank 中 $M$ 个参考特征计算相似度，得到相似度矩阵：

$$
\mathbf{S} = \hat{\mathbf{P}} \cdot \hat{\mathcal{B}}_c^\top \in \mathbb{R}^{VN \times M}
$$

其中 $\hat{\mathbf{P}} \in \mathbb{R}^{VN \times D}$ 为展平后的归一化测试特征，$\hat{\mathcal{B}}_c \in \mathbb{R}^{M \times D}$ 为已归一化的 Memory Bank。

矩阵元素 $S_{ij}$ 即为第 $i$ 个测试 patch 与第 $j$ 个参考特征的余弦相似度：

$$
S_{ij} = \cos\theta_{ij} = \frac{\mathbf{p}_i \cdot \mathbf{b}_j}{\|\mathbf{p}_i\| \|\mathbf{b}_j\|}
$$

> **归一化后内积等价于余弦相似度**：因为 $\|\hat{\mathbf{p}}\| = \|\hat{\mathbf{b}}\| = 1$，所以 $\hat{\mathbf{p}} \cdot \hat{\mathbf{b}} = \cos\theta$。

#### Step 3: Top-k 最近邻选择

对每个测试 patch，取相似度最高的 $k$ 个参考特征（$k = 3$）：

$$
\text{top}_k(S_{i,:}) = \text{sort}_{\downarrow}(S_{i,:})[:k] \in \mathbb{R}^{k}
$$

#### Step 4: 平均相似度与异常得分

$$
\bar{s}_i = \frac{1}{k} \sum_{j=1}^{k} S_{i,\sigma_j}
$$

其中 $\sigma_j$ 为第 $j$ 大相似度的索引。异常得分为：

$$
\boxed{d_i = 1 - \bar{s}_i = 1 - \frac{1}{k}\sum_{j=1}^{k}\cos\theta_{i,\sigma_j}}
$$

- 正常 patch：与 Memory Bank 中最近邻高度相似 $\bar{s} \approx 1$ → $d \approx 0$
- 异常 patch：最近邻相似度低 $\bar{s} \ll 1$ → $d$ 接近 1

#### Step 5: 图像级 k-NN 得分聚合

对 $V$ 个视角的所有 $N$ 个 patch 得分，取 top-10% 的均值作为图像级得分：

$$
S_{\text{knn}} = \frac{1}{|\mathcal{T}|}\sum_{i \in \mathcal{T}} d_i, \quad |\mathcal{T}| = \max\left(1, \left\lfloor \frac{VN}{10} \right\rfloor\right)
$$

其中 $\mathcal{T}$ 为所有 $VN$ 个 patch 得分中值最大的 top-10% 索引集合。

### 5.1.3 像素级最大得分

$$
S_{\text{pixel}} = \max_{v,n} \left(1 - \text{sim}_k(\mathbf{p}_{v,n})\right)
$$

### 5.1.4 三路融合

归一化后加权融合（当前权重已关闭 Flow）：

$$
S_{\text{image}} = \alpha_{\text{flow}} \cdot \tilde{S}_{\text{flow}} + \alpha_{\text{knn}} \cdot \tilde{S}_{\text{knn}} + \alpha_{\text{pixel}} \cdot \tilde{S}_{\text{pixel}}
$$

当前默认：$\alpha_{\text{flow}}=0, \; \alpha_{\text{knn}}=0.6, \; \alpha_{\text{pixel}}=0.4$

> Min-Max 归一化：$\tilde{S} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$

### 5.2 像素级 Mask 生成

### 5.2.1 Patch 级异常得分

每个 patch $(v, n)$ 的异常得分由 k-NN 余弦距离（主路）和 Flow patch 距离（辅路）加权融合：

$$
\text{ps}_{v,n} = w_{\text{knn}} \cdot \tilde{d}_{\text{knn}}(v,n) + w_{\text{flow}} \cdot \tilde{d}_{\text{flow}}(v,n)
$$

当前默认（Flow 塌缩后）：$w_{\text{knn}} = 1.0, \; w_{\text{flow}} = 0.0$

**k-NN patch 距离**（与 5.1.2 Step 4 一致）：

$$
d_{\text{knn}}(v,n) = 1 - \frac{1}{k}\sum_{j=1}^{k}\cos\theta_{(v,n),\sigma_j}
$$

**Flow patch 距离**（编码特征到全局均值的 L2 距离）：

$$
d_{\text{flow}}(v,n) = \|\mathbf{P}_{v,n} - \bar{\mathbf{P}}\|_2, \quad \bar{\mathbf{P}} = \frac{1}{VN}\sum_{v=1}^{V}\sum_{n=1}^{N}\mathbf{P}_{v,n}
$$

其中 $\mathbf{P}_{v,n} \in \mathbb{R}^{d}$ 为 patch_head 输出的编码特征（$d = 256$），$\tilde{d}$ 表示百分位归一化后的值。

### 5.2.2 双线性上采样

将 $37 \times 37$ patch grid 上采样到 $448 \times 448$：

$$
\mathbf{M} = \text{Bilinear}(\text{reshape}(\text{ps}), \text{size}=448)
$$

### 5.2.3 高斯平滑

二维高斯卷积核（$\sigma=2.0$）：

$$
G(x, y) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
$$

$$
\mathbf{M}_{\text{smooth}} = \mathbf{M} * G
$$

### 5.2.4 百分位截断

低于 $P_{30}$ 百分位的得分置零，减少假阳性：

$$
\tau = \text{Percentile}(\mathbf{M}_{\text{smooth}}, 30)
$$

$$
\mathbf{M}' = \begin{cases} M_{\text{smooth}} - \tau & \text{if } M_{\text{smooth}} > \tau \\ 0 & \text{otherwise} \end{cases}
$$

### 5.2.5 全局归一化

跨所有视角计算全局上界（99.5 百分位），归一化到 $[0, 255]$：

$$
M_{\text{global}} = \text{Percentile}(\mathbf{M}', 99.5)
$$

$$
\mathbf{M}_{\text{final}} = \text{clip}\left(\frac{\mathbf{M}'}{M_{\text{global}}}, 0, 1\right) \times 255
$$

---

## 6. 优化器与学习率调度

### 6.1 AdamW 优化器

参数更新（含解耦权重衰减）：

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
$$

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon} + \lambda_{\text{wd}} \boldsymbol{\theta}_{t-1}\right)
$$

默认：$\eta=1\times10^{-4}, \; \beta_1=0.9, \; \beta_2=0.999, \; \epsilon=10^{-8}, \; \lambda_{\text{wd}}=0.01$

### 6.2 余弦退火学习率

$$
\eta_t = \eta_{\min} + \frac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

其中 $T$ 为总训练步数，$\eta_{\min}=0, \; \eta_{\max}=10^{-4}$。

### 6.3 梯度裁剪

$$
\mathbf{g} \leftarrow \frac{\mathbf{g}}{\max(1, \|\mathbf{g}\| / \tau)} \cdot \tau, \quad \tau = 1.0
$$

### 6.4 AMP 混合精度

前向传播用 FP16，反向传播后 unscale 梯度再裁剪：

$$
\text{scale} = 2^{16}, \quad \mathbf{g}_{\text{fp32}} = \mathbf{g}_{\text{fp16}} / \text{scale}
$$

---

## 7. 评估指标公式

### 7.1 比赛总评分

$$
S = 100 \times \left(0.3 \times S_{\text{cls}} + 0.5 \times S_{\text{seg}} + 0.2 \times S_{\text{zs}}\right)
$$

### 7.2 图像级 AUROC

$$
\text{AUROC} = \int_0^1 \text{TPR}(\theta)\, d(\text{FPR}(\theta))
$$

其中 TPR = TP/(TP+FN)，FPR = FP/(FP+TN)。

### 7.3 图像级 AUPR

$$
\text{AUPR} = \int_0^1 \text{Precision}(\theta)\, d(\text{Recall}(\theta))
$$

$$
\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}
$$

### 7.4 像素级 F1max

遍历所有阈值 $\theta \in [0, 255]$，取 F1 分数最大值：

$$
F1_{\max} = \max_{\theta} \frac{2 \cdot \text{Precision}(\theta) \cdot \text{Recall}(\theta)}{\text{Precision}(\theta) + \text{Recall}(\theta)}
$$

### 7.5 像素级 AP

$$
\text{AP} = \sum_{n=1}^{N} (R_n - R_{n-1}) \cdot P_n
$$

其中 $R_n, P_n$ 为阈值取第 $n$ 大预测值时的召回率和精确率。

---

## 8. Memory Bank 构建

### 8.1 特征采样

对每个类别 $c$ 的训练集，随机采样 patch 特征构建 memory bank：

$$
\mathcal{B}_c = \{[\mathbf{H}^{(9)}_i \| \mathbf{H}^{(11)}_i]\}_{i=1}^{M}, \quad M \leq M_{\max}
$$

其中 $\|$ 为维度拼接，$M_{\max} = 10000$。

### 8.2 L2 归一化

$$
\hat{\mathbf{b}}_i = \frac{\mathbf{b}_i}{\|\mathbf{b}_i\|_2}
$$

归一化后余弦相似度等价于内积：

$$
\cos(\mathbf{p}, \mathbf{b}) = \frac{\mathbf{p} \cdot \mathbf{b}}{\|\mathbf{p}\| \|\mathbf{b}\|} = \hat{\mathbf{p}} \cdot \hat{\mathbf{b}}
$$

---

## 9. 数据预处理

### 9.1 图像归一化

ImageNet 标准化：

$$
x_{\text{norm}} = \frac{x/255 - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$

其中 $\boldsymbol{\mu} = (0.485, 0.456, 0.406)$，$\boldsymbol{\sigma} = (0.229, 0.224, 0.225)$。

### 9.2 Resize

输入图像 resize 到 $518 \times 518$，保证 patch 划分为整数：

$$
N = \left\lfloor \frac{518}{14} \right\rfloor = 37, \quad N_{\text{total}} = 37^2 = 1369
$$
