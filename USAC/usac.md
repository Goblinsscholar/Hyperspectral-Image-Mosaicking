# USAC vs RANSAC 对比演示

从零实现的 **USAC（Universal RANSAC）** 误匹配剔除算法，包含 PROSAC、SPRT、MAGSAC、LO-RANSAC、自适应终止五大组件，并与标准 RANSAC 进行可视化对比。

## 算法概述

USAC 并不是一种全新的算法，而是将 RANSAC 提出四十多年间多种经典改进方法整合到统一的框架中：

| 组件 | 说明 | 对应模块 | 论文 |
|------|------|---------|------|
| **PROSAC** | 渐进采样：按匹配质量排序，优先从高质量匹配中采样 | `usac_core.py` | Chum et al. CVPR 2005 |
| **SPRT** | 序列概率比检验：边验证边判断，快速淘汰错误模型 | `usac_core.py` | Chum & Matas ICCV 2005 |
| **MAGSAC** | 多阈值评分：边缘化噪声标准差，加权求和替代硬阈值 | `usac_core.py` | Barath et al. CVPR 2019 |
| **LO-RANSAC** | 局部优化：发现更优模型时从内点中重新采样拟合 | `usac_core.py` | Chum et al. DAGM 2003 |
| **自适应终止** | 动态更新迭代次数：每次发现更优模型后重新估算所需迭代 | `usac_core.py` | Raguram et al. TPAMI 2013 |

### PROSAC 渐进采样

标准 RANSAC 对所有匹配点一视同仁，而 SIFT 匹配后我们已经拿到了描述子距离信息——**距离越近的匹配点越可能是正确的**。

PROSAC 的核心思路：
1. **排序**：将所有匹配按描述子欧氏距离从小到大排序（质量高 → 质量低）
2. **渐进采样**：初始只从前 $k$ 个最佳匹配中采样；随迭代增加逐步扩大 $k$ 值

这样，前期大部分采样集中在高质量匹配上，快速找到正确的模型。实验表明，相同置信度下 PROSAC 仅需标准 RANSAC **1/2 到 1/10 的迭代次数**。

### SPRT 快速验证

标准 RANSAC 每生成一个候选模型，需要验证全部 $N$ 个匹配点。但很多错误模型在验证初期就会暴露——何必验证全部？

SPRT 把模型验证看作统计假设检验问题：
- $H_g$（好模型）：当前模型是真正的好模型，内点率 $\varepsilon \approx 0.6$
- $H_b$（坏模型）：当前模型是随机错误模型，内点率 $\delta \approx 0.1$

维护对数似然比 $\log\Lambda$，每验证一个点就更新一次。若 $\log\Lambda < -\log A$，立即拒绝该模型，进入下一轮采样。

这样，一个错误模型可能只验证 **20-30 个点**就被丢弃，而不是全部几百个。

### MAGSAC 多阈值评分

标准 RANSAC 使用 0/1 硬阈值评分：误差小于阈值计 1 分，否则 0 分。这忽略了误差大小的差异，且阈值选择对结果影响极大。

MAGSAC 的核心思想是：**既然不知道最优阈值是多少，就不要猜。** 而是在多个可能的噪声标准差 $\sigma$ 下分别评分后加权求和：

$$Q(M) \approx \sum_i P(\sigma_i) \cdot L(M, \sigma_i)$$

#### 多尺度范围推导

本实现的 MAGSAC 从用户传入的参考阈值 `--usac-threshold` 自动推导多尺度范围：

1. **反推基础噪声标准差**：$\sigma_{\text{base}} = \text{threshold} / k$，其中 $k = 2.5$
2. **生成多尺度范围**：$\sigma = [0.5\sigma_{\text{base}}, 1.0\sigma_{\text{base}}, 1.5\sigma_{\text{base}}, 2.0\sigma_{\text{base}}]$
3. **计算对应的阈值**：$\tau_i = k \cdot \sigma_i$

例如 `--usac-threshold 3.0` 时：

| 尺度 | 噪声标准差 $\sigma$ | 阈值 $\tau = 2.5\sigma$ |
|------|-------------------|----------------------|
| 严格 | 0.6 px | 1.5 px |
| **参考（用户设定）** | **1.2 px** | **3.0 px** |
| 宽松 | 1.8 px | 4.5 px |
| 最宽松 | 2.4 px | 6.0 px |

四个尺度均匀加权，最终评分是各尺度下内点数量的加权和。\**SPRT 快速验证使用最宽松阈值（6.0px）**，避免错误拒绝被 MAGSAC 认可的好模型，但**模型选择完全基于 MAGSAC 加权评分**。

### LO-RANSAC 局部优化

标准 RANSAC 每次迭代只使用 4 个点拟合模型，其余内点仅用于评分。但真实场景中往往存在几十上百个正确匹配——只用 4 个点浪费了大量信息。

LO-RANSAC 在发现**比历史最优更好的模型**时，立即从当前内点集中重新采样拟合，迭代改善模型质量，最后用全部内点进行最小二乘精炼。

### 自适应终止策略

每当找到新最佳模型，重新估计内点率 $w$，并动态计算还需要多少次迭代：
$$N = \frac{\ln(1-p)}{\ln(1-w^m)}$$

如果已完成的迭代次数超过理论所需值，直接结束。这保证了 $p=0.99$ 的置信度，同时避免无效迭代。

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本运行（USAC vs RANSAC 对比）

```bash
python main.py
```

输出五张对比图：
- `pipeline.png` — USAC 算法管道图
- `usac_vs_ransac_matches.png` — 三栏对比：USAC 内点（绿）| RANSAC 内点（红）| 差异对比（共有黄 / 独有绿/红）
- `error_comparison.png` — 重投影误差分布直方图（带阈值线 + 均值标注）
- `convergence_comparison.png` — **四合一仪表盘**：执行时间/迭代次数、内点/拒绝次数、收敛曲线、精度/效率
- `inlier_ratio_comparison.png` — 内点比例分组柱状图

### 调参示例

```bash
# 使用更宽松的 Ratio Test（让更多匹配进入几何验证）
python main.py --ratio-threshold 0.8

# USAC 的 --usac-threshold 是参考阈值：
# MAGSAC 会据此自动推导多尺度范围 [τ/2, τ, 1.5τ, 2τ]
# 例如 --usac-threshold 3.0 → 实际使用 [1.5, 3.0, 4.5, 6.0]
python main.py --usac-threshold 3.0

# 使用更大的参考阈值（适合噪声较大的数据）
python main.py --usac-threshold 5.0

# 禁用特定 USAC 组件以观察效果
python main.py --no-prosac      # 禁用 PROSAC，使用标准随机采样
python main.py --no-sprt        # 禁用 SPRT，使用标准验证
python main.py --no-magsac      # 禁用 MAGSAC，回退到内点计数评分
python main.py --no-lo          # 禁用 LO-RANSAC 局部优化
python main.py --no-adaptive    # 禁用自适应终止

# 单独观察 PROSAC + LO 的效果（禁用其他组件）
python main.py --no-sprt --no-magsac --no-adaptive

# 使用自定义图片
python main.py --image1 ../SIFT/data1.jpg --image2 ../SIFT/data3.jpg
```

### 查看全部参数

```bash
python main.py --help
```

## 项目结构

```
USAC/
├── main.py                 # 主入口，编排 SIFT → 匹配 → RANSAC+USAC → 对比
├── usac_core.py            # USAC 核心算法（PROSAC+SPRT+MAGSAC+LO+自适应终止）
├── ransac_baseline.py      # RANSAC 基线包装器（调用 ../RANSAC/ 模块，带计时统计）
├── matching.py             # 特征匹配（导入 ../SIFT/ 模块，按距离排序供 PROSAC）
├── visualize.py            # 对比可视化（四合一仪表盘 / 误差直方图 / 管道图）
├── requirements.txt        # 依赖声明
├── usac.md                 # 本说明文档
└── result/                 # 输出图片目录
    ├── pipeline.png                # USAC 算法管道图
    ├── usac_vs_ransac_matches.png  # 内点分布三栏对比
    ├── error_comparison.png        # 误差分布直方图对比
    ├── convergence_comparison.png  # 性能四合一仪表盘
    └── inlier_ratio_comparison.png # 内点比例分组柱状图
```

## 对比指标说明

运行 `python main.py` 后输出 8 项对比指标：

| 指标 | 说明 |
|------|------|
| 内点数量 | 最终模型的内点匹配对数（越多越好） |
| 内点率 | 内点 / 总匹配对（越高越好） |
| 平均误差 | 内点的平均重投影误差（越低越好） |
| 迭代次数 | 总循环次数（RANSAC 固定，USAC 可能因自适应终止提前） |
| **评估模型数** | 实际拟合+评估的模型数量（越少越高效） |
| **执行时间** | 总耗时（秒），反映计算效率 |
| **拒绝次数** | SPRT 提前拒绝的候选模型数（仅 USAC） |
| **局部优化次数** | LO-RANSAC 触发的优化次数 |

**公平性保证**：RANSAC 和 USAC 使用完全相同的超参数（阈值、最大迭代次数、置信度），且输入同一组匹配数据。

## 典型对比结果

在 data1.jpg + data2.jpg（631 + 282 SIFT 特征点 → 52 对 Ratio Test 匹配）上的运行结果：

| 指标 | RANSAC | USAC | 差异 |
|------|--------|------|------|
| 内点数量 | 6 | **7** | +1 (更多) |
| 内点率 | 11.5% | **13.5%** | +1.9% |
| 平均误差 | 0.000px | 0.478px | USAC 包含更多边界内点 |
| 迭代次数 | 2000 | 2000 | 相同（公平对比） |
| **评估模型数** | 2000 | **1492** | **-25%** |
| **SPRT 拒绝** | 0 | **1478** | 提前淘汰坏模型 |
| 执行时间 | 0.76s | 0.77s | 几乎一致 |

**核心结论**：在完全相同超参数下，USAC 通过 MAGSAC 多尺度评分比 RANSAC 多发现 1 个正确匹配（+17%），同时 SPRT 拒绝 1478 个坏模型使实际评估量减少 25%。执行时间基本一致，说明 USAC 的附加计算开销被 SPRT 的效率提升所抵消。

## 参考

- Raguram, R., et al. "USAC: A Universal Framework for Random Sample Consensus." TPAMI 2013.
- Chum, O., & Matas, J. "Matching with PROSAC – Progressive Sample Consensus." CVPR 2005.
- Chum, O., & Matas, J. "Randomized RANSAC with Sequential Probability Ratio Test." ICCV 2005.
- Barath, D., et al. "MAGSAC: Marginalizing Sample Consensus." CVPR 2019.
- Chum, O., et al. "Locally Optimized RANSAC." DAGM 2003.
- 博客：[高光谱拼接算法（七）USAC](https://www.cnblogs.com/Goblinscholar/p/21065457)
