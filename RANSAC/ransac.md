# RANSAC 误匹配剔除

从零实现的 **RANSAC (Random Sample Consensus)** 误匹配剔除算法教学演示。基于 SIFT 特征匹配结果，通过几何约束剔除误匹配，拟合单应性矩阵。

## 算法概述

RANSAC 的核心思想是：**与其试图一次性用所有数据拟合模型（容易被少量错误数据严重带偏），不如反复随机采样最小子集来拟合模型，然后看每个模型能获得多少其他数据的支持，最终选择支持度最高的那个模型。**

### 完整流水线

| 步骤 | 说明 | 对应模块 |
|------|------|---------|
| 1–8. SIFT 特征检测与匹配 | 调用 `../SIFT/` 模块完成特征点提取、描述子构建、暴力匹配和 Ratio Test | `main.py` (调用 SIFT 模块) |
| 9. RANSAC 误匹配剔除 | 随机采样 4 对匹配点 → DLT 求 H → 重投影误差判定内外点 → 迭代取最优 | `ransac_core.py` |
| 10. 单应性矩阵精炼 | 用所有内点进行加权最小二乘优化 | `ransac_core.py` |

### 单应性矩阵

RANSAC 的前置内容是**单应性矩阵（Homography）**，描述同一平面场景在两幅图像之间的投影变换关系：

$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
\propto
H
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

其中 $H$ 为 $3\times3$ 矩阵，自由度为 8（齐次坐标下 $h_{33}=1$）。理论上只需要 4 对不共线的匹配点即可求解。

### DLT（Direct Linear Transform）

将非线性几何关系改写成关于未知参数的线性方程组求解：

$$
\begin{bmatrix}
x & y & 1 & 0 & 0 & 0 & -x'x & -x'y & -x'\\
0 & 0 & 0 & x & y & 1 & -y'x & -y'y & -y'
\end{bmatrix}
\mathbf{h} = \mathbf{0}
$$

每对匹配点贡献 2 个约束，通过 SVD 求解。

### RANSAC 流程

1. **随机采样**：从匹配点中随机选取 4 对（进行退化检查）
2. **拟合模型**：用 DLT 求解单应性矩阵 $H$
3. **内点判定**：计算所有匹配点的重投影误差，$e_i < T$ 的为内点
4. **重复迭代**：重复 1–3 共 $N$ 次，取内点数最多的 $H$
5. **最终优化**：用所有内点最小二乘精炼 $H$

### 迭代次数估算

给定外点比例 $\epsilon$，4 个点全部为内点的概率为 $P_{good} = (1-\epsilon)^4$，需要 $N$ 次迭代以置信度 $p$ 找到全内点采样：

$$
N = \frac{\log(1-p)}{\log\left(1-(1-\epsilon)^4\right)}
$$

| 外点比例 | 单次全内点概率 | N (p=0.99) | N (p=0.999) |
|---------|:-------------:|:----------:|:-----------:|
| 10%     | 65.6%         | 4          | 6           |
| 30%     | 24.0%         | 17         | 25          |
| 50%     | 6.25%         | 72         | 107         |
| 70%     | 0.81%         | 567        | 850         |

## 参数说明

| 参数 | 作用 | 典型范围 | 教学比喻 |
|------|------|---------|---------|
| `ransac-threshold` | 内点距离阈值（像素） | 1.0–5.0 | "多近才算对得上" |
| `ransac-max-iter` | 最大迭代次数 | 500–5000 | "最多抽多少次签" |
| `ransac-confidence` | 置信度 | 0.95–0.999 | "多确定才满意" |
| `ratio-threshold` | Ratio Test 阈值（越小匹配越严格） | 0.6–0.9 | "预处理去粗差" |

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本运行（双图匹配 + RANSAC + 全景拼接）

```bash
python main.py
```

输出七张图：
- `pipeline.png` — RANSAC 误匹配剔除管道图
- `ransac_matches.png` — RANSAC 前后匹配对比（绿=内点，红=外点）
- `inlier_ratio.png` — 内点/外点比例柱状图
- `error_histogram.png` — 重投影误差分布直方图
- `inliers_img1.png` — 图像 1 内点/外点标注
- `inliers_img2.png` — 图像 2 内点/外点标注

### 调参示例

```bash
# 更严格的内点判定
python main.py --ransac-threshold 2.0

# 更多迭代次数 + 更高置信度
python main.py --ransac-max-iter 5000 --ransac-confidence 0.999

# 先用更严格的 Ratio Test 减少误匹配
python main.py --ratio-threshold 0.6 --ransac-threshold 3.0

# 使用自定义图片
python main.py --image1 ../SIFT/data1.jpg --image2 ../SIFT/data3.jpg

# 指定输出目录
python main.py --output-dir my_results
```

### 参数扫描

对同一个 RANSAC 参数取多个值，对比内点比例：

```bash
# 扫描阈值
python main.py --sweep threshold 1.0 2.0 3.0 5.0

# 扫描迭代次数
python main.py --sweep max-iter 100 500 1000 2000

# 扫描置信度
python main.py --sweep confidence 0.90 0.95 0.99 0.999
```

输出为 `sweep_{参数名}.png`。

### 查看全部参数

```bash
python main.py --help
```

## 项目结构

```
RANSAC/
├── main.py                 # 主入口，编排 SIFT→RANSAC 流水线
├── ransac_core.py          # RANSAC 核心算法（DLT/采样/精炼）
├── visualize.py            # RANSAC 可视化（匹配对比/误差分布/管道图）
├── requirements.txt        # 依赖声明
└── ransac.md               # 本说明文档
```

## 依赖其他文件夹

本模块**不复制** SIFT 代码，而是通过 `sys.path` 直接调用 `../SIFT/` 中的模块：
- `gaussian_pyramid.py` — 高斯金字塔
- `dog_pyramid.py` — DoG 金字塔
- `scale_space_extrema.py` — 尺度空间极值检测
- `keypoint_refinement.py` — 关键点精炼
- `orientation.py` — 主方向赋值
- `descriptor.py` — 描述子构建
- `matching.py` — 暴力匹配 + Ratio Test

同时使用 `../SIFT/` 目录下的 `data1.jpg`, `data2.jpg`, `data3.jpg` 作为测试图像。

## 参考

- Fischler, M.A. and Bolles, R.C. "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." Communications of the ACM, 1981.
- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints." IJCV 2004.
- Hartley, R. and Zisserman, A. "Multiple View Geometry in Computer Vision." Cambridge University Press, 2004.
- 博客：[高光谱拼接算法（六）RANSAC 误匹配剔除](https://www.cnblogs.com/Goblinscholar/p/21113687)
