# SIFT 特征检测与匹配

从零实现的 SIFT（Scale-Invariant Feature Transform）特征检测与匹配算法。包含完整的**特征点检测、方向赋值、描述子构建、特征匹配**流水线，每一步均有可视化输出。

## 算法概述

SIFT 的核心贡献在于同时解决了**尺度不变性**和**特征描述**两个问题，并且对旋转、光照、仿射变化都有很强的鲁棒性。

### 步骤

| 步骤 | 说明 | 对应模块 |
|------|------|---------|
| 1. 高斯金字塔 | 构建多 Octave、多 Interval 的高斯尺度空间 | `gaussian_pyramid.py` |
| 2. DoG 金字塔 | 相邻 Gaussian 层做差，近似尺度归一化 LoG | `dog_pyramid.py` |
| 3. 尺度空间极值检测 | 在 3×3×3 邻域内检测 DoG 极大/极小值 | `scale_space_extrema.py` |
| 4. 关键点精炼 | 亚像素定位 + 低对比度剔除 + 边缘响应剔除 | `keypoint_refinement.py` |
| 5. 主方向赋值 | 梯度方向直方图投票，抛物线插值精确定位 | `orientation.py` |
| 6. 描述子构建 | 旋转归一 + 4×4×8 三线性插值 + L2 归一化 + 截断 | `descriptor.py` |
| 7. 暴力匹配 | 最近邻欧氏距离搜索 | `matching.py` |
| 8. Ratio Test | Lowe's 比率测试筛选误匹配 | `matching.py` |

### 尺度空间

SIFT 的核心创新之一是用**高斯差分（Difference of Gaussian, DoG）** 近似**尺度归一化高斯拉普拉斯（LoG）**：

$$
D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma) \approx (k-1)\sigma^2\nabla^2L
$$

通过构建高斯金字塔（多 Octave × 多 Interval）和 DoG 金字塔，在三维空间 (x, y, σ) 中搜索极值点，从而获得具有尺度不变性的特征点。

### 关键点精炼

经过粗检测的候选点需要三步精炼：

1. **亚像素定位**：用三阶泰勒展开拟合连续空间极值位置

$$
\hat{\mathbf x} = -\left(\frac{\partial^2 D}{\partial \mathbf x^2}\right)^{-1} \frac{\partial D}{\partial \mathbf x}
$$

2. **低对比度剔除**： $|D(\hat{\mathbf x})| < T_c$ （默认 0.03）则丢弃
3. **边缘响应剔除**：Hessian 矩阵主曲率比 $\frac{\text{Tr}(H)^2}{\det(H)} < \frac{(r+1)^2}{r}$（默认 r=10）

### 描述子

为每个关键点构建 128 维特征向量：

1. 旋转至主方向以获得旋转不变性
2. 4×4 子区域 × 8 方向 bin 进行三线性插值投票
3. L2 归一化 → 截断 0.2 → 再次 L2 归一化

## 参数说明

| 参数 | 作用 | 典型范围 | 教学比喻 |
|------|------|---------|---------|
| `sigma` | 初始尺度，越大检测的特征越粗糙 | 1.0–2.5 | "观察的精细程度" |
| `num-intervals` | 每 Octave 有效 Interval 数 s | 3 | "每层放大倍数的细分" |
| `contrast-threshold` | 最低特征对比度 | 0.01–0.05 | "只保留足够醒目的特征" |
| `edge-threshold` | 边缘响应剔除阈值 r | 5–20 | "排除细长边缘上的点" |
| `ratio-threshold` | Ratio Test 阈值，越小匹配越严格 | 0.6–0.9 | "最近邻必须显著优于次近邻" |

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本运行（双图匹配）

```bash
python main.py
```

输出六张图：
- `pipeline.png` — SIFT 算法管道图
- `gaussian_pyramid.png` — 高斯金字塔可视化
- `dog_pyramid.png` — DoG 金字塔可视化
- `sift_keypoints_img1.png` — 图像 1 关键点检测结果
- `sift_keypoints_img2.png` — 图像 2 关键点检测结果
- `sift_matches.png` — 匹配连线图（Ratio Test 筛选后）

### 调参示例

```bash
# 更严格的匹配
python main.py --ratio-threshold 0.6 --contrast-threshold 0.04

# 检测更多特征点
python main.py --contrast-threshold 0.02 --edge-threshold 15

# 使用更大的初始尺度（检测更大尺度的特征）
python main.py --sigma 2.0

# 使用自定义图片
python main.py --image1 my_photo1.jpg --image2 my_photo2.jpg

# 指定输出目录
python main.py --output-dir results
```

### 参数扫描

对同一个参数取多个值，并列对比检测结果：

```bash
# 扫描 sigma
python main.py --sweep sigma 1.0 1.6 2.0 2.5

# 扫描 contrast-threshold
python main.py --sweep contrast-threshold 0.01 0.02 0.03 0.04

# 扫描 edge-threshold
python main.py --sweep edge-threshold 5 10 15 20

# 扫描 ratio-threshold
python main.py --sweep ratio-threshold 0.6 0.7 0.8 0.9
```

输出为 `sweep_{参数名}.png`。

### 查看全部参数

```bash
python main.py --help
```

## 项目结构

```
SIFT/
├── main.py                 # 主入口，编排检测→描述→匹配流水线
├── gaussian_pyramid.py     # 高斯金字塔构建
├── dog_pyramid.py          # DoG 金字塔构建
├── scale_space_extrema.py  # 尺度空间极值检测
├── keypoint_refinement.py  # 亚像素定位 + 低对比度/边缘剔除
├── orientation.py          # 主方向赋值
├── descriptor.py           # 128 维描述子构建
├── matching.py             # 暴力匹配 + Ratio Test
├── visualize.py            # 可视化输出
├── data1.jpg               # 输入测试图 1
├── data2.jpg               # 输入测试图 2
├── data3.jpg               # 输入测试图 3
├── pipeline.png            # 算法管道图
├── requirements.txt        # 依赖声明
└── sift.md                 # 本说明文档
```

## 参考

- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints." IJCV 2004.
- 博客：[高光谱拼接算法（三）SIFT 特征点检测](https://www.cnblogs.com/Goblinscholar/p/20978668)
- 博客：[高光谱拼接算法（四）SIFT 特征匹配](https://www.cnblogs.com/Goblinscholar/p/21033977)
