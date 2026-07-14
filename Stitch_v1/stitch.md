# 图像拼接 — SIFT → USAC → Stitching

从零实现的**图像拼接**算法，复用已有 **SIFT 特征检测**和 **USAC 鲁棒估计**管道，完成从特征匹配到图像拼接的完整流程。

## 算法概述

图像拼接的核心思路是：**选择其中一张图像作为参考坐标系，通过单应性矩阵将其他所有图像都变换到这个坐标系下，然后填充像素。**

### 拼接流程

| 步骤 | 说明 | 对应模块 |
|------|------|---------|
| 1. SIFT 特征检测 | 检测两幅图像中的尺度不变特征点 | `../SIFT/`（复用） |
| 2. 特征匹配 | 暴力最近邻匹配 + Lowe's Ratio Test 筛选 | `../USAC/matching.py`（复用） |
| 3. USAC 鲁棒估计 | PROSAC+SPRT+MAGSAC+LO 估计单应性矩阵 H | `../USAC/usac_core.py`（复用） |
| 4. **画布构建** | 根据 H 计算变换后图像位置，创建最小包围画布 | `stitching_core.py` |
| 5. **反向映射** | 遍历画布像素，用 H⁻¹ 在原图中双线性插值采样 | `stitching_core.py` |
| 6. **图像融合** | 加权融合处理重叠区域，消除拼接接缝 | `stitching_core.py` |

### 1. 画布构建

有了单应性矩阵之后，不能直接把图像 A 变换到图像 B 上，因为**正确的单应性变换会反应真实镜头间的几何关系，图像的位置和大小都会发生改变**（负坐标、超出原图范围、旋转倾斜等）。

具体做法：
1. **保持参考图像 B 的位置不变。**
2. **利用 H 计算图像 A 四个顶点变换后的坐标。**
3. **统计所有顶点的坐标范围：**将图像 B 的四个角点与图像 A 变换后的四个角点统一放到同一坐标系中，统计 $x_{\min}, x_{\max}, y_{\min}, y_{\max}$。
4. **整体平移坐标系：**由于图像数组不能使用负坐标，施加平移矩阵 T：

$$
T = \begin{bmatrix}
1 & 0 & -x_{\min} \\
0 & 1 & -y_{\min} \\
0 & 0 & 1
\end{bmatrix}
$$

5. **构建最终画布：**宽、高分别为 $(x_{\max} - x_{\min} + 1) \times (y_{\max} - y_{\min} + 1)$。

### 2. 反向映射

**前向映射**（遍历原图像素，用 H 计算目标位置）存在严重问题：变换后的坐标几乎不可能落在整数像素位置上，导致空洞或重叠。

因此使用**反向映射（Backward/Inverse Warping）**：

1. 组合变换：$H' = T \cdot H$（单应性变换 + 平移）
2. 对画布中每个整数像素坐标 $(x', y')$，计算其在原图中的浮点坐标：

$$
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
\sim
H'^{-1}
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix}
$$

3. **双线性插值**采样像素值，填入画布。

$$
\begin{aligned}
&I(x,y) \approx (1-dy)[(1-dx)I(x_0,y_0) + dx\cdot I(x_1,y_0)] \\
&\quad + dy[(1-dx)I(x_0,y_1) + dx\cdot I(x_1,y_1)]
\end{aligned}
$$

### 3. 图像融合

两幅图像映射到同一画布后存在重叠区域，需要融合处理：

| 方法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| **直接覆盖** | 重叠区直接采用一幅图像像素 | 实现最简单 | 接缝明显 |
| **加权融合（Feather Blending）** | 根据到边界距离线性加权 | 有效减弱接缝 | 对配准误差敏感 |

本实现默认使用 **Feather Blending**：
- **非重叠区**：直接复制对应图像像素
- **重叠区**：各自计算到边界的距离，归一化后作为权重，加权平均

### 4. 鲁棒回退机制

当 SIFT 匹配点之间不存在有效单应关系时（例如匹配点几何不一致、H 矩阵退化），管道会自动触发三级回退：

| 级别 | 方法 | 触发条件 |
|------|------|---------|
| 1 | **RANSAC 平移估计** | 从匹配点中聚类主流位移，排除野值 |
| 2 | **DLT 直接估计** | 用内点重新拟合单应矩阵 |
| 3 | **像素对齐** | 通过模板匹配找到最大互相关位移 |
| 4 | **单位矩阵** | 以上均失败，两图保持原位 |

回退过程会输出提示信息，最终使用的 H 矩阵可以通过返回值的 `H_used` 字段获取。

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本运行

```bash
python main.py
```

默认使用本地 `data3.png` 和 `data4.png` 作为测试数据。输出结果：
- `pipeline.png` — 拼接算法管道图
- `canvas_visualization.png` — 画布构建过程可视化
- `warping_visualization.png` — 映射过程与重叠区域
- `stitched_result.png` — 四合一拼接结果展示
- `panorama.jpg` — 拼接全景图

### 调参示例

```bash
# 使用直接覆盖融合（非加权）
python main.py --blend-mode direct

# 使用自定义图片
python main.py --image1 ../SIFT/data1.jpg --image2 ../SIFT/data2.jpg

# 更宽松的 Ratio Test
python main.py --ratio-threshold 0.8

# 更大的 USAC 阈值（适合噪声较大的数据）
python main.py --usac-threshold 5.0

# 禁用 USAC 特定组件
python main.py --no-magsac --no-lo

# 指定输出目录
python main.py --output-dir my_results
```

### 数据复用说明

本模块**不修改** `../SIFT/` 和 `../USAC/` 文件夹的任何内容：
- 通过 `sys.path` 导入 SIFT 的尺度空间、特征检测模块
- 通过 `sys.path` 导入 USAC 的 `usac_core` 和 `matching` 模块
- 默认使用本地 `data3.png` + `data4.png` 测试数据

### 所有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--image1` | str | `data3.png` | 图像 A 路径 |
| `--image2` | str | `data4.png` | 图像 B 路径 |
| `--output-dir` | str | `result` | 输出目录 |
| `--blend-mode` | str | `feather` | 融合模式：`feather` 或 `direct` |
| `--ratio-threshold` | float | 0.75 | Ratio Test 阈值 |
| `--usac-threshold` | float | 3.0 | USAC 内点距离阈值(像素) |
| `--usac-max-iter` | int | 2000 | USAC 最大迭代 |
| `--no-prosac` | flag | — | 禁用 PROSAC 渐进采样 |
| `--no-sprt` | flag | — | 禁用 SPRT 快速验证 |
| `--no-magsac` | flag | — | 禁用 MAGSAC 多阈值评分 |
| `--no-lo` | flag | — | 禁用 LO-RANSAC 局部优化 |
| `--no-adaptive` | flag | — | 禁用自适应终止 |

## 项目结构

```
Stitch_v1/
├── main.py                 # 主入口，编排 SIFT→USAC→Stitching 流水线
├── stitching_core.py       # 拼接核心：画布构建 / 反向映射 / 融合 / 鲁棒回退
├── visualize.py            # 可视化：管道图 / 画布 / 映射 / 结果
├── requirements.txt        # 依赖声明 (numpy, scipy, matplotlib)
├── stitch.md               # 本说明文档
├── data3.png               # 测试图像 A
├── data4.png               # 测试图像 B
└── result/                 # 输出图片目录
    ├── pipeline.png                 # 拼接算法管道图
    ├── canvas_visualization.png     # 画布构建过程
    ├── warping_visualization.png    # 映射过程可视化
    ├── stitched_result.png          # 四合一拼接结果
    └── panorama.jpg                 # 全景图
```

## 核心函数说明

| 函数 | 所在模块 | 功能 |
|------|---------|------|
| `compute_canvas_bounds(H, shape1, shape2)` | `stitching_core.py` | 根据 H 计算最小包围画布 |
| `backward_warp(image, H, bounds)` | `stitching_core.py` | 反向映射 + 双线性插值 |
| `translate_image(image, bounds)` | `stitching_core.py` | 参考图像平移复制 |
| `feather_blend(canvas1, mask1, canvas2, mask2)` | `stitching_core.py` | 加权融合 |
| `direct_copy_blend(canvas1, mask1, canvas2, mask2)` | `stitching_core.py` | 直接覆盖融合 |
| `stitch(img1, img2, H, ...)` | `stitching_core.py` | 拼接主函数（含鲁棒回退） |
| `is_homography_degenerate(H)` | `stitching_core.py` | H 退化检测 |
| `estimate_translation_ransac(pts1, pts2, ...)` | `stitching_core.py` | RANSAC 平移估计 |
| `plot_pipeline_diagram(path)` | `visualize.py` | 管道图生成 |
| `plot_canvas_visualization(...)` | `visualize.py` | 画布构建可视化 |
| `plot_warping_result(...)` | `visualize.py` | 映射过程可视化 |
| `plot_stitch_result(...)` | `visualize.py` | 四合一结果展示 |

## 参考

- 博客：[高光谱拼接算法（八）从特征匹配到图像拼接](https://www.cnblogs.com/Goblinscholar/p/21068269)
- Lowe, D.G. "Distinctive Image Features from Scale-Invariant Keypoints." IJCV 2004.
- Raguram, R., et al. "USAC: A Universal Framework for Random Sample Consensus." TPAMI 2013.
