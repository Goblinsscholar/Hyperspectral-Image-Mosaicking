# Harris 角点检测 

从零实现的 Harris 角点检测算法。包含**灰度图**和**多通道 RGB** 两条独立流水线，每步均有可视化输出。

## 算法概述

Harris 角点检测的核心思想：**角点是图像中梯度在两个方向上变化都大的位置**。

### 步骤

| 步骤 | 说明 | 对应模块 |
|------|------|---------|
| 1. 梯度计算 | 用 Sobel 算子计算 X / Y 方向梯度 Ix, Iy | `sobel.py` |
| 2. 结构张量 | 计算梯度乘积：Ix², Ixy, Iy² | `gradient_products.py` |
| 3. 高斯加权 | 对每个像素邻域加权求和，得 M 矩阵分量 Sxx, Sxy, Syy | `gaussian_weighting.py` |
| 4. 角点响应 | R = det(M) - k · trace(M)² | `harris_response.py` |
| 5. 非极大值抑制 | 只保留局部邻域内的最大响应 | `non_max_suppression.py` |
| 6. 阈值筛选 | R > threshold × R_max 的位置即为角点 | `harris_response.py` |

### 多通道 RGB 扩展

彩色图像的处理方式：分别计算 R、G、B 三个通道的 Sobel 梯度，然后融合为统一的梯度乘积：

```
Ix² = Rx² + Gx² + Bx²
Ixy = Rx·Ry + Gx·Gy + Bx·By
Iy² = Ry² + Gy² + By²
```

后续步骤与灰度图完全一致。

## 参数说明

| 参数 | 作用 | 典型范围 | 教学比喻 |
|------|------|---------|---------|
| `k` | 角点响应灵敏度，越大对角点要求越苛刻 | 0.04–0.06 | "对角线与角点的惩罚权重" |
| `sigma` | 高斯平滑的尺度，越大考虑的邻域越宽 | 0.5–3.0 | "看周围多大的区域" |
| `threshold` | 最低角点强度（相对于最大响应的比例） | 0.001–0.1 | "只保留最强的响应" |
| `min-distance` | NMS 邻域半径，控制角点之间的最小间距 | 1–10 | "角点之间要保持距离" |
| `gaussian-size` | 高斯核边长（必须为奇数） | 3–11 | "加权窗口有多大" |

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本运行

```bash
python main.py
```

输出三张对比图：
- `harris_result.png` — 灰度路径 7 行对比图
- `harris_result_rgb.png` — RGB 路径 8 行对比图
- `harris_comparison.png` — 灰度 vs RGB 总对比图

### 调参示例

```bash
# 降低 k 值 → 检测更多角点
python main.py --k 0.02

# 增加平滑 → 忽略细节纹理
python main.py --sigma 2.0

# 稀疏角点（高阈值 + 大间距）
python main.py --threshold 0.05 --min-distance 5

# 使用自定义图片
python main.py --image my_photo.jpg

# 指定输出目录
python main.py --output-dir results
```

### 参数扫描

对同一个参数取多个值，并列对比检测结果：

```bash
# 扫描 k 值
python main.py --sweep k 0.02 0.04 0.06 0.08

# 扫描 sigma 值
python main.py --sweep sigma 0.5 1.0 2.0 3.0

# 扫描 threshold 值
python main.py --sweep threshold 0.005 0.01 0.05 0.1
```

输出为 `sweep_{参数名}.png`。

### 查看全部参数

```bash
python main.py --help
```

## 项目结构

```
harris/
├── main.py                 # 主入口，编排两条流水线
├── sobel.py                # Sobel 梯度算子
├── gradient_products.py    # 梯度乘积计算
├── gaussian_weighting.py   # 高斯核生成与加权
├── harris_response.py      # Harris 响应与角点检测
├── non_max_suppression.py  # 非极大值抑制
├── visualize.py            # 可视化输出
├── data.jpg                # 输入测试图
├── requirements.txt        # 依赖声明
└── README.md
```
