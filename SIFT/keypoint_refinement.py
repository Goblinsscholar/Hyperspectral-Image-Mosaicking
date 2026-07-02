"""关键点精确定位与筛选。

对 DoG 尺度空间极值点候选进行三步精炼：
1. 亚像素定位：三阶泰勒展开，计算连续空间极值偏移量
2. 剔除低对比度点：响应值低于阈值则丢弃
3. 剔除边缘响应点：Hessian 矩阵主曲率比过大则丢弃
"""

import numpy as np


def _compute_gradient(dog_pyramid, octave, layer, y, x):
    """用中心差分计算 DoG 函数在 (y,x,layer) 处的一阶偏导。

    返回:
        (Dx, Dy, Ds) 即相对于 x, y, scale 的一阶偏导。
    """
    dog = dog_pyramid[octave]
    Dx = (dog[layer][y, x+1] - dog[layer][y, x-1]) / 2.0
    Dy = (dog[layer][y+1, x] - dog[layer][y-1, x]) / 2.0
    Ds = (dog[layer+1][y, x] - dog[layer-1][y, x]) / 2.0
    return Dx, Dy, Ds


def _compute_hessian_3d(dog_pyramid, octave, layer, y, x):
    """计算 3x3 Hessian 矩阵（空间 x,y + 尺度 s）。

    返回:
        3x3 numpy 数组。
    """
    dog = dog_pyramid[octave]
    val = dog[layer][y, x]

    # 二阶偏导
    Dxx = dog[layer][y, x+1] - 2*val + dog[layer][y, x-1]
    Dyy = dog[layer][y+1, x] - 2*val + dog[layer][y-1, x]
    Dss = dog[layer+1][y, x] - 2*val + dog[layer-1][y, x]

    # 混合偏导
    Dxy = (dog[layer][y+1, x+1] - dog[layer][y+1, x-1]
           - dog[layer][y-1, x+1] + dog[layer][y-1, x-1]) / 4.0
    Dxs = (dog[layer+1][y, x+1] - dog[layer+1][y, x-1]
           - dog[layer-1][y, x+1] + dog[layer-1][y, x-1]) / 4.0
    Dys = (dog[layer+1][y+1, x] - dog[layer+1][y-1, x]
           - dog[layer-1][y+1, x] + dog[layer-1][y-1, x]) / 4.0

    H = np.array([
        [Dxx, Dxy, Dxs],
        [Dxy, Dyy, Dys],
        [Dxs, Dys, Dss],
    ], dtype=np.float64)
    return H


def _compute_hessian_2d(dog_pyramid, octave, layer, y, x):
    """计算 2x2 空间 Hessian 矩阵（仅 x,y，用于边缘响应判断）。

    返回:
        2x2 numpy 数组。
    """
    dog = dog_pyramid[octave][layer]
    val = dog[y, x]

    Dxx = dog[y, x+1] - 2*val + dog[y, x-1]
    Dyy = dog[y+1, x] - 2*val + dog[y-1, x]
    Dxy = (dog[y+1, x+1] - dog[y+1, x-1]
           - dog[y-1, x+1] + dog[y-1, x-1]) / 4.0

    return np.array([
        [Dxx, Dxy],
        [Dxy, Dyy],
    ], dtype=np.float64)


def refine_keypoints(dog_pyramid, keypoints, num_intervals=3,
                     contrast_threshold=0.03, edge_threshold=10.0,
                     max_iter=5):
    """对候选关键点进行精炼。

    对每个候选点：
    1. 迭代执行亚像素定位（最多 max_iter 次），求解偏移量 offset = -H^{-1}·g
    2. 若任一维度偏移量 > 0.5，移至相邻点重新计算
    3. 若偏移量超出图像边界，丢弃
    4. 计算真实极值响应 |D(hat_x)|，低于 contrast_threshold 则丢弃
    5. 计算 2x2 Hessian 主曲率比，超过 edge_threshold 则丢弃

    参数:
        dog_pyramid: DoG 金字塔。
        keypoints: detect_extrema 输出的候选点列表。
        num_intervals: 每 Octave 有效 Interval 数。
        contrast_threshold: 对比度阈值（原论文 0.03）。
        edge_threshold: 边缘响应阈值 r（原论文 10.0）。
        max_iter: 亚像素定位最大迭代次数。

    返回:
        list of dict: 精炼后的关键点，每个 dict 包含:
            - 'octave', 'layer', 'x', 'y': 亚像素精度坐标
            - 's': 尺度值（绝对尺度）
            - 'value': 修正后的 DoG 响应值
    """
    refined = []

    for kp in keypoints:
        oct_idx = kp['octave']
        layer = kp['layer']
        y = float(kp['y'])
        x = float(kp['x'])

        dog_oct = dog_pyramid[oct_idx]
        h_oct, w_oct = dog_oct[0].shape

        converged = False
        offset = None

        for _ in range(max_iter):
            # 检查边界
            yi, xi, li = int(round(y)), int(round(x)), layer
            if (yi < 1 or yi >= h_oct - 1 or
                xi < 1 or xi >= w_oct - 1 or
                li < 1 or li >= len(dog_oct) - 1):
                break

            # 计算一阶梯度和 Hessian
            gx, gy, gs = _compute_gradient(dog_pyramid, oct_idx, li, yi, xi)
            g = np.array([gx, gy, gs], dtype=np.float64)

            H = _compute_hessian_3d(dog_pyramid, oct_idx, li, yi, xi)

            # 检查 Hessian 是否可逆
            try:
                offset = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break

            # 检查偏移量是否过大
            if (abs(offset[0]) < 0.5 and
                abs(offset[1]) < 0.5 and
                abs(offset[2]) < 0.5):
                converged = True
                break

            # 移动至相邻点
            x += offset[0]
            y += offset[1]
            layer += int(round(offset[2]))

            # 限制在有效范围内
            layer = max(1, min(len(dog_oct) - 2, layer))

        if not converged or offset is None:
            continue

        # 计算修正后的精确坐标（亚像素精度）
        x_sub = x + offset[0]
        y_sub = y + offset[1]
        layer_sub = layer + offset[2]

        # 计算修正后的 DoG 响应值
        yi, xi, li = int(round(y)), int(round(x)), layer
        if (yi < 1 or yi >= h_oct - 1 or
            xi < 1 or xi >= w_oct - 1 or
            li < 1 or li >= len(dog_oct) - 1):
            continue

        dog_val = dog_pyramid[oct_idx][li][yi, xi]
        gx, gy, gs = _compute_gradient(dog_pyramid, oct_idx, li, yi, xi)
        g = np.array([gx, gy, gs], dtype=np.float64)
        d_extremum = dog_val + 0.5 * np.dot(g, offset)

        # ---- 步骤 2: 剔除低对比度点 ----
        if abs(d_extremum) < contrast_threshold:
            continue

        # ---- 步骤 3: 剔除边缘响应点 ----
        yi2, xi2 = int(round(y_sub)), int(round(x_sub))
        yi2 = max(1, min(h_oct - 2, yi2))
        xi2 = max(1, min(w_oct - 2, xi2))

        H_2d = _compute_hessian_2d(dog_pyramid, oct_idx, li, yi2, xi2)
        trace_H = H_2d[0, 0] + H_2d[1, 1]
        det_H = H_2d[0, 0] * H_2d[1, 1] - H_2d[0, 1] ** 2

        if det_H <= 0:
            continue

        # (Tr(H)^2) / det(H) < (r+1)^2 / r
        r = edge_threshold
        curvature_ratio = (trace_H ** 2) / det_H
        if curvature_ratio >= ((r + 1) ** 2) / r:
            continue

        # ---- 通过所有筛选 ----
        # 计算绝对尺度：sigma * 2^octave * k^layer
        # 这里 layer 是 DoG 层，对应的 Gaussian 层为 layer（或 layer+1，取决于约定）
        base_sigma = 1.6  # 初始 sigma
        k = 2.0 ** (1.0 / num_intervals)
        # 特征点的尺度：sigma * 2^octave * k^layer
        # 注意：DoG 的层号与 Gaussian 层号对应关系
        scale = base_sigma * (2.0 ** oct_idx) * (k ** layer)

        refined.append({
            'octave': oct_idx,
            'layer': float(layer_sub),
            'x': float(x_sub),
            'y': float(y_sub),
            's': float(scale),
            'value': float(d_extremum),
        })

    return refined
