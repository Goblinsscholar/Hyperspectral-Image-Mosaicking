"""主方向赋值。

为每个精炼后的 SIFT 关键点计算主方向（和辅助方向），
使描述子具有旋转不变性。

流程：
1. 在关键点对应尺度的高斯图像上取邻域
2. 计算梯度幅值和方向
3. 构建 36-bin 加权方向直方图
4. 抛物线插值精确主方向
5. 超过主峰值 80% 的次峰作为辅助方向
"""

import numpy as np


def _compute_gradient(gaussian_img):
    """计算高斯图像的梯度幅值和方向。

    通过中心差分计算 Ix, Iy，然后求幅值和方向。

    参数:
        gaussian_img: 二维高斯图像。

    返回:
        (magnitude, orientation): 均为与输入同形状的二维数组。
            magnitude: 梯度幅值。
            orientation: 梯度方向（弧度，范围 [-π, π]）。
    """
    Ix = np.zeros_like(gaussian_img)
    Iy = np.zeros_like(gaussian_img)

    Ix[:, 1:-1] = (gaussian_img[:, 2:] - gaussian_img[:, :-2]) / 2.0
    Iy[1:-1, :] = (gaussian_img[2:, :] - gaussian_img[:-2, :]) / 2.0

    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    orientation = np.arctan2(Iy, Ix)

    return magnitude, orientation


def _gaussian_weight(shape, sigma):
    """生成二维高斯权重窗口。

    参数:
        shape: (h, w) 窗口尺寸。
        sigma: 高斯标准差。

    返回:
        shape 形状的二维权重数组。
    """
    h, w = shape
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    weights = np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2 * sigma ** 2))
    return weights


def assign_orientation(keypoints, gaussian_pyramid, pyramid_sigmas,
                       num_bins=36, scale_factor=1.5):
    """为关键点分配主方向（及辅助方向）。

    参数:
        keypoints: refine_keypoints 输出的精炼关键点列表。
        gaussian_pyramid: 高斯金字塔。
        pyramid_sigmas: 金字塔每层对应的尺度值。
        num_bins: 方向直方图 bin 数量（默认 36，每 bin 10°）。
        scale_factor: 高斯权重的尺度因子（默认 1.5）。

    返回:
        list of dict: 添加方向后的关键点，每个 dict 包含原始键及：
            - 'orientation': 主方向（弧度）
            - 'magnitude': 关键点的梯度幅值
            - 如果有辅助方向，会拆分为多个关键点条目
    """
    oriented_keypoints = []

    for kp in keypoints:
        oct_idx = kp['octave']
        layer_idx = int(round(kp['layer']))
        y = kp['y']
        x = kp['x']
        scale = kp['s']

        # 获取对应尺度的高斯图像
        if oct_idx >= len(gaussian_pyramid):
            continue
        gauss_oct = gaussian_pyramid[oct_idx]
        # DoG[layer] = Gaussian[layer+1] - Gaussian[layer]
        # 关键点尺度对应 Gaussian[layer]（较低 sigma 的那一层）
        gauss_layer = min(layer_idx, len(gauss_oct) - 1)
        if gauss_layer < 0 or gauss_layer >= len(gauss_oct):
            continue

        gauss_img = gauss_oct[gauss_layer]
        h_img, w_img = gauss_img.shape

        # 邻域半径：转换到当前 Octave 分辨率下
        # 绝对尺度 sigma 对应原图分辨率，需除以 2^octave 得到当前 Octave 像素
        oct_scale = scale / (2.0 ** oct_idx)
        radius = int(round(scale_factor * oct_scale))

        # 当前 Octave 的分辨率，用于坐标转换
        # 将亚像素坐标转为整数像素
        yi, xi = int(round(y)), int(round(x))

        # 取邻域范围
        y_start = max(0, yi - radius)
        y_end = min(h_img, yi + radius + 1)
        x_start = max(0, xi - radius)
        x_end = min(w_img, xi + radius + 1)

        if y_start >= y_end or x_start >= x_end:
            continue

        # 提取邻域
        patch = gauss_img[y_start:y_end, x_start:x_end]
        ph, pw = patch.shape

        if ph < 3 or pw < 3:
            continue

        # 计算梯度
        mag, orient = _compute_gradient(patch)

        # 高斯权重（使用当前 Octave 分辨率下的尺度）
        weight_sigma = scale_factor * oct_scale
        weights = _gaussian_weight((ph, pw), weight_sigma)

        # 构建方向直方图
        hist = np.zeros(num_bins, dtype=np.float64)
        bin_width = 2.0 * np.pi / num_bins

        for py in range(ph):
            for px in range(pw):
                if mag[py, px] == 0:
                    continue
                # 方向映射到 bin
                angle = orient[py, px]
                # 将角度归一化到 [0, 2π)
                if angle < 0:
                    angle += 2.0 * np.pi
                bin_idx = angle / bin_width
                bin_lo = int(np.floor(bin_idx))
                bin_hi = (bin_lo + 1) % num_bins
                frac = bin_idx - bin_lo

                # 线性插值投票
                vote = mag[py, px] * weights[py, px]
                hist[bin_lo] += vote * (1 - frac)
                hist[bin_hi] += vote * frac

        # 直方图平滑（环状）
        smoothed = hist.copy()
        for _ in range(2):  # 多次平滑
            prev = smoothed.copy()
            smoothed[0] = (prev[0] + prev[1] + prev[-1]) / 3.0
            smoothed[-1] = (prev[-1] + prev[-2] + prev[0]) / 3.0
            for i in range(1, num_bins - 1):
                smoothed[i] = (prev[i - 1] + prev[i] + prev[i + 1]) / 3.0

        # 找到主峰
        max_val = smoothed.max()
        if max_val <= 0:
            continue

        # 抛物线插值精确主方向
        peak_threshold = 0.8 * max_val  # 辅助方向阈值

        for i in range(num_bins):
            if smoothed[i] < peak_threshold:
                continue
            # 确保是局部峰值
            prev_i = (i - 1) % num_bins
            next_i = (i + 1) % num_bins
            if smoothed[i] < smoothed[prev_i] or smoothed[i] < smoothed[next_i]:
                continue

            # 抛物线插值
            left = smoothed[prev_i]
            center = smoothed[i]
            right = smoothed[next_i]
            if 2 * center - left - right == 0:
                delta = 0.0
            else:
                delta = 0.5 * (left - right) / (2 * center - left - right)

            angle_precise = (i + delta) * bin_width
            # 归一化到 [0, 2π)
            angle_precise = angle_precise % (2.0 * np.pi)

            # 创建新关键点（含方向）
            new_kp = kp.copy()
            new_kp['orientation'] = angle_precise
            new_kp['magnitude'] = max_val
            oriented_keypoints.append(new_kp)

    return oriented_keypoints
