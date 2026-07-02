"""128 维 SIFT 描述子构建。

对每个已分配方向的关键点，在其对应尺度的高斯图像上：
1. 取 16σ × 16σ 邻域，旋转至主方向
2. 划分 4×4 子区域，每子区域计算 8-bin 梯度直方图
3. 三线性插值投票
4. L2 归一化 → 截断 0.2 → 再次 L2 归一化
"""

import numpy as np


def _sample_gaussian(gauss_img, y, x):
    """双线性插值采样高斯图像在 (y,x) 处的值。"""
    h, w = gauss_img.shape
    yi, xi = int(np.floor(y)), int(np.floor(x))
    if yi < 0 or yi >= h - 1 or xi < 0 or xi >= w - 1:
        return 0.0
    dy, dx = y - yi, x - xi
    return (gauss_img[yi, xi] * (1 - dy) * (1 - dx) +
            gauss_img[yi + 1, xi] * dy * (1 - dx) +
            gauss_img[yi, xi + 1] * (1 - dy) * dx +
            gauss_img[yi + 1, xi + 1] * dy * dx)


def build_descriptor(keypoints, gaussian_pyramid, num_subregions=4,
                     num_ori_bins=8, descriptor_max_val=0.2):
    """为已定向的关键点构建 128 维 SIFT 描述子。

    参数:
        keypoints: assign_orientation 输出的关键点列表，需含 'x','y','s','orientation'。
        gaussian_pyramid: 高斯金字塔。
        num_subregions: 每个方向上的子区域数（默认 4，共 4×4=16 子区域）。
        num_ori_bins: 每子区域方向 bin 数（默认 8，每 bin 45°）。
        descriptor_max_val: 截断阈值（默认 0.2）。

    返回:
        list: 添加 'descriptor' 后的关键点。
            descriptor 为 128 维 numpy 数组。
    """
    described_keypoints = []

    for kp in keypoints:
        oct_idx = kp['octave']
        layer_idx = int(round(kp['layer']))
        y = kp['y']
        x = kp['x']
        scale = kp['s']
        orientation = kp['orientation']

        # 获取高斯图像
        if oct_idx >= len(gaussian_pyramid):
            continue
        gauss_oct = gaussian_pyramid[oct_idx]
        # DoG[layer] = Gaussian[layer+1] - Gaussian[layer]
        # 关键点尺度对应 Gaussian[layer]，而非 layer+1
        gauss_layer = min(layer_idx, len(gauss_oct) - 1)
        if gauss_layer < 0 or gauss_layer >= len(gauss_oct):
            continue
        gauss_img = gauss_oct[gauss_layer]
        h_img, w_img = gauss_img.shape

        # 坐标已在当前 Octave 分辨率下
        y_oct = y
        x_oct = x

        # 将尺度转换到当前 Octave 分辨率下
        oct_scale = scale / (2.0 ** oct_idx)

        # 窗口大小（在当前 Octave 分辨率下）
        # SIFT 使用 4×4 子区域，每子区域 4 像素，总窗口 16×16
        subregion_width = 4  # 每个子区域的像素宽度
        win_width = subregion_width * num_subregions  # 16
        half_win = win_width / 2.0  # 8

        cos_ori = np.cos(orientation)
        sin_ori = np.sin(orientation)

        # 初始化描述子
        descriptor = np.zeros(num_subregions * num_subregions * num_ori_bins,
                              dtype=np.float64)

        # 高斯权重 sigma = 窗口宽度的一半
        gauss_sigma = half_win

        # 遍历窗口内的采样点（步长 1 像素）
        for dy in range(-int(half_win), int(half_win)):
            for dx in range(-int(half_win), int(half_win)):
                # 旋转到主方向
                rot_y = dy * cos_ori - dx * sin_ori
                rot_x = dy * sin_ori + dx * cos_ori

                # 当前采样点在原图中的坐标
                sample_y = y_oct + rot_y
                sample_x = x_oct + rot_x

                # 检查边界
                if (sample_y < 1 or sample_y >= h_img - 1 or
                    sample_x < 1 or sample_x >= w_img - 1):
                    continue

                # 双线性插值采样高斯图像得到像素值
                # 计算梯度（用中心差分）
                val_plus_x = _sample_gaussian(gauss_img, sample_y, sample_x + 1)
                val_minus_x = _sample_gaussian(gauss_img, sample_y, sample_x - 1)
                val_plus_y = _sample_gaussian(gauss_img, sample_y + 1, sample_x)
                val_minus_y = _sample_gaussian(gauss_img, sample_y - 1, sample_x)

                grad_x = (val_plus_x - val_minus_x) / 2.0
                grad_y = (val_plus_y - val_minus_y) / 2.0

                mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                if mag == 0:
                    continue

                ori = np.arctan2(grad_y, grad_x)
                # 相对于主方向
                rel_ori = ori - orientation
                if rel_ori < 0:
                    rel_ori += 2.0 * np.pi
                if rel_ori >= 2.0 * np.pi:
                    rel_ori -= 2.0 * np.pi

                # 子区域索引（连续值），范围 [0, num_subregions)
                # dy, dx 范围 [-half_win, half_win)，映射到 [0, num_subregions)
                rbin = (dy + half_win) / subregion_width
                cbin = (dx + half_win) / subregion_width

                # 方向 bin（连续值），范围 [0, num_ori_bins)
                obin = rel_ori / (2.0 * np.pi / num_ori_bins)

                # 三线性插值权重
                r0 = int(np.floor(rbin))
                c0 = int(np.floor(cbin))
                o0 = int(np.floor(obin))
                dr = rbin - r0
                dc = cbin - c0
                do_bin = obin - o0

                # 高斯权重
                weight = np.exp(-(dy ** 2 + dx ** 2) / (2 * gauss_sigma ** 2))

                # 遍历 2×2×2 邻域进行三线性插值
                for dr_i in range(2):
                    r_idx = r0 + dr_i
                    if r_idx < 0 or r_idx >= num_subregions:
                        continue
                    wr = (1 - dr) if dr_i == 0 else dr

                    for dc_i in range(2):
                        c_idx = c0 + dc_i
                        if c_idx < 0 or c_idx >= num_subregions:
                            continue
                        wc = (1 - dc) if dc_i == 0 else dc

                        for do_i in range(2):
                            o_idx = (o0 + do_i) % num_ori_bins
                            wo = (1 - do_bin) if do_i == 0 else do_bin

                            bin_idx = (((r_idx * num_subregions + c_idx)
                                        * num_ori_bins) + o_idx)
                            descriptor[bin_idx] += mag * weight * wr * wc * wo

        # ---- L2 归一化 ----
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm

        # ---- 截断 ----
        descriptor = np.clip(descriptor, 0, descriptor_max_val)

        # ---- 再次 L2 归一化 ----
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm

        new_kp = kp.copy()
        new_kp['descriptor'] = descriptor
        described_keypoints.append(new_kp)

    return described_keypoints
