"""RANSAC 误匹配剔除与单应性矩阵求解核心算法。

功能：
1. DLT（Direct Linear Transform）求解单应性矩阵
2. RANSAC 迭代：随机采样 4 点 → 拟合 H → 内点判定
3. 最小二乘法精炼单应性矩阵
"""

import numpy as np


def normalize_points(points):
    """归一化点坐标，提高 DLT 数值稳定性。

    将点集平移至质心位于原点，并缩放使平均距离为 sqrt(2)。

    参数:
        points: (N, 2) numpy 数组。

    返回:
        normalized: (N, 2) 归一化后坐标。
        T: 3x3 归一化变换矩阵。
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(centered ** 2, axis=1)))
    scale = np.sqrt(2) / (mean_dist + 1e-10)

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    normalized = (T @ homogeneous.T).T[:, :2]
    return normalized, T


def dlt_homography(points1, points2):
    """直接线性变换（DLT）求解单应性矩阵。

    给定至少 4 对匹配点，通过解线性方程组求解 3x3 单应性矩阵 H，
    使得 points2_i ≅ H · points1_i。

    参数:
        points1: (N, 2) numpy 数组，图像 1 中的点坐标。
        points2: (N, 2) numpy 数组，图像 2 中的对应点坐标。

    返回:
        H: 3x3 单应性矩阵（h33=1 归一化）。
    """
    if points1.shape[0] < 4 or points2.shape[0] < 4:
        raise ValueError(f"至少需要 4 对匹配点，当前为 {points1.shape[0]} 对")

    # 归一化提高数值稳定性
    pts1_norm, T1 = normalize_points(points1)
    pts2_norm, T2 = normalize_points(points2)

    N = pts1_norm.shape[0]
    A = []

    for i in range(N):
        x, y = pts1_norm[i]
        xp, yp = pts2_norm[i]

        # 每对点贡献两行
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])

    A = np.array(A, dtype=np.float64)

    # 解 Ah = 0，SVD 取最小奇异值对应的右奇异向量
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]

    H_norm = h.reshape(3, 3)

    # 反归一化：H = T2^{-1} · H_norm · T1
    H = np.linalg.inv(T2) @ H_norm @ T1

    # 归一化 h33 = 1
    H = H / H[2, 2]

    return H


def compute_reprojection_error(H, points1, points2):
    """计算单应性矩阵 H 下的重投影误差。

    将 points1 通过 H 映射到图像 2 坐标系，计算与实际 points2 的欧氏距离。

    参数:
        H: 3x3 单应性矩阵。
        points1: (N, 2) numpy 数组。
        points2: (N, 2) numpy 数组。

    返回:
        errors: (N,) numpy 数组，每个点的重投影误差。
    """
    N = points1.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pts1_h = np.hstack([points1, ones])  # (N, 3)

    # 映射
    projected = (H @ pts1_h.T).T  # (N, 3)
    projected = projected / projected[:, 2:3]  # 齐次归一化

    # 计算欧氏距离
    diff = projected[:, :2] - points2
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    return errors


def ransac(points1, points2, threshold=3.0, max_iter=2000, confidence=0.99):
    """RANSAC 算法求解单应性矩阵并剔除误匹配。

    参数:
        points1: (N, 2) numpy 数组，图像 1 中的特征点坐标。
        points2: (N, 2) numpy 数组，图像 2 中的对应特征点坐标。
        threshold: 内点判定距离阈值（像素，默认 3.0）。
        max_iter: 最大迭代次数（默认 2000）。
        confidence: 置信度（默认 0.99）。

    返回:
        result: dict，包含：
            - 'H': 3x3 最优单应性矩阵。
            - 'inlier_mask': (N,) bool 数组，True 表示内点。
            - 'inlier_count': int，内点数量。
            - 'iterations': int，实际迭代次数。
    """
    N = points1.shape[0]
    if N < 4:
        return {
            'H': np.eye(3),
            'inlier_mask': np.zeros(N, dtype=bool),
            'inlier_count': 0,
            'iterations': 0,
        }

    best_H = None
    best_inlier_mask = None
    best_inlier_count = 0

    # 自适应迭代次数
    iterations = max_iter
    max_inlier_ratio = 0.0

    for i in range(iterations):
        # 1. 随机采样 4 对匹配点
        sample_indices = np.random.choice(N, 4, replace=False)
        sample_pts1 = points1[sample_indices]
        sample_pts2 = points2[sample_indices]

        # 退化检查：检查是否有重复点或共线
        if _is_degenerate(sample_pts1) or _is_degenerate(sample_pts2):
            continue

        try:
            # 2. 拟合单应性矩阵
            H = dlt_homography(sample_pts1, sample_pts2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # 3. 内点判定
        errors = compute_reprojection_error(H, points1, points2)
        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)

        # 更新最优结果
        if inlier_count > best_inlier_count:
            best_H = H
            best_inlier_mask = inlier_mask
            best_inlier_count = inlier_count

            # 自适应更新迭代次数（提前终止）
            inlier_ratio = inlier_count / N
            if inlier_ratio > max_inlier_ratio:
                max_inlier_ratio = inlier_ratio
                # 估算所需迭代次数
                if inlier_ratio > 0:
                    p_outlier = 1.0 - inlier_ratio
                    # 4 个全是内点的概率
                    p_good = (1.0 - p_outlier) ** 4
                    if p_good > 1e-12:
                        needed = int(np.log(1 - confidence) / np.log(1 - p_good)) + 1
                        iterations = min(max_iter, max(needed, 10))

    # 如果没有找到有效模型，返回单位矩阵
    if best_H is None or best_inlier_count < 4:
        return {
            'H': np.eye(3),
            'inlier_mask': np.zeros(N, dtype=bool),
            'inlier_count': 0,
            'iterations': iterations,
        }

    return {
        'H': best_H,
        'inlier_mask': best_inlier_mask,
        'inlier_count': int(best_inlier_count),
        'iterations': iterations,
    }


def _is_degenerate(points, eps=1e-6):
    """检查 4 个点是否退化（共线或近似共线）。

    参数:
        points: (4, 2) numpy 数组。
        eps: 面积阈值。

    返回:
        bool: True 表示退化。
    """
    # 检查是否有重复点
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(points[i] - points[j]) < eps:
                return True

    # 检查面积（通过叉积）
    # 取前三个点构成的三角形面积
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    area = abs(np.cross(v1, v2))
    if area < eps:
        return True

    # 检查第 4 个点是否与前三个共线
    v3 = points[3] - points[0]
    area2 = abs(np.cross(v1, v3))
    if area2 < eps:
        return True

    return False


def refine_homography(points1, points2, inlier_mask, iterations=10):
    """使用所有内点通过最小二乘法迭代优化单应性矩阵。

    参数:
        points1: (N, 2) numpy 数组。
        points2: (N, 2) numpy 数组。
        inlier_mask: (N,) bool 数组，内点标记。
        iterations: 迭代优化次数。

    返回:
        H: 3x3 精炼后的单应性矩阵。
    """
    inlier_pts1 = points1[inlier_mask]
    inlier_pts2 = points2[inlier_mask]

    if inlier_pts1.shape[0] < 4:
        return np.eye(3)

    # 初始估计
    H = dlt_homography(inlier_pts1, inlier_pts2)

    # 迭代重加权最小二乘
    for _ in range(iterations):
        errors = compute_reprojection_error(H, inlier_pts1, inlier_pts2)
        weights = 1.0 / (errors + 1e-6)

        # 加权 DLT
        N = inlier_pts1.shape[0]
        A = []

        for i in range(N):
            w = weights[i]
            x, y = inlier_pts1[i]
            xp, yp = inlier_pts2[i]

            A.append([0, 0, 0, -w * x, -w * y, -w, w * yp * x, w * yp * y, w * yp])
            A.append([w * x, w * y, w, 0, 0, 0, -w * xp * x, -w * xp * y, -w * xp])

        A = np.array(A, dtype=np.float64)
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1, :]
        H_new = h.reshape(3, 3)
        H_new = H_new / H_new[2, 2]
        H = H_new

    return H




