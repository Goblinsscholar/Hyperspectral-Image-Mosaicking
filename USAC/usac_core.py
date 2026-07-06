"""USAC（Universal RANSAC）核心算法实现。

集成五大改进组件：
1. PROSAC 渐进采样 — 按匹配质量排序，逐步扩大采样范围
2. SPRT 序列概率比检验 — 快速淘汰错误模型
3. MAGSAC 多阈值评分 — 边缘化噪声标准差，加权评分
4. LO-RANSAC 局部优化 — 发现更优模型时利用内点重新拟合
5. 自适应终止策略 — 动态更新所需迭代次数

参考：
- Raguram et al. "USAC: A Universal Framework for Random Sample Consensus." TPAMI 2013.
- Chum et al. "Matching with PROSAC – Progressive Sample Consensus." CVPR 2005.
- Chum et al. "Randomized RANSAC with Sequential Probability Ratio Test." ICCV 2005.
- Barath et al. "MAGSAC: Marginalizing Sample Consensus." CVPR 2019.
- Chum et al. "Locally Optimized RANSAC." DAGM 2003.
"""

import numpy as np
import time


# ============================================================
# 基础几何工具
# ============================================================

def normalize_points(points):
    """归一化点坐标，提高 DLT 数值稳定性。"""
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

    参数:
        points1: (N, 2) numpy 数组。
        points2: (N, 2) numpy 数组。

    返回:
        H: 3x3 单应性矩阵（h33=1 归一化）。
    """
    N = points1.shape[0]
    if N < 4:
        raise ValueError(f"至少需要 4 对匹配点，当前为 {N} 对")

    # 归一化
    pts1_norm, T1 = normalize_points(points1)
    pts2_norm, T2 = normalize_points(points2)

    A = []
    for i in range(N):
        x, y = pts1_norm[i]
        xp, yp = pts2_norm[i]
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H_norm = h.reshape(3, 3)

    H = np.linalg.inv(T2) @ H_norm @ T1
    H = H / H[2, 2]
    return H


def compute_reprojection_error(H, points1, points2):
    """计算单应性矩阵 H 下的重投影误差。

    返回:
        errors: (N,) numpy 数组，每个点的重投影误差。
    """
    N = points1.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pts1_h = np.hstack([points1, ones])

    projected = (H @ pts1_h.T).T
    projected = projected / (projected[:, 2:3] + 1e-10)

    diff = projected[:, :2] - points2
    errors = np.sqrt(np.sum(diff ** 2, axis=1))
    return errors


def _is_degenerate(points, eps=1e-6):
    """检查 4 个点是否退化（共线或近似共线）。"""
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(points[i] - points[j]) < eps:
                return True
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    area = abs(np.cross(v1, v2))
    if area < eps:
        return True
    v3 = points[3] - points[0]
    area2 = abs(np.cross(v1, v3))
    if area2 < eps:
        return True
    return False


# ============================================================
# 组件 1: PROSAC 渐进采样
# ============================================================

class PROSACSampler:
    """PROSAC 渐进采样器。

    匹配点已按质量（距离）从好到差排序。
    初始只从前 k_min 个最佳匹配中采样，逐步扩大范围。
    """

    def __init__(self, n_matches, k_min=10, alpha=0.05):
        """
        参数:
            n_matches: 总匹配数 N。
            k_min: 初始采样范围（最佳匹配数）。
            alpha: 每次增长控制参数（越小增长越慢）。
        """
        self.N = n_matches
        self.k_min = min(k_min, n_matches)
        self.alpha = alpha
        self.k = self.k_min
        self.iteration = 0

    def update(self):
        """每次迭代后更新采样范围。"""
        self.iteration += 1
        # PROSAC 渐进增长公式
        # k 从 k_min 逐渐增长到 N
        # 增长速度随迭代递减
        if self.k < self.N:
            # 使用 PROSAC 论文中的增长函数
            growth = self.alpha * (self.N - self.k_min)
            new_k = self.k_min + int(self.iteration * growth / (self.iteration + 1))
            self.k = min(self.N, max(self.k, new_k))

    def sample(self, rng):
        """从当前最佳 k 个匹配中采样 4 个不重复索引。

        返回:
            tuple of 4 ints: 采样到的匹配索引。
        """
        # 以概率 p 从最佳 k 个中采样
        # 以概率 1-p 从全部 N 个中采样（保证全局探索）
        p = min(0.95, max(0.5, self.k / self.N))

        if rng.random() < p:
            # 从最佳 k 个中采样
            pool_size = min(self.k, self.N)
            if pool_size < 4:
                pool_size = self.N
            indices = rng.choice(pool_size, 4, replace=False)
        else:
            # 从全部匹配中采样
            indices = rng.choice(self.N, 4, replace=False)

        return tuple(indices.tolist())


# ============================================================
# 组件 2: SPRT 序列概率比检验
# ============================================================

class SPRTVerifier:
    """SPRT 快速模型验证器。

    边验证边判断当前模型是否还有继续验证的必要。
    若似然比低于拒绝阈值，立即停止验证。
    """

    def __init__(self, epsilon=0.6, delta=0.1, rejection_ratio=100):
        """
        参数:
            epsilon: 好模型的预期内点率（默认 0.6）。
            delta: 错误模型的预期内点率（默认 0.1）。
            rejection_ratio: 拒绝阈值倒数 Λ < 1/A 时拒绝（默认 100）。
        """
        self.epsilon = epsilon
        self.delta = delta
        self.A = rejection_ratio

        # 预计算对数似然比步长
        self.log_p_in_good = np.log(epsilon / delta)
        self.log_p_out_good = np.log((1 - epsilon) / (1 - delta))

    def verify(self, H, points1, points2, threshold, max_checks=None):
        """使用 SPRT 验证候选模型。

        参数:
            H: 3x3 候选单应性矩阵。
            points1: (N, 2) 图像 1 点坐标。
            points2: (N, 2) 图像 2 点坐标。
            threshold: 内点距离阈值（像素）。
            max_checks: 最大检查点数（默认 None = 全部）。

        返回:
            inlier_mask: (N,) bool 数组。
            inlier_count: int。
            checks_used: int，实际检查的点数。
        """
        N = points1.shape[0]
        if max_checks is None:
            max_checks = N

        inlier_mask = np.zeros(N, dtype=bool)
        inlier_count = 0
        log_likelihood_ratio = 0.0

        # 随机顺序检查
        check_order = np.random.permutation(N)

        for i, idx in enumerate(check_order):
            if i >= max_checks:
                break

            # 计算该点的重投影误差
            x1, y1 = points1[idx]
            x2, y2 = points2[idx]

            # 快速单点重投影
            p1 = np.array([x1, y1, 1.0])
            p2_proj = H @ p1
            p2_proj = p2_proj / (p2_proj[2] + 1e-10)
            error = np.sqrt((p2_proj[0] - x2) ** 2 + (p2_proj[1] - y2) ** 2)

            is_inlier = error < threshold

            if is_inlier:
                inlier_mask[idx] = True
                inlier_count += 1
                log_likelihood_ratio += self.log_p_in_good
            else:
                log_likelihood_ratio += self.log_p_out_good

            # SPRT 拒绝判断
            if log_likelihood_ratio < -np.log(self.A):
                # 模型极可能是坏的，提前终止
                break

        return inlier_mask, inlier_count, i + 1


# ============================================================
# 组件 3: MAGSAC 多阈值评分
# ============================================================

class MAGSACScorer:
    """MAGSAC 边缘化噪声标准差评分器。

    从用户传入的 base_threshold 自动推导多尺度范围。
    阈值取 τ = k × σ（k=2.5），σ 在 [σ_base/2, σ_base, σ_base*1.5, σ_base*2]
    范围内变化，覆盖从严格到宽松的阈值区间。
    """

    def __init__(self, base_threshold=3.0, k=2.5):
        """
        参数:
            base_threshold: 用户设定的参考阈值（像素），对应中间尺度。
            k: 阈值系数 τ = kσ，默认 2.5。
        """
        self.k = k
        # 从用户阈值反推基础 σ
        base_sigma = base_threshold / k
        # 多尺度范围：以 base_sigma 为中心，覆盖 0.5×~2×
        self.sigmas = np.array([
            base_sigma * 0.5,
            base_sigma * 1.0,
            base_sigma * 1.5,
            base_sigma * 2.0,
        ])
        # 对应的实际阈值列表
        self.thresholds = self.k * self.sigmas
        # 均匀权重
        self.weights = np.ones(len(self.sigmas)) / len(self.sigmas)

    def score(self, H, points1, points2):
        """计算模型在 MAGSAC 评分下的得分。

        在各尺度阈值下分别统计内点数，加权求和。
        USAC 的特色：边缘化噪声标准差，不依赖单一硬阈值。

        参数:
            H: 3x3 单应性矩阵。
            points1: (N, 2) numpy 数组。
            points2: (N, 2) numpy 数组。

        返回:
            total_score: float，加权得分。
            inlier_counts: list，各 σ 下的内点数量。
            inlier_masks: list，各 σ 下的内点 mask。
            thresholds: list，各尺度对应的实际阈值。
        """
        errors = compute_reprojection_error(H, points1, points2)
        N = points1.shape[0]

        total_score = 0.0
        inlier_counts = []
        inlier_masks = []

        for threshold, weight in zip(self.thresholds, self.weights):
            mask = errors < threshold
            count = int(np.sum(mask))
            inlier_counts.append(count)
            inlier_masks.append(mask)
            total_score += weight * count

        return total_score, inlier_counts, inlier_masks, self.thresholds.tolist()


# ============================================================
# 组件 4: LO-RANSAC 局部优化
# ============================================================

def local_optimization(points1, points2, H_init, inlier_mask,
                       threshold=3.0, max_iterations=10):
    """LO-RANSAC 局部优化。

    当发现更优模型时，从当前内点集中重新采样拟合，
    并通过迭代改善模型质量。

    参数:
        points1: (N, 2) numpy 数组。
        points2: (N, 2) numpy 数组。
        H_init: 3x3 初始单应性矩阵。
        inlier_mask: (N,) bool 数组，当前内点。
        threshold: 内点阈值（像素）。
        max_iterations: 局部优化迭代次数。

    返回:
        H: 3x3 优化后的单应性矩阵。
        inlier_mask: (N,) bool 数组，优化后的内点。
        inlier_count: int。
    """
    N = points1.shape[0]
    inlier_indices = np.where(inlier_mask)[0]
    n_inliers = len(inlier_indices)

    if n_inliers < 4:
        return H_init, inlier_mask, n_inliers

    best_H = H_init
    best_mask = inlier_mask.copy()
    best_count = n_inliers

    for _ in range(max_iterations):
        # 从当前内点中采样 4 个点重新拟合
        if len(inlier_indices) < 4:
            break

        sample_idx = np.random.choice(inlier_indices, 4, replace=False)
        sample_pts1 = points1[sample_idx]
        sample_pts2 = points2[sample_idx]

        if _is_degenerate(sample_pts1) or _is_degenerate(sample_pts2):
            continue

        try:
            H_new = dlt_homography(sample_pts1, sample_pts2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # 评估新模型
        errors = compute_reprojection_error(H_new, points1, points2)
        new_mask = errors < threshold
        new_count = int(np.sum(new_mask))

        if new_count > best_count:
            best_H = H_new
            best_mask = new_mask
            best_count = new_count
            inlier_indices = np.where(new_mask)[0]

    # 最终：使用所有内点进行最小二乘精炼
    if best_count >= 4:
        try:
            final_H = dlt_homography(points1[best_mask], points2[best_mask])
            # 评估精炼结果
            errors = compute_reprojection_error(final_H, points1, points2)
            final_mask = errors < threshold
            final_count = int(np.sum(final_mask))
            if final_count >= best_count:
                best_H = final_H
                best_mask = final_mask
                best_count = final_count
        except (np.linalg.LinAlgError, ValueError):
            pass

    return best_H, best_mask, best_count


# ============================================================
# 组件 5: 自适应终止策略
# ============================================================

def compute_required_iterations(inlier_ratio, confidence=0.99, sample_size=4):
    """计算所需迭代次数。

    基于当前内点率估算还需要多少次迭代才能以给定
    置信度确保至少采样到一次全部由内点组成的样本。

    参数:
        inlier_ratio: float，当前内点比例。
        confidence: float，置信度（默认 0.99）。
        sample_size: int，最小采样集大小（默认 4）。

    返回:
        int: 所需迭代次数。
    """
    if inlier_ratio <= 0:
        return 1000000
    p_outlier = 1.0 - inlier_ratio
    p_good = (1.0 - p_outlier) ** sample_size
    if p_good <= 1e-15:
        return 1000000
    needed = int(np.log(1 - confidence) / np.log(1 - p_good)) + 1
    return max(needed, 5)


# ============================================================
# USAC 主算法
# ============================================================

def usac(points1, points2, qualities=None,
         threshold=3.0, max_iter=5000, confidence=0.99,
         use_prosac=True, use_sprt=True, use_magsac=True,
         use_lo=True, use_adaptive=True):
    """USAC 通用采样一致性算法。

    集成 PROSAC + SPRT + MAGSAC + LO-RANSAC + 自适应终止。

    参数:
        points1: (N, 2) numpy 数组，图像 1 中的特征点坐标。
        points2: (N, 2) numpy 数组，图像 2 中的对应特征点坐标。
        qualities: (N,) numpy 数组，匹配质量（越小越好），
                   供 PROSAC 使用。若为 None 则使用欧氏距离。
        threshold: 内点判定距离阈值（像素，默认 3.0）。
        max_iter: 最大迭代次数（默认 5000）。
        confidence: 置信度（默认 0.99）。
        use_prosac: 是否使用 PROSAC 采样（默认 True）。
        use_sprt: 是否使用 SPRT 验证（默认 True）。
        use_magsac: 是否使用 MAGSAC 评分（默认 True）。
        use_lo: 是否使用 LO-RANSAC 局部优化（默认 True）。
        use_adaptive: 是否使用自适应终止（默认 True）。

    返回:
        dict: 包含以下键：
            - 'H': 3x3 最优单应性矩阵
            - 'inlier_mask': (N,) bool 数组
            - 'inlier_count': int
            - 'outlier_count': int
            - 'inlier_ratio': float
            - 'iterations_used': int
            - 'errors': (N,) 重投影误差数组
            - 'mean_error': float（内点平均重投影误差）
            - 'method': str，固定为 'USAC'
            - 'stats': dict，统计信息
    """
    N = points1.shape[0]

    if N < 4:
        empty_mask = np.zeros(N, dtype=bool)
        return {
            'H': np.eye(3),
            'inlier_mask': empty_mask,
            'inlier_count': 0,
            'outlier_count': N,
            'inlier_ratio': 0.0,
            'iterations_used': 0,
            'errors': np.full(N, np.inf),
            'mean_error': np.inf,
            'method': 'USAC',
            'time_sec': 0.0,
            'stats': {},
        }

    # ---- 初始化 ----
    rng = np.random.RandomState()

    # PROSAC 采样器
    if use_prosac and qualities is not None:
        # 按质量排序：qualities 升序排列对应的索引
        sorted_indices = np.argsort(qualities)
        pts1_sorted = points1[sorted_indices]
        pts2_sorted = points2[sorted_indices]
        # 记录原始索引到排序索引的映射
        inv_sorted = np.argsort(sorted_indices)
        sampler = PROSACSampler(N)
    else:
        pts1_sorted = points1
        pts2_sorted = points2
        inv_sorted = np.arange(N)
        sampler = None

    # 验证器 & 评分器
    sprt = SPRTVerifier() if use_sprt else None
    magsac = MAGSACScorer(base_threshold=threshold) if use_magsac else None

    # 若 MAGSAC 启用，SPRT 使用最宽松的阈值（避免错误拒绝）
    sprt_threshold = magsac.thresholds[-1] if (use_magsac and magsac is not None) else threshold

    # 最佳模型状态
    best_H = None
    best_inlier_mask = None
    best_inlier_count = 0
    best_score = -1.0

    # 自适应终止
    max_inlier_ratio = 0.0
    iterations = max_iter

    # 统计信息
    stats = {
        'total_iterations': 0,
        'sprt_early_rejections': 0,
        'lo_refinements': 0,
        'degenerate_skipped': 0,
        'models_evaluated': 0,
        'iterations_over_time': [],
        'inlier_ratio_over_time': [],
    }

    start_time = time.time()

    for i in range(iterations):
        stats['total_iterations'] = i + 1

        # ---- 步骤 1: 采样 ----
        if use_prosac and sampler is not None:
            # PROSAC 渐进采样
            idx_tuple = sampler.sample(rng)
            sample_idx = list(idx_tuple)
            sampler.update()
        else:
            # 标准随机采样
            sample_idx = rng.choice(N, 4, replace=False).tolist()

        sample_pts1 = pts1_sorted[sample_idx]
        sample_pts2 = pts2_sorted[sample_idx]

        # 退化检查
        if _is_degenerate(sample_pts1) or _is_degenerate(sample_pts2):
            stats['degenerate_skipped'] += 1
            continue

        # ---- 步骤 2: 拟合模型 ----
        stats['models_evaluated'] += 1
        try:
            H = dlt_homography(sample_pts1, sample_pts2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # ---- 步骤 3: 验证模型 ----
        if use_sprt and sprt is not None:
            # SPRT 快速验证（使用 MAGSAC 最宽松阈值，避免错误拒绝）
            sprt_mask, sprt_count, checks_used = sprt.verify(
                H, pts1_sorted, pts2_sorted, sprt_threshold)

            # 如果 SPRT 快速拒绝了该模型
            if sprt_count < 4:
                stats['sprt_early_rejections'] += 1
                continue

            # 用完整验证补全 SPRT 的结果
            if checks_used < N:
                errors_full = compute_reprojection_error(H, pts1_sorted, pts2_sorted)
                sprt_mask = errors_full < sprt_threshold
                sprt_count = int(np.sum(sprt_mask))
        else:
            # 标准验证
            errors = compute_reprojection_error(H, pts1_sorted, pts2_sorted)
            sprt_mask = errors < threshold
            sprt_count = int(np.sum(sprt_mask))

        # ---- 步骤 4: MAGSAC 评分（多尺度） ----
        if use_magsac and magsac is not None:
            score, _, _, _ = magsac.score(H, pts1_sorted, pts2_sorted)
        else:
            score = sprt_count  # 标准内点计数评分

        # ---- 步骤 5: 更新最优模型 ----
        current_is_better = False

        if use_magsac and magsac is not None:
            # MAGSAC 评分比较
            if score > best_score:
                current_is_better = True
        else:
            # 内点计数比较
            if sprt_count > best_inlier_count:
                current_is_better = True

        if current_is_better:
            best_H = H
            best_inlier_mask = sprt_mask
            best_inlier_count = sprt_count
            best_score = score

            # ---- LO-RANSAC 局部优化 ----
            if use_lo and best_inlier_count >= 4:
                lo_H, lo_mask, lo_count = local_optimization(
                    pts1_sorted, pts2_sorted, best_H, best_inlier_mask,
                    threshold=threshold)

                if lo_count > best_inlier_count:
                    best_H = lo_H
                    best_inlier_mask = lo_mask
                    best_inlier_count = lo_count
                    # 重新计算 MAGSAC 评分（使用多尺度）
                    if use_magsac and magsac is not None:
                        best_score, _, _, _ = magsac.score(best_H, pts1_sorted, pts2_sorted)
                    else:
                        best_score = lo_count
                    stats['lo_refinements'] += 1

            # ---- 自适应终止 ----
            if use_adaptive:
                inlier_ratio = best_inlier_count / N
                if inlier_ratio > max_inlier_ratio:
                    max_inlier_ratio = inlier_ratio
                    needed = compute_required_iterations(
                        inlier_ratio, confidence)
                    iterations = min(max_iter, max(needed, 10))

            # 记录收敛过程
            stats['iterations_over_time'].append(i + 1)
            stats['inlier_ratio_over_time'].append(best_inlier_count / N)

    elapsed = time.time() - start_time

    # ---- 最终处理 ----
    if best_H is None or best_inlier_count < 4:
        return {
            'H': np.eye(3),
            'inlier_mask': np.zeros(N, dtype=bool),
            'inlier_count': 0,
            'outlier_count': N,
            'inlier_ratio': 0.0,
            'iterations_used': stats['total_iterations'],
            'errors': np.full(N, np.inf),
            'mean_error': np.inf,
            'method': 'USAC',
            'time_sec': elapsed,
            'stats': stats,
        }

    # 将排序后的结果映射回原始索引
    inlier_mask_orig = np.zeros(N, dtype=bool)
    # best_inlier_mask 对应排序后的 pts1_sorted
    for sort_idx, is_inlier in enumerate(best_inlier_mask):
        if is_inlier:
            orig_idx = inv_sorted[sort_idx]
            inlier_mask_orig[orig_idx] = True

    # 使用所有内点精炼
    if best_inlier_count >= 4:
        try:
            H_refined = dlt_homography(
                points1[inlier_mask_orig], points2[inlier_mask_orig])
            # 验证精炼结果
            refined_errors = compute_reprojection_error(
                H_refined, points1, points2)
            refined_mask = refined_errors < threshold
            refined_count = int(np.sum(refined_mask))
            if refined_count >= best_inlier_count:
                best_H = H_refined
                best_inlier_mask = refined_mask
                best_inlier_count = refined_count
        except (np.linalg.LinAlgError, ValueError):
            pass

    # 最终重投影误差
    errors = compute_reprojection_error(best_H, points1, points2)
    mean_error = float(np.mean(errors[best_inlier_mask])) if best_inlier_count > 0 else np.inf

    return {
        'H': best_H,
        'inlier_mask': best_inlier_mask,
        'inlier_count': best_inlier_count,
        'outlier_count': N - best_inlier_count,
        'inlier_ratio': best_inlier_count / N if N > 0 else 0.0,
        'iterations_used': stats['total_iterations'],
        'errors': errors,
        'mean_error': mean_error,
        'method': 'USAC',
        'time_sec': elapsed,
        'stats': stats,
    }
