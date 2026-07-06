"""标准 RANSAC 基线实现（调用 ../RANSAC/ 模块）。

作为对比基线供 USAC 调用，直接复用 ../RANSAC/ransac_core.py 中的
标准 RANSAC 实现（最小集采样 + 固定迭代 + 内点计数评分），
不重写不修改 RANSAC 文件夹中的任何内容。

与 USAC 使用完全相同的输入匹配点，确保对比公平。
补充计时、模型评估数等统计信息用于可视化对比。
"""

import sys
import time
from pathlib import Path
import numpy as np

# ---- 导入 RANSAC 核心（不修改 RANSAC 文件夹） ----
_RANSAC_DIR = str(Path(__file__).resolve().parent.parent / 'RANSAC')
if _RANSAC_DIR not in sys.path:
    sys.path.append(_RANSAC_DIR)

from ransac_core import (
    ransac as _ransac_impl,
    refine_homography as _refine_homography,
    compute_reprojection_error as _compute_error,
    dlt_homography as _dlt_homography,
)


def run_ransac_baseline(points1, points2, threshold=3.0, max_iter=2000,
                        confidence=0.99, refine=True):
    """运行标准 RANSAC 基线，返回与 USAC 格式统一的结果。

    直接委托给 ../RANSAC/ransac_core.ransac() 实现，并补充统计信息。

    返回:
        dict: 包含以下键：
            - 'H', 'inlier_mask', 'inlier_count', 'outlier_count'
            - 'inlier_ratio', 'iterations_used', 'errors', 'mean_error'
            - 'method': 'RANSAC'
            - 'time_sec': float，总执行时间
            - 'stats': dict，详细统计（models_evaluated, degenerate_skipped 等）
    """
    N = points1.shape[0]
    start_time = time.time()

    if N < 4:
        elapsed = time.time() - start_time
        return {
            'H': np.eye(3),
            'inlier_mask': np.zeros(N, dtype=bool),
            'inlier_count': 0,
            'outlier_count': N,
            'inlier_ratio': 0.0,
            'iterations_used': 0,
            'errors': np.full(N, np.inf),
            'mean_error': np.inf,
            'method': 'RANSAC',
            'time_sec': elapsed,
            'stats': {
                'models_evaluated': 0,
                'degenerate_skipped': 0,
                'sprt_early_rejections': 0,
                'lo_refinements': 0,
                'final_inlier_ratio': 0.0,
                'mean_error': np.inf,
            },
        }

    # 调用 RANSAC 核心实现
    result = _ransac_impl(
        points1, points2,
        threshold=threshold,
        max_iter=max_iter,
        confidence=confidence,
    )

    H = result['H']
    inlier_mask = result['inlier_mask']
    inlier_count = result['inlier_count']
    iterations_used = result['iterations']

    # 最小二乘精炼（更新内点掩码）
    if refine and inlier_count >= 4:
        H = _refine_homography(points1, points2, inlier_mask)

    # 重投影误差（使用最新的 H 重新计算）
    errors = _compute_error(H, points1, points2)
    inlier_mask = errors < threshold
    inlier_count = int(np.sum(inlier_mask))
    mean_error = float(np.mean(errors[inlier_mask])) if inlier_count > 0 else np.inf

    elapsed = time.time() - start_time

    return {
        'H': H,
        'inlier_mask': inlier_mask,
        'inlier_count': inlier_count,
        'outlier_count': N - inlier_count,
        'inlier_ratio': inlier_count / N if N > 0 else 0.0,
        'iterations_used': iterations_used,
        'errors': errors,
        'mean_error': mean_error,
        'method': 'RANSAC',
        'time_sec': elapsed,
        'stats': {
            'models_evaluated': iterations_used,
            'degenerate_skipped': max_iter - iterations_used,  # 近似
            'sprt_early_rejections': 0,
            'lo_refinements': 0,
            'final_inlier_ratio': inlier_count / N if N > 0 else 0.0,
            'mean_error': mean_error,
        },
    }


def compute_homography_dlt(points1, points2):
    """使用 DLT 直接求解单应性矩阵（不经过 RANSAC）。

    委托给 ../RANSAC/ransac_core.dlt_homography()。

    参数:
        points1: (N, 2) numpy 数组。
        points2: (N, 2) numpy 数组。

    返回:
        H: 3x3 单应性矩阵。
    """
    return _dlt_homography(points1, points2)


# 方便导入 — 暴露 RANSAC 核心的底层函数
dlt_homography = _dlt_homography
compute_reprojection_error = _compute_error
refine_homography = _refine_homography
