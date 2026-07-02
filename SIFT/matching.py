"""特征匹配。

对两组 SIFT 描述子进行：
1. 暴力最近邻匹配（欧氏距离）
2. Lowe's Ratio Test 筛选
"""

import numpy as np


def brute_force_match(descriptors1, descriptors2):
    """暴力最近邻匹配。

    对 desciptors1 中的每个描述子，在 desciptors2 中找到
    距离最近和次近的两个匹配。

    参数:
        descriptors1: (N1, 128) numpy 数组。
        descriptors2: (N2, 128) numpy 数组。

    返回:
        list of dict: 每个 dict 包含:
            - 'idx1': int, 在 desciptors1 中的索引
            - 'idx2': int, 在 desciptors2 中的索引（最近邻）
            - 'distance': float, 最近邻距离
            - 'distance2': float, 次近邻距离
            - 'ratio': float, 最近邻 / 次近邻 比率
    """
    n1 = descriptors1.shape[0]
    matches = []

    for i in range(n1):
        desc1 = descriptors1[i]
        # 计算与所有 desc2 的欧氏距离
        diffs = descriptors2 - desc1
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        # 排序找到最近和次近
        sorted_indices = np.argsort(distances)
        idx2 = sorted_indices[0]
        d1 = distances[idx2]
        d2 = distances[sorted_indices[1]] if len(sorted_indices) > 1 else d1
        ratio = d1 / d2 if d2 > 0 else 1.0

        matches.append({
            'idx1': i,
            'idx2': int(idx2),
            'distance': float(d1),
            'distance2': float(d2),
            'ratio': float(ratio),
        })

    return matches


def ratio_test(matches, threshold=0.7):
    """Lowe's Ratio Test 筛选匹配。

    仅保留最近邻距离 / 次近邻距离 < threshold 的匹配。

    参数:
        matches: brute_force_match 输出的匹配列表。
        threshold: 比率阈值（原论文推荐 0.7-0.8）。

    返回:
        list of dict: 筛选后的匹配子集。
    """
    return [m for m in matches if m['ratio'] < threshold]
