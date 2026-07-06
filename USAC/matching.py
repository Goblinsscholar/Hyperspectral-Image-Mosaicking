"""特征匹配与数据准备。

为 USAC 提供：
1. 使用 SIFT 描述子进行暴力匹配 + Ratio Test
2. 将匹配结果组装为 USAC 可用的格式（带质量排序信息）

数据复用：通过 sys.path 导入 ../SIFT/ 下的模块，不修改 SIFT 文件夹。
"""

import sys
from pathlib import Path

import numpy as np

# ---- 导入 SIFT 模块（不修改 SIFT 文件夹） ----
_SIFT_DIR = str(Path(__file__).resolve().parent.parent / 'SIFT')
if _SIFT_DIR not in sys.path:
    sys.path.append(_SIFT_DIR)

from gaussian_pyramid import build_gaussian_pyramid
from dog_pyramid import build_dog_pyramid
from scale_space_extrema import detect_extrema
from keypoint_refinement import refine_keypoints
from orientation import assign_orientation
from descriptor import build_descriptor


def brute_force_match(descriptors1, descriptors2):
    """暴力最近邻匹配。

    对 desciptors1 中的每个描述子，在 desciptors2 中找到
    距离最近和次近的两个匹配。

    参数:
        descriptors1: (N1, 128) numpy 数组。
        descriptors2: (N2, 128) numpy 数组。

    返回:
        list of dict: 每个 dict 包含:
            - 'idx1': int
            - 'idx2': int
            - 'distance': float, 最近邻距离
            - 'distance2': float, 次近邻距离
            - 'ratio': float, 最近邻 / 次近邻
    """
    n1 = descriptors1.shape[0]
    matches = []

    for i in range(n1):
        desc1 = descriptors1[i]
        diffs = descriptors2 - desc1
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        sorted_indices = np.argsort(distances)
        idx2 = int(sorted_indices[0])
        d1 = distances[idx2]
        d2 = distances[sorted_indices[1]] if len(sorted_indices) > 1 else d1
        ratio = d1 / d2 if d2 > 0 else 1.0

        matches.append({
            'idx1': i,
            'idx2': idx2,
            'distance': float(d1),
            'distance2': float(d2),
            'ratio': float(ratio),
        })

    return matches


def ratio_test(matches, threshold=0.75):
    """Lowe's Ratio Test 筛选匹配。

    仅保留最近邻距离 / 次近邻距离 < threshold 的匹配。
    USAC 中使用相对宽松的阈值，让更多匹配进入后续几何筛选。

    参数:
        matches: brute_force_match 输出的匹配列表。
        threshold: 比率阈值（默认 0.75）。

    返回:
        list of dict: 筛选后的匹配子集。
    """
    return [m for m in matches if m['ratio'] < threshold]


def match_keypoints(keypoints1, keypoints2, ratio_threshold=0.75):
    """对两组 SIFT 关键点进行匹配，返回按匹配质量排序的结果。

    USAC 的 PROSAC 需要按匹配质量（距离）排序，因此
    输出匹配已按 distance 升序排列（质量最高的排前面）。

    参数:
        keypoints1: SIFT 关键点列表（需含 'descriptor'）。
        keypoints2: SIFT 关键点列表（需含 'descriptor'）。
        ratio_threshold: Ratio Test 阈值（默认 0.75，可放宽至 0.8）。

    返回:
        list of dict: 已按 distance 升序排列的匹配结果。
            - 'idx1', 'idx2': 关键点索引
            - 'distance': 最近邻距离（排序依据，越小质量越高）
            - 'ratio': 比率测试值
            - 'kp1', 'kp2': 对应关键点 dict（含 x,y,octave 等）
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return []

    descs1 = np.array([kp['descriptor'] for kp in keypoints1])
    descs2 = np.array([kp['descriptor'] for kp in keypoints2])

    all_matches = brute_force_match(descs1, descs2)
    good_matches = ratio_test(all_matches, threshold=ratio_threshold)

    # 按 distance 升序排列（质量最高的排前面）→ 供 PROSAC 使用
    good_matches.sort(key=lambda m: m['distance'])

    # 补充关键点信息
    result = []
    for m in good_matches:
        result.append({
            'idx1': m['idx1'],
            'idx2': m['idx2'],
            'distance': m['distance'],
            'distance2': m['distance2'],
            'ratio': m['ratio'],
            'kp1': keypoints1[m['idx1']],
            'kp2': keypoints2[m['idx2']],
        })

    return result


def extract_point_pairs(matches):
    """从匹配结果中提取对应点坐标对（原图分辨率）。

    返回:
        pts1: (N, 2) 原图分辨率下图像 1 的点坐标。
        pts2: (N, 2) 原图分辨率下图像 2 的点坐标。
        qualities: (N,) 匹配质量数组（distance 值，越小越好）。
    """
    pts1 = []
    pts2 = []
    qualities = []

    for m in matches:
        kp1 = m['kp1']
        kp2 = m['kp2']
        # 转换坐标到原图分辨率
        x1 = kp1['x'] * (2 ** kp1['octave'])
        y1 = kp1['y'] * (2 ** kp1['octave'])
        x2 = kp2['x'] * (2 ** kp2['octave'])
        y2 = kp2['y'] * (2 ** kp2['octave'])

        pts1.append([x1, y1])
        pts2.append([x2, y2])
        qualities.append(m['distance'])

    return (np.array(pts1, dtype=np.float64),
            np.array(pts2, dtype=np.float64),
            np.array(qualities, dtype=np.float64))
