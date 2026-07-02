"""尺度空间极值检测。

在 DoG 金字塔中进行三维 (x, y, σ) 极值检测。
每个候选点需同时大于（或小于）其当前层 8 邻域
以及上下相邻层各 9 个邻域，共 26 个邻居。
"""

import numpy as np


def detect_extrema(dog_pyramid, num_intervals=3, contrast_threshold=0.03):
    """在 DoG 金字塔中检测尺度空间极值点。

    遍历每个 Octave 的中间 s 层 DoG 图像（排除最顶和最底层），
    对每个像素检查其是否为 26 邻域的极值。

    参数:
        dog_pyramid: list of list, dog_pyramid[octave][layer] = 二维 DoG 图像。
        num_intervals: 每 Octave 的有效 Interval 数 s（原论文 s=3）。
        contrast_threshold: 对比度阈值，用于极值检测时的快速预筛。

    返回:
        list of dict: 每个 dict 包含:
            - 'octave': int, 所在 Octave
            - 'layer': int, 所在 DoG 层号（相对于该 Octave）
            - 'x': int, 列坐标（在该 Octave 图像分辨率下）
            - 'y': int, 行坐标（在该 Octave 图像分辨率下）
            - 'value': float, DoG 响应值
    """
    keypoints = []
    pre_filter = 0.5 * contrast_threshold / num_intervals

    for oct_idx, octave in enumerate(dog_pyramid):
        # 有效 DoG 层为中间 s 层（去除最顶层和最底层）
        # octave 共有 s+2 层，有效层为 1..s（0-based index: 1..s）
        # 需要上下各一层做比较
        for layer in range(1, len(octave) - 1):
            current = octave[layer]
            above = octave[layer + 1]
            below = octave[layer - 1]

            h, w = current.shape
            # 边缘像素跳过，因为需要 3x3x3 邻域
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    val = current[y, x]

                    # 快速预筛：绝对值必须大于阈值
                    if abs(val) < pre_filter:
                        continue

                    # 判断是否为极大值（大于所有 26 个邻居）
                    # 当前层 8 邻域
                    if val <= current[y-1, x-1] or val <= current[y-1, x] or val <= current[y-1, x+1] or \
                       val <= current[y, x-1] or val <= current[y, x+1] or \
                       val <= current[y+1, x-1] or val <= current[y+1, x] or val <= current[y+1, x+1]:
                        continue
                    # 上层 9 邻域
                    if val <= above[y-1, x-1] or val <= above[y-1, x] or val <= above[y-1, x+1] or \
                       val <= above[y, x-1] or val <= above[y, x] or val <= above[y, x+1] or \
                       val <= above[y+1, x-1] or val <= above[y+1, x] or val <= above[y+1, x+1]:
                        continue
                    # 下层 9 邻域
                    if val <= below[y-1, x-1] or val <= below[y-1, x] or val <= below[y-1, x+1] or \
                       val <= below[y, x-1] or val <= below[y, x] or val <= below[y, x+1] or \
                       val <= below[y+1, x-1] or val <= below[y+1, x] or val <= below[y+1, x+1]:
                        continue

                    # 通过所有检查，为极大值
                    keypoints.append({
                        'octave': oct_idx,
                        'layer': layer,
                        'x': x,
                        'y': y,
                        'value': float(val),
                    })

    return keypoints
