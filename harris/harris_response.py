import numpy as np


def compute_harris_response(Sxx, Sxy, Syy, k=0.04):
    """计算每个像素的 Harris 角点响应 R。

        R = det(M) - k * trace(M)^2

    其中
        M = [[Sxx, Sxy],
             [Sxy, Syy]]
        det(M) = Sxx * Syy - Sxy^2
        trace(M) = Sxx + Syy

    参数:
        Sxx, Sxy, Syy: 二维数组（高斯加权后的 M 分量）。
        k: 经验常数，通常取 0.04–0.06。

    返回:
        与输入相同形状的二维数组 R。
    """
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    return det - k * trace ** 2


def detect_corners(R_nms, threshold=0.01):
    """从经过 NMS 的响应图中通过阈值筛选角点坐标。

    注意：NMS 必须在调用此函数前已完成。输入的 R_nms 应已将非极大值像素置零。

    参数:
        R_nms: 二维数组，经过 NMS 后的 Harris 响应值。
        threshold: 相对阈值（相对于 R_nms 最大值）。

    返回:
        (N, 2) 形状的数组，每行为 (y, x) 角点坐标。
    """
    corner_mask = R_nms > threshold * R_nms.max()
    return np.argwhere(corner_mask)
