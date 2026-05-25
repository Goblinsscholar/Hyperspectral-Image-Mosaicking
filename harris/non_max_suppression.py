import numpy as np
from scipy.ndimage import maximum_filter


def apply_nms(R, min_distance=1):
    """对 Harris 响应图执行非极大值抑制（NMS）。

    仅在每个局部邻域（半径 = min_distance）内保留最大值，
    将非极大值像素置零，使角点响应更加稀疏。

    通过加入微量位置相关的微扰（ ≈ 1e-12 ）打破平坦区域的值相同问题，
    确保同一平坦区域内仅保留一个角点，避免假阳性簇。

    参数:
        R: 二维数组，Harris 响应值。
        min_distance: 非极大值抑制的邻域半径。

    返回:
        与 R 形状相同的二维数组，非极大值位置已置零。
    """
    size = 2 * min_distance + 1
    # 加入微量位置相关微扰打破平坦区域平局
    eps = 1e-12 * np.indices(R.shape).sum(axis=0)
    R_perturbed = R + eps
    local_max = maximum_filter(R_perturbed, size=size) == R_perturbed
    return np.where(local_max, R, 0.0)
