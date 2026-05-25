import numpy as np
from scipy.signal import convolve2d


def gaussian_kernel(size, sigma):
    """生成二维高斯卷积核。

    参数:
        size: 奇数，卷积核边长。
        sigma: 高斯标准差。

    返回:
        形状为 (size, size) 的二维数组，归一化后总和为 1。

    抛出:
        ValueError: 当 size 为偶数时。
    """
    if size % 2 == 0:
        raise ValueError(
            f"高斯核尺寸必须为奇数才能居中，当前为 {size}。"
            f"请使用奇数尺寸，如 {size + 1}。"
        )
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g1d = np.exp(-0.5 * (x / sigma) ** 2)
    g1d /= g1d.sum()
    kernel = np.outer(g1d, g1d)
    return kernel


def apply_gaussian_weighting(Ix2, Ixy, Iy2, kernel):
    """用高斯核对结构张量的每个分量做卷积（加权求和）。

    参数:
        Ix2, Ixy, Iy2: 二维数组（梯度乘积图）。
        kernel: 二维高斯核。

    返回:
        dict，键为 'Sxx', 'Sxy', 'Syy'。
    """
    return {
        'Sxx': convolve2d(Ix2, kernel, mode='same', boundary='symm'),
        'Sxy': convolve2d(Ixy, kernel, mode='same', boundary='symm'),
        'Syy': convolve2d(Iy2, kernel, mode='same', boundary='symm'),
    }
