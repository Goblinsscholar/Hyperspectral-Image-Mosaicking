import numpy as np


def compute_gradient_products(Ix, Iy):
    """计算梯度图的逐元素乘积，得到结构张量分量（高斯加权前）。

    适用于单通道（灰度图）输入。

    返回 I_x^2, I_x * I_y, I_y^2 三个分量。

    参数:
        Ix: 二维数组，X 方向梯度。
        Iy: 二维数组，Y 方向梯度。

    返回:
        dict，键为 'Ix2', 'Ixy', 'Iy2'。
    """
    return {
        'Ix2': Ix ** 2,
        'Ixy': Ix * Iy,
        'Iy2': Iy ** 2,
    }


def compute_multichannel_gradient_products(Rx, Ry, Gx, Gy, Bx, By):
    """计算多通道（RGB）融合的梯度乘积。

    对 RGB 三个通道分别求梯度后，按以下方式融合：
        I_x² = R_x² + G_x² + B_x²
        I_y² = R_y² + G_y² + B_y²
        I_x I_y = R_x R_y + G_x G_y + B_x B_y

    参数:
        Rx, Ry: 二维数组，R 通道的 X / Y 方向梯度。
        Gx, Gy: 二维数组，G 通道的 X / Y 方向梯度。
        Bx, By: 二维数组，B 通道的 X / Y 方向梯度。

    返回:
        dict，键为 'Ix2', 'Ixy', 'Iy2'（融合后的结果）。
    """
    return {
        'Ix2': Rx ** 2 + Gx ** 2 + Bx ** 2,
        'Ixy': Rx * Ry + Gx * Gy + Bx * By,
        'Iy2': Ry ** 2 + Gy ** 2 + By ** 2,
    }
