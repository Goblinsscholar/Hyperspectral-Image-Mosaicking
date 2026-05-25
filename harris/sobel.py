import numpy as np
from scipy.signal import convolve2d


def sobel_x(image):
    """计算图像在 X 方向上的 Sobel 梯度 I_x。

    参数:
        image: 二维 numpy 数组（灰度图）。

    返回:
        与输入相同形状的二维数组，表示 X 方向梯度。
    """
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
    return convolve2d(image, kernel, mode='same', boundary='symm')


def sobel_y(image):
    """计算图像在 Y 方向上的 Sobel 梯度 I_y。

    参数:
        image: 二维 numpy 数组（灰度图）。

    返回:
        与输入相同形状的二维数组，表示 Y 方向梯度。
    """
    kernel = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float64)
    return convolve2d(image, kernel, mode='same', boundary='symm')
