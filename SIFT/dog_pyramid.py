"""DoG (Difference of Gaussian) 金字塔构建。

对高斯金字塔中每个 Octave 内的相邻 Gaussian 层做差，
得到 DoG 金字塔。每个 Octave 有 (s+2) 层 DoG 图像。
"""

import numpy as np


def build_dog_pyramid(gaussian_pyramid):
    """从高斯金字塔构建 DoG 金字塔。

    参数:
        gaussian_pyramid: list of list，
            pyramid[octave][layer] = 二维高斯模糊图像。

    返回:
        list of list: dog_pyramid[octave][layer] = 二维 DoG 图像。
            每个 Octave 包含 (len(gaussian_pyramid[octave]) - 1) 层 DoG。
    """
    dog_pyramid = []
    for oct_idx, octave in enumerate(gaussian_pyramid):
        dog_octave = []
        for layer in range(len(octave) - 1):
            dog = octave[layer + 1] - octave[layer]
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    return dog_pyramid
