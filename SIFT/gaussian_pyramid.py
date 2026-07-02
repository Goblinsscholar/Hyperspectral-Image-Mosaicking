"""高斯拉普拉斯金字塔构建。

为 SIFT 构建高斯尺度空间金字塔。
每 Octave 包含 (s+3) 张不同尺度的高斯模糊图像，
其中 s = num_intervals 为有效尺度层数。
"""

import numpy as np
from scipy.signal import convolve2d


def gaussian_kernel(size, sigma):
    """生成二维高斯卷积核。

    参数:
        size: 奇数，卷积核边长。
        sigma: 高斯标准差。

    返回:
        形状为 (size, size) 的二维数组，归一化后总和为 1。
    """
    if size % 2 == 0:
        raise ValueError(
            f"高斯核尺寸必须为奇数才能居中，当前为 {size}。"
        )
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g1d = np.exp(-0.5 * (x / sigma) ** 2)
    g1d /= g1d.sum()
    return np.outer(g1d, g1d)


def _gaussian_blur(image, sigma, kernel_size=None):
    """对图像用指定 sigma 进行高斯模糊。

    参数:
        image: 二维/三维 numpy 数组。
        sigma: 高斯标准差。
        kernel_size: 核尺寸（默认为 2*ceil(2*sigma)+1）。

    返回:
        与输入相同形状的模糊后图像。
    """
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(2 * sigma)) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
    kernel = gaussian_kernel(kernel_size, sigma)
    if image.ndim == 2:
        return convolve2d(image, kernel, mode='same', boundary='symm')
    elif image.ndim == 3:
        blur = np.zeros_like(image)
        for c in range(image.shape[2]):
            blur[:, :, c] = convolve2d(image[:, :, c], kernel,
                                       mode='same', boundary='symm')
        return blur


def build_gaussian_pyramid(image, num_octaves, num_intervals=3, sigma=1.6):
    """构建 SIFT 高斯金字塔。

    每个 Octave 包含 (num_intervals + 3) 层高斯模糊图像。
    第 0 Octave 的第 0 层用 sigma 模糊；后续层按 k=2^(1/s) 递增。
    跨 Octave 时，取上一 Octave 中 sigma=2*sigma 的层进行 2 倍降采样。

    参数:
        image: 输入图像，二维（灰度）或三维（RGB），值域 [0, 1]。
        num_octaves: 金字塔八度数。
        num_intervals: 每 Octave 有效尺度层数 s（原论文 s=3）。
        sigma: 初始尺度（原论文默认 1.6）。

    返回:
        list of list: pyramid[octave][layer] = 二维图像数组。
        list: 每个图像对应的实际 sigma 值列表 pyramid_sigmas[octave][layer]。
    """
    if image.ndim == 3:
        # 转灰度（同 harris 加权公式）
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()

    k = 2.0 ** (1.0 / num_intervals)
    total_layers = num_intervals + 3  # s+3

    pyramid = []
    pyramid_sigmas = []

    # 第 0 Octave：从原始图像开始
    # 假设输入图像已经过相机预模糊（sigma=0.5），
    # 因此需要额外模糊 sigma_eff = sqrt(sigma^2 - 0.5^2) 以达到目标 sigma
    sigma_pre = 0.5  # 假设的相机预模糊
    sigma_eff = np.sqrt(max(sigma ** 2 - sigma_pre ** 2, 0.01))
    octave0 = []
    octave0_sigmas = []
    img = _gaussian_blur(gray, sigma_eff)
    octave0.append(img.astype(np.float64))
    octave0_sigmas.append(sigma)

    for layer in range(1, total_layers):
        # 每层相对前一层增加 k 倍
        # 但因为是用前一层做基准直接模糊，只需要模糊增量的 sqrt
        prev_sigma = sigma * (k ** (layer - 1))
        cur_sigma = sigma * (k ** layer)
        sigma_diff = np.sqrt(max(cur_sigma ** 2 - prev_sigma ** 2, 0.01))
        blurred = _gaussian_blur(octave0[layer - 1], sigma_diff)
        octave0.append(blurred)
        octave0_sigmas.append(cur_sigma)

    pyramid.append(octave0)
    pyramid_sigmas.append(octave0_sigmas)

    # 后续 Octave：降采样
    for oct_idx in range(1, num_octaves):
        # 上一个 Octave 中 sigma = 2*sigma 的层（即第 num_intervals 层）
        src = pyramid[oct_idx - 1][num_intervals]
        # 2 倍降采样
        h, w = src.shape
        h2, w2 = h // 2, w // 2
        downsampled = src[:h2 * 2:2, :w2 * 2:2]

        oct_imgs = []
        oct_sigmas = []

        # 第 0 层：降采样后的图像，其尺度等价于 2*sigma
        oct_imgs.append(downsampled)
        # 对应尺度：上一 Octave 的基线 sigma 翻倍，但这里我们用 sigma * 2
        base_sigma = sigma * (2 ** oct_idx)
        oct_sigmas.append(base_sigma)

        for layer in range(1, total_layers):
            prev_sigma_layer = base_sigma * (k ** (layer - 1))
            cur_sigma_layer = base_sigma * (k ** layer)
            sigma_diff = np.sqrt(max(cur_sigma_layer ** 2 - prev_sigma_layer ** 2, 0.01))
            blurred = _gaussian_blur(oct_imgs[layer - 1], sigma_diff)
            oct_imgs.append(blurred)
            oct_sigmas.append(cur_sigma_layer)

        pyramid.append(oct_imgs)
        pyramid_sigmas.append(oct_sigmas)

    return pyramid, pyramid_sigmas
