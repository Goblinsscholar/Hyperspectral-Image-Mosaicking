"""Harris 角点检测主流程。

包含两条独立路径：
  A. 灰度图路径（单通道）
  B. 多通道 RGB 路径（每个通道分别求梯度后融合）

路径 A — 灰度图 Harris 数据传播顺序：
  data.jpg
    │  转为灰度
    ▼
  1. Sobel 算子 → I_x, I_y ────────────── sobel.py
    │
    ▼
  2. 逐元素乘积 → I_x², I_xI_y, I_y² ─── gradient_products.py
    │
    ▼
  3. 高斯加权 → S_xx, S_xy, S_yy ─────── gaussian_weighting.py
    │
    ▼
  4. R = det(M) - k·trace(M)² ────────── harris_response.py
    │
    ├── 4b. 非极大值抑制 NMS ──────────── non_max_suppression.py
    │
    ├── 4c. 阈值筛选 → 角点坐标 ───────── harris_response.py
    │
    ▼
  5. 绘制 5 行对比图 ─────────────────── visualize.py

路径 B — 多通道 RGB Harris 数据传播顺序：
  data.jpg
    │  分离 R、G、B 三通道
    ├── R → Sobel → R_x, R_y ──┐
    ├── G → Sobel → G_x, G_y ──┤  sobel.py
    └── B → Sobel → B_x, B_y ──┘
    │
    ▼
  2. 跨通道融合梯度乘积：
       I_x² = R_x²+G_x²+B_x²
       I_xy = R_xR_y+G_xG_y+B_xB_y       gradient_products.py
       I_y² = R_y²+G_y²+B_y²
    │
    ▼
  3. 高斯加权 → S_xx, S_xy, S_yy ─────── gaussian_weighting.py
    │
    ▼
  4. R = det(M) - k·trace(M)² ────────── harris_response.py
    │
    ├── 4b. 非极大值抑制 NMS ──────────── non_max_suppression.py
    │
    ├── 4c. 阈值筛选 → 角点坐标 ───────── harris_response.py
    │
    ▼
  5. 绘制 6 行对比图（含各通道 Sobel）─── visualize.py
    同时绘制灰度 vs RGB 总对比图（3 行 × 2 列）
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from sobel import sobel_x, sobel_y
from gradient_products import compute_gradient_products, compute_multichannel_gradient_products
from gaussian_weighting import gaussian_kernel, apply_gaussian_weighting
from harris_response import compute_harris_response, detect_corners
from non_max_suppression import apply_nms
from visualize import (plot_harris_steps, plot_multichannel_harris_steps,
                       plot_comparison_figure)


def _to_grayscale(image):
    """用亮度加权公式将 RGB 图 (H, W, 3) 转为灰度图。"""
    if image.ndim == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return image


def run_grayscale_pipeline(gray, gaussian_size, gaussian_sigma, k,
                           corner_threshold, corner_min_distance):
    """灰度图 Harris 角点检测管线。

    返回 (Ix, Iy, products, M, R, R_nms, corners)。
    """
    Ix = sobel_x(gray)
    Iy = sobel_y(gray)
    products = compute_gradient_products(Ix, Iy)
    kernel = gaussian_kernel(gaussian_size, gaussian_sigma)
    M = apply_gaussian_weighting(products['Ix2'], products['Ixy'],
                                  products['Iy2'], kernel)
    R = compute_harris_response(M['Sxx'], M['Sxy'], M['Syy'], k=k)
    R_nms = apply_nms(R, min_distance=corner_min_distance)
    corners = detect_corners(R_nms, threshold=corner_threshold)
    return Ix, Iy, products, M, R, R_nms, corners


def run_multichannel_pipeline(rgb, gaussian_size, gaussian_sigma, k,
                              corner_threshold, corner_min_distance):
    """多通道 RGB Harris 角点检测管线。

    每个通道独立求 Sobel 梯度后再融合为统一的梯度乘积。

    返回 (Rx, Ry, Gx, Gy, Bx, By, products, M, R, R_nms, corners)。
    """
    R_ch = rgb[:, :, 0]
    G_ch = rgb[:, :, 1]
    B_ch = rgb[:, :, 2]

    # 各通道独立 Sobel
    Rx, Ry = sobel_x(R_ch), sobel_y(R_ch)
    Gx, Gy = sobel_x(G_ch), sobel_y(G_ch)
    Bx, By = sobel_x(B_ch), sobel_y(B_ch)

    # 融合梯度乘积
    products = compute_multichannel_gradient_products(Rx, Ry, Gx, Gy, Bx, By)

    # 后续步骤与灰度图完全一致
    kernel = gaussian_kernel(gaussian_size, gaussian_sigma)
    M = apply_gaussian_weighting(products['Ix2'], products['Ixy'],
                                  products['Iy2'], kernel)
    R = compute_harris_response(M['Sxx'], M['Sxy'], M['Syy'], k=k)
    R_nms = apply_nms(R, min_distance=corner_min_distance)
    corners = detect_corners(R_nms, threshold=corner_threshold)
    return Rx, Ry, Gx, Gy, Bx, By, products, M, R, R_nms, corners


def _load_image(image_path):
    """加载图像并归一化到 [0, 1] 范围，确保为 RGB 3 通道。"""
    original = plt.imread(image_path)
    if original.max() > 1.0:
        original = original.astype(np.float64) / 255.0

    if original.ndim == 2:
        original = np.stack([original] * 3, axis=-1)
    elif original.shape[2] == 4:
        original = original[:, :, :3]
    return original


def main(image_path, output_dir, gaussian_size, gaussian_sigma, k,
         corner_threshold, corner_min_distance):
    """运行完整的 Harris 检测演示：灰度路径 + RGB 路径 + 对比图。"""
    original = _load_image(image_path)
    gray = _to_grayscale(original)

    # ===== 路径 A：灰度图 Harris =====
    Ix, Iy, products_g, M_g, R_g, R_nms_g, corners_g = run_grayscale_pipeline(
        gray, gaussian_size, gaussian_sigma, k, corner_threshold, corner_min_distance)
    save_gray = os.path.join(output_dir, 'harris_result.png')
    plot_harris_steps(gray, Ix, Iy, products_g, M_g, corners_g,
                      save_gray)
    print(f'[灰度图] 检测到 {len(corners_g)} 个角点，结果已保存至 {save_gray}')

    # ===== 路径 B：多通道 RGB Harris =====
    Rx, Ry, Gx, Gy, Bx, By, products_c, M_c, R_c, R_nms_c, corners_c = \
        run_multichannel_pipeline(
            original, gaussian_size, gaussian_sigma, k,
            corner_threshold, corner_min_distance)
    save_rgb = os.path.join(output_dir, 'harris_result_rgb.png')
    plot_multichannel_harris_steps(original, Rx, Ry, Gx, Gy, Bx, By,
                                   products_c, M_c, corners_c,
                                   save_rgb)
    print(f'[多通道 RGB] 检测到 {len(corners_c)} 个角点，结果已保存至 {save_rgb}')

    # ===== 总对比图：灰度 vs RGB =====
    corners_g_no_nms = np.argwhere(R_g > corner_threshold * R_g.max())
    corners_c_no_nms = np.argwhere(R_c > corner_threshold * R_c.max())
    save_comp = os.path.join(output_dir, 'harris_comparison.png')
    plot_comparison_figure(gray, original,
                           corners_g_no_nms, corners_g,
                           corners_c_no_nms, corners_c,
                           save_comp)
    print(f'[对比图] 灰度 vs RGB 总对比图已保存至 {save_comp}')


def _resolve_param(val, param_name, default, cast=float):
    return cast(val) if param_name == param_name else default


def run_sweep(image_path, output_dir, param_name, param_values,
              gaussian_size, gaussian_sigma, k,
              corner_threshold, corner_min_distance):
    """参数扫描模式：对同一参数取多个值，并列对比灰度 / RGB 检测结果。"""
    original = _load_image(image_path)
    gray = _to_grayscale(original)

    param_labels = {
        'k': 'k',
        'sigma': 'σ',
        'threshold': 'threshold',
        'min-distance': 'min_distance',
        'gaussian-size': 'gaussian_size',
    }
    label = param_labels.get(param_name, param_name)

    n = len(param_values)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    row_titles = ['灰度图', '多通道 RGB']

    for col, val in enumerate(param_values):
        _k = float(val) if param_name == 'k' else k
        _sigma = float(val) if param_name == 'sigma' else gaussian_sigma
        _threshold = float(val) if param_name == 'threshold' else corner_threshold
        _min_dist = int(val) if param_name == 'min-distance' else corner_min_distance
        _g_size = int(val) if param_name == 'gaussian-size' else gaussian_size

        # 灰度图管线
        _, _, _, _, _, _, corners_g = run_grayscale_pipeline(
            gray, _g_size, _sigma, _k, _threshold, _min_dist)
        # RGB 管线
        *_, corners_c = run_multichannel_pipeline(
            original, _g_size, _sigma, _k, _threshold, _min_dist)

        for row, (corners, img_display) in enumerate([
            (corners_g, np.stack([gray] * 3, axis=-1)),
            (corners_c, original),
        ]):
            ax = axes[row, col]
            ax.imshow(img_display)
            if len(corners) > 0:
                ax.scatter(corners[:, 1], corners[:, 0],
                           s=10, c='red', marker='o', edgecolors='white', linewidths=0.5)
            if row == 0:
                ax.set_title(f'{label} = {val}\n({len(corners)} 个)', fontsize=10)
            else:
                ax.set_title(f'({len(corners)} 个)', fontsize=10)
            ax.axis('off')

        # 行标签
        axes[0, 0].set_ylabel(row_titles[0], fontsize=12)
        axes[1, 0].set_ylabel(row_titles[1], fontsize=12)

    save_path = os.path.join(output_dir, f'sweep_{param_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[扫描] 参数 "{param_name}" 扫描结果已保存至 {save_path}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Harris 角点检测教学演示工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--image', default=None,
                        help='输入图片路径（默认使用项目目录下的 data.jpg）')
    parser.add_argument('--output-dir', default='result',
                        help='输出图片保存目录')
    parser.add_argument('--gaussian-size', type=int, default=5,
                        help='高斯核尺寸（须为奇数）')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='高斯标准差')
    parser.add_argument('--k', type=float, default=0.04,
                        help='Harris k 参数（典型范围 0.04–0.06）')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='角点响应阈值（相对 R 最大值的比例）')
    parser.add_argument('--min-distance', type=int, default=1,
                        help='NMS 最小角点间距')
    parser.add_argument('--sweep', nargs='+', default=None,
                        metavar=('PARAM', 'VALUE'),
                        help=(
                            '参数扫描模式：指定参数名和多个取值，'
                            '如 --sweep k 0.02 0.04 0.06 0.08。'
                            '支持的参数：k, sigma, threshold, min-distance, gaussian-size'
                        ))
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    # 默认图片路径：优先使用 --image，否则用脚本所在目录的 data.jpg
    if args.image is None:
        args.image = str(Path(__file__).parent / 'data.jpg')

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep is not None:
        if len(args.sweep) < 2:
            print('错误：--sweep 需要至少指定一个参数名和两个取值')
            sys.exit(1)
        param_name = args.sweep[0]
        param_values = args.sweep[1:]
        run_sweep(
            image_path=args.image,
            output_dir=args.output_dir,
            param_name=param_name,
            param_values=param_values,
            gaussian_size=args.gaussian_size,
            gaussian_sigma=args.sigma,
            k=args.k,
            corner_threshold=args.threshold,
            corner_min_distance=args.min_distance,
        )
    else:
        main(
            image_path=args.image,
            output_dir=args.output_dir,
            gaussian_size=args.gaussian_size,
            gaussian_sigma=args.sigma,
            k=args.k,
            corner_threshold=args.threshold,
            corner_min_distance=args.min_distance,
        )
