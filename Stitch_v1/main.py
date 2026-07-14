"""图像拼接主流程：SIFT → USAC → Stitching。

管道概述：
    data1.jpg ──┐
                 ├── SIFT 特征检测 ── 暴力匹配 ── Ratio Test
    data2.jpg ──┘
                 │
                 ├── USAC 鲁棒估计 ── 单应性矩阵 H
                 │
                 └── 图像拼接 ────── 画布构建 → 反向映射 → 融合

数据复用：通过 sys.path 导入 ../SIFT/ 和 ../USAC/ 下的模块，
不修改其他文件夹内容。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# ---- 导入本模块 ----
_CURR_DIR = str(Path(__file__).parent)
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

# ---- 导入 USAC 模块（不修改 USAC 文件夹） ----
_USAC_DIR = str(Path(__file__).resolve().parent.parent / 'USAC')
if _USAC_DIR not in sys.path:
    sys.path.insert(1, _USAC_DIR)

# ---- 导入 SIFT 模块（不修改 SIFT 文件夹） ----
_SIFT_DIR = str(Path(__file__).resolve().parent.parent / 'SIFT')
if _SIFT_DIR not in sys.path:
    sys.path.insert(2, _SIFT_DIR)

from stitching_core import stitch
from visualize import (
    plot_pipeline_diagram,
    plot_canvas_visualization,
    plot_warping_result,
    plot_stitch_result,
)
from usac_core import usac

# SIFT 模块导入
from gaussian_pyramid import build_gaussian_pyramid
from dog_pyramid import build_dog_pyramid
from scale_space_extrema import detect_extrema
from keypoint_refinement import refine_keypoints
from orientation import assign_orientation
from descriptor import build_descriptor

# USAC matching
from matching import match_keypoints, extract_point_pairs


def _to_grayscale(image):
    """将 RGB 图转为灰度图。"""
    if image.ndim == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return image


def _load_image(image_path):
    """加载图像并归一化到 [0, 1] 范围。"""
    original = plt.imread(image_path)
    if original.max() > 1.0:
        original = original.astype(np.float64) / 255.0
    if original.ndim == 2:
        original = np.stack([original] * 3, axis=-1)
    elif original.shape[2] == 4:
        original = original[:, :, :3]
    return original


def process_single_image(image, num_octaves, num_intervals, sigma,
                         contrast_threshold, edge_threshold,
                         num_ori_bins, scale_factor):
    """对单张图像执行 SIFT 特征点检测完整流水线。

    返回 (gray, described_keypoints, gp, dp, sigs)。
    """
    gray = _to_grayscale(image)

    gp, sigs = build_gaussian_pyramid(
        gray, num_octaves, num_intervals, sigma)

    dp = build_dog_pyramid(gp)

    candidates = detect_extrema(dp, num_intervals, contrast_threshold)

    refined = refine_keypoints(
        dp, candidates, num_intervals, contrast_threshold, edge_threshold)

    oriented = assign_orientation(
        refined, gp, sigs, num_bins=num_ori_bins, scale_factor=scale_factor)

    described = build_descriptor(oriented, gp)

    return gray, described, gp, dp, sigs


def main(image1_path, image2_path, output_dir,
         num_octaves, num_intervals, sigma,
         contrast_threshold, edge_threshold,
         ratio_threshold,
         num_ori_bins, scale_factor,
         usac_threshold, usac_max_iter, usac_confidence,
         use_prosac, use_sprt, use_magsac, use_lo, use_adaptive,
         blend_mode):
    """运行完整的 SIFT → USAC → 图像拼接演示。"""
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('高光谱图像拼接演示：SIFT → USAC → Stitching')
    print('=' * 60)

    # ---- 加载图像 ----
    print(f'\n[Step 1] 加载图像')
    img1 = _load_image(image1_path)
    img2 = _load_image(image2_path)
    print(f'  图像 1: {image1_path} ({img1.shape[1]}×{img1.shape[0]})')
    print(f'  图像 2: {image2_path} ({img2.shape[1]}×{img2.shape[0]})')

    # ---- SIFT 特征检测 ----
    print(f'\n[Step 2] SIFT 特征点检测')
    gray1, desc1, gp1, dp1, sigs1 = process_single_image(
        img1, num_octaves, num_intervals, sigma,
        contrast_threshold, edge_threshold,
        num_ori_bins, scale_factor)
    print(f'  图像 1: {len(desc1)} 个 SIFT 特征点')

    gray2, desc2, gp2, dp2, sigs2 = process_single_image(
        img2, num_octaves, num_intervals, sigma,
        contrast_threshold, edge_threshold,
        num_ori_bins, scale_factor)
    print(f'  图像 2: {len(desc2)} 个 SIFT 特征点')

    # ---- 特征匹配 ----
    print(f'\n[Step 3] 特征匹配 (Ratio Test = {ratio_threshold})')
    if len(desc1) == 0 or len(desc2) == 0:
        print('  错误：特征点不足，无法匹配')
        return

    matches = match_keypoints(desc1, desc2, ratio_threshold=ratio_threshold)
    print(f'  Ratio Test 后: {len(matches)} 对匹配点')

    if len(matches) < 4:
        print('  错误：匹配点少于 4 对，无法进行几何估计')
        return

    pts1, pts2, qualities = extract_point_pairs(matches)
    N = len(matches)
    print(f'  有效匹配点: {N} 对')

    # ---- USAC 鲁棒估计 ----
    print(f'\n[Step 4] USAC 单应性矩阵估计')
    usac_components = []
    if use_prosac:
        usac_components.append('PROSAC')
    if use_sprt:
        usac_components.append('SPRT')
    if use_magsac:
        usac_components.append('MAGSAC')
    if use_lo:
        usac_components.append('LO-RANSAC')
    if use_adaptive:
        usac_components.append('自适应终止')
    print(f'  启用组件: {", ".join(usac_components)}')

    usac_result = usac(
        pts1, pts2,
        qualities=qualities,
        threshold=usac_threshold,
        max_iter=usac_max_iter,
        confidence=usac_confidence,
        use_prosac=use_prosac,
        use_sprt=use_sprt,
        use_magsac=use_magsac,
        use_lo=use_lo,
        use_adaptive=use_adaptive,
    )

    H = usac_result['H']
    print(f'  USAC 内点: {usac_result["inlier_count"]} / {N} '
          f'({usac_result["inlier_ratio"] * 100:.1f}%)')
    print(f'  内点平均重投影误差: {usac_result["mean_error"]:.3f} px')

    # ---- 图像拼接 ----
    print(f'\n[Step 5] 图像拼接 (融合模式: {blend_mode})')
    stitch_result = stitch(
        img1, img2, H, blend_mode=blend_mode,
        pts1=pts1, pts2=pts2, inlier_mask=usac_result['inlier_mask'])

    result = stitch_result['result']
    cb = stitch_result['canvas_bounds']
    print(f'  画布尺寸: {cb["canvas_width"]}×{cb["canvas_height"]}')
    print(f'  平移偏移: ({cb["offset_x"]}, {cb["offset_y"]})')
    print(f'  重叠像素: {stitch_result["overlap"].sum():.0f} 个')

    # 拼接结果归一化到 [0, 1] 显示
    result_display = np.clip(result, 0, 1)

    # ---- 可视化输出 ----
    print(f'\n[Step 6] 生成可视化结果')

    # 管道图
    plot_pipeline_diagram(os.path.join(output_dir, 'pipeline.png'))
    print(f'  pipeline.png — 拼接算法管道图')

    # 画布构建可视化
    plot_canvas_visualization(
        img1, img2, H, cb,
        os.path.join(output_dir, 'canvas_visualization.png'))
    print(f'  canvas_visualization.png — 画布构建可视化')

    # 映射过程可视化
    plot_warping_result(
        stitch_result['mask1'], stitch_result['mask2'],
        result_display,
        os.path.join(output_dir, 'warping_visualization.png'))
    print(f'  warping_visualization.png — 映射过程可视化')

    # 最终拼接结果
    plot_stitch_result(
        img1, img2, result_display, stitch_result['overlap'],
        matches, usac_result,
        os.path.join(output_dir, 'stitched_result.png'))
    print(f'  stitched_result.png — 最终拼接结果')

    # ---- 保存拼接结果 ----
    result_path = os.path.join(output_dir, 'panorama.jpg')
    plt.imsave(result_path, result_display)
    print(f'  panorama.jpg — 拼接全景图已保存')

    print(f'\n所有结果已保存至 {output_dir}/')
    print('=' * 60)

    return stitch_result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='高光谱图像拼接 — SIFT → USAC → Stitching',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--image1', default=None,
                        help='第一张输入图片路径（默认使用项目目录下的 data3.png）')
    parser.add_argument('--image2', default=None,
                        help='第二张输入图片路径（默认使用项目目录下的 data4.png）')
    parser.add_argument('--output-dir', default='result',
                        help='输出图片保存目录')
    parser.add_argument('--blend-mode', default='feather',
                        choices=['feather', 'direct'],
                        help='融合模式：feather（加权融合）或 direct（直接覆盖）')

    # SIFT 参数
    parser.add_argument('--num-octaves', type=int, default=4,
                        help='金字塔八度数')
    parser.add_argument('--num-intervals', type=int, default=3,
                        help='每 Octave 有效 Interval 数 s')
    parser.add_argument('--sigma', type=float, default=1.6,
                        help='初始尺度')
    parser.add_argument('--contrast-threshold', type=float, default=0.01,
                        help='对比度阈值')
    parser.add_argument('--edge-threshold', type=float, default=10.0,
                        help='边缘响应阈值 r')
    parser.add_argument('--ratio-threshold', type=float, default=0.75,
                        help='Ratio Test 阈值')
    parser.add_argument('--num-ori-bins', type=int, default=36,
                        help='方向直方图 bin 数')
    parser.add_argument('--scale-factor', type=float, default=1.5,
                        help='方向赋值高斯权重尺度因子')

    # USAC 参数
    parser.add_argument('--usac-threshold', type=float, default=3.0,
                        help='USAC 内点距离阈值（像素）')
    parser.add_argument('--usac-max-iter', type=int, default=2000,
                        help='USAC 最大迭代次数')
    parser.add_argument('--usac-confidence', type=float, default=0.99,
                        help='USAC 置信度')

    # USAC 组件开关
    parser.add_argument('--no-prosac', action='store_true',
                        help='禁用 PROSAC 渐进采样')
    parser.add_argument('--no-sprt', action='store_true',
                        help='禁用 SPRT 快速验证')
    parser.add_argument('--no-magsac', action='store_true',
                        help='禁用 MAGSAC 多阈值评分')
    parser.add_argument('--no-lo', action='store_true',
                        help='禁用 LO-RANSAC 局部优化')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='禁用自适应终止')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    script_dir = Path(__file__).parent

    image1 = args.image1 or str(script_dir / 'data3.png')
    image2 = args.image2 or str(script_dir / 'data4.png')

    os.makedirs(args.output_dir, exist_ok=True)

    main(
        image1_path=image1,
        image2_path=image2,
        output_dir=args.output_dir,
        num_octaves=args.num_octaves,
        num_intervals=args.num_intervals,
        sigma=args.sigma,
        contrast_threshold=args.contrast_threshold,
        edge_threshold=args.edge_threshold,
        ratio_threshold=args.ratio_threshold,
        num_ori_bins=args.num_ori_bins,
        scale_factor=args.scale_factor,
        usac_threshold=args.usac_threshold,
        usac_max_iter=args.usac_max_iter,
        usac_confidence=args.usac_confidence,
        use_prosac=not args.no_prosac,
        use_sprt=not args.no_sprt,
        use_magsac=not args.no_magsac,
        use_lo=not args.no_lo,
        use_adaptive=not args.no_adaptive,
        blend_mode=args.blend_mode,
    )
