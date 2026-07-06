"""USAC vs RANSAC 对比演示主流程。

管道概述：
    data1.jpg ──┐
                 ├── SIFT 特征检测 ── 暴力匹配 ── Ratio Test
    data2.jpg ──┘
                 │
                 ├── RANSAC 基线 ──── 内点/外点判定
                 │
                 ├── USAC 管道 ────── PROSAC → SPRT → MAGSAC → LO → 自适应终止
                 │
                 └── 对比可视化 ───── 内点分布对比 / 误差分布 / 收敛曲线

数据复用：通过 sys.path 导入 ../SIFT/ 下的模块，不修改 SIFT 文件夹。
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

# ---- 导入 SIFT 模块（不修改 SIFT 文件夹） ----
_SIFT_DIR = str(Path(__file__).resolve().parent.parent / 'SIFT')
if _SIFT_DIR not in sys.path:
    sys.path.insert(1, _SIFT_DIR)

from usac_core import usac
from ransac_baseline import run_ransac_baseline
from matching import (
    match_keypoints, extract_point_pairs, brute_force_match, ratio_test,
)
from visualize import (
    plot_comparison_matches, plot_error_comparison,
    plot_convergence_comparison, plot_pipeline_diagram,
    plot_inlier_ratio_comparison,
)

# SIFT 模块导入
from gaussian_pyramid import build_gaussian_pyramid
from dog_pyramid import build_dog_pyramid
from scale_space_extrema import detect_extrema
from keypoint_refinement import refine_keypoints
from orientation import assign_orientation
from descriptor import build_descriptor


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
         ransac_threshold, ransac_max_iter, ransac_confidence,
         usac_threshold, usac_max_iter, usac_confidence,
         use_prosac, use_sprt, use_magsac, use_lo, use_adaptive):
    """运行完整的 USAC vs RANSAC 对比演示。"""
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('USAC vs RANSAC 对比演示')
    print('=' * 60)

    # ---- 加载图像 ----
    print(f'\n[加载图像]')
    img1 = _load_image(image1_path)
    img2 = _load_image(image2_path)
    print(f'  图像 1: {image1_path} ({img1.shape[1]}×{img1.shape[0]})')
    print(f'  图像 2: {image2_path} ({img2.shape[1]}×{img2.shape[0]})')

    # ---- SIFT 特征检测 ----
    print(f'\n[Step 1] SIFT 特征点检测')
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
    print(f'\n[Step 2] 特征匹配 (Ratio Test = {ratio_threshold})')
    if len(desc1) == 0 or len(desc2) == 0:
        print('  错误：特征点不足，无法匹配')
        return

    matches = match_keypoints(desc1, desc2, ratio_threshold=ratio_threshold)
    print(f'  Ratio Test 后: {len(matches)} 对匹配点')

    if len(matches) < 4:
        print('  错误：匹配点少于 4 对，无法进行几何验证')
        return

    # 提取点坐标和匹配质量
    pts1, pts2, qualities = extract_point_pairs(matches)
    N = len(matches)
    print(f'  有效匹配点: {N} 对')

    # ---- RANSAC 基线 ----
    print(f'\n[Step 3a] RANSAC 基线（阈值={ransac_threshold}, '
          f'最大迭代={ransac_max_iter}）')

    ransac_result = run_ransac_baseline(
        pts1, pts2,
        threshold=ransac_threshold,
        max_iter=ransac_max_iter,
        confidence=ransac_confidence,
    )

    print(f'  迭代次数: {ransac_result["iterations_used"]}')
    print(f'  内点: {ransac_result["inlier_count"]} / {N} '
          f'({ransac_result["inlier_ratio"] * 100:.1f}%)')
    print(f'  外点: {ransac_result["outlier_count"]} / {N}')
    print(f'  内点平均重投影误差: {ransac_result["mean_error"]:.3f} px')

    # ---- USAC 管道 ----
    print(f'\n[Step 3b] USAC 管道（阈值={usac_threshold}, '
          f'最大迭代={usac_max_iter}）')
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

    print(f'  迭代次数: {usac_result["iterations_used"]}')
    print(f'  内点: {usac_result["inlier_count"]} / {N} '
          f'({usac_result["inlier_ratio"] * 100:.1f}%)')
    print(f'  外点: {usac_result["outlier_count"]} / {N}')
    print(f'  内点平均重投影误差: {usac_result["mean_error"]:.3f} px')
    if 'stats' in usac_result:
        stats = usac_result['stats']
        print(f'  SPRT 提前拒绝: {stats.get("sprt_early_rejections", 0)} 次')
        print(f'  LO 局部优化: {stats.get("lo_refinements", 0)} 次')

    # ---- 对比总结 ----
    print(f'\n[对比总结]')
    print(f'  {"指标":<22} {"RANSAC":<14} {"USAC":<14} {"差异":<14}')
    print(f'  {"-" * 64}')
    print(f'  {"内点数量":<22} {ransac_result["inlier_count"]:<14} {usac_result["inlier_count"]:<14} '
          f'{usac_result["inlier_count"] - ransac_result["inlier_count"]:<+14}')
    print(f'  {"内点率":<22} {ransac_result["inlier_ratio"] * 100:<13.1f}% {usac_result["inlier_ratio"] * 100:<13.1f}% '
          f'{(usac_result["inlier_ratio"] - ransac_result["inlier_ratio"]) * 100:<+13.1f}%')
    print(f'  {"平均误差":<22} {ransac_result["mean_error"]:<13.3f}px {usac_result["mean_error"]:<13.3f}px '
          f'{usac_result["mean_error"] - ransac_result["mean_error"]:<+13.3f}')
    print(f'  {"迭代次数":<22} {ransac_result["iterations_used"]:<14} {usac_result["iterations_used"]:<14} '
          f'{usac_result["iterations_used"] - ransac_result["iterations_used"]:<+14}')
    print(f'  {"评估模型数":<22} {ransac_result["stats"]["models_evaluated"]:<14} '
          f'{usac_result["stats"]["models_evaluated"]:<14} '
          f'{usac_result["stats"]["models_evaluated"] - ransac_result["stats"]["models_evaluated"]:<+14}')
    print(f'  {"执行时间":<22} {ransac_result["time_sec"]:<13.4f}s {usac_result["time_sec"]:<13.4f}s '
          f'{usac_result["time_sec"] - ransac_result["time_sec"]:<+13.4f}')
    ransac_rej = ransac_result["stats"].get("sprt_early_rejections", 0)
    usac_rej = usac_result["stats"].get("sprt_early_rejections", 0)
    print(f'  {"拒绝次数":<22} {ransac_rej:<14} {usac_rej:<14} '
          f'{usac_rej - ransac_rej:<+14}')
    ransac_lo = ransac_result["stats"].get("lo_refinements", 0)
    usac_lo = usac_result["stats"].get("lo_refinements", 0)
    print(f'  {"局部优化次数":<22} {ransac_lo:<14} {usac_lo:<14} '
          f'{usac_lo - ransac_lo:<+14}')

    # ---- 可视化输出 ----
    print(f'\n[Step 4] 生成对比可视化结果')

    # 管道图
    plot_pipeline_diagram(os.path.join(output_dir, 'pipeline.png'))
    print(f'  pipeline.png — USAC 算法管道图')

    # 内点分布对比图（三栏）
    plot_comparison_matches(
        img1, img2, desc1, desc2, matches,
        usac_result['inlier_mask'], ransac_result['inlier_mask'],
        os.path.join(output_dir, 'usac_vs_ransac_matches.png'))
    print(f'  usac_vs_ransac_matches.png — USAC vs RANSAC 内点分布对比')

    # 重投影误差分布对比
    plot_error_comparison(
        usac_result['errors'], ransac_result['errors'],
        usac_result['inlier_mask'], ransac_result['inlier_mask'],
        os.path.join(output_dir, 'error_comparison.png'))
    print(f'  error_comparison.png — 重投影误差分布对比')

    # 收敛曲线与性能指标对比
    usac_result['stats']['final_inlier_ratio'] = usac_result['inlier_ratio']
    usac_result['stats']['mean_error'] = usac_result['mean_error']

    ransac_stats = ransac_result.get('stats', {})
    ransac_stats['time_sec'] = ransac_result['time_sec']
    ransac_stats['final_inlier_ratio'] = ransac_result['inlier_ratio']
    ransac_stats['mean_error'] = ransac_result['mean_error']
    ransac_stats['inlier_count'] = ransac_result['inlier_count']
    ransac_stats['outlier_count'] = ransac_result['outlier_count']

    usac_result['stats']['time_sec'] = usac_result['time_sec']
    usac_result['stats']['inlier_count'] = usac_result['inlier_count']
    usac_result['stats']['outlier_count'] = usac_result['outlier_count']

    plot_convergence_comparison(
        usac_result['stats'], ransac_stats,
        os.path.join(output_dir, 'convergence_comparison.png'))
    print(f'  convergence_comparison.png — 收敛曲线与性能对比')

    # 内点比例对比柱状图
    plot_inlier_ratio_comparison(
        usac_result['inlier_mask'], ransac_result['inlier_mask'],
        os.path.join(output_dir, 'inlier_ratio_comparison.png'))
    print(f'  inlier_ratio_comparison.png — 内点比例对比')

    print(f'\n所有结果已保存至 {output_dir}/')
    print('=' * 60)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='USAC vs RANSAC 对比教学演示工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--image1', default=None,
                        help='第一张输入图片路径（默认使用 ../SIFT/ 下的 data1.jpg）')
    parser.add_argument('--image2', default=None,
                        help='第二张输入图片路径（默认使用 ../SIFT/ 下的 data2.jpg）')
    parser.add_argument('--output-dir', default='result',
                        help='输出图片保存目录')

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
                        help='Ratio Test 阈值（USAC 可稍宽松）')
    parser.add_argument('--num-ori-bins', type=int, default=36,
                        help='方向直方图 bin 数')
    parser.add_argument('--scale-factor', type=float, default=1.5,
                        help='方向赋值高斯权重尺度因子')

    # RANSAC 参数
    parser.add_argument('--ransac-threshold', type=float, default=3.0,
                        help='RANSAC 内点距离阈值（像素）')
    parser.add_argument('--ransac-max-iter', type=int, default=2000,
                        help='RANSAC 最大迭代次数')
    parser.add_argument('--ransac-confidence', type=float, default=0.99,
                        help='RANSAC 置信度')

    # USAC 参数
    parser.add_argument('--usac-threshold', type=float, default=3.0,
                        help='USAC 内点距离阈值（像素）')
    parser.add_argument('--usac-max-iter', type=int, default=2000,
                        help='USAC 最大迭代次数（默认与 RANSAC 相同，确保对比公平）')
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

    # 默认图片路径
    script_dir = Path(__file__).parent
    sift_dir = Path(__file__).resolve().parent.parent / 'SIFT'

    image1 = args.image1 or str(sift_dir / 'data1.jpg')
    image2 = args.image2 or str(sift_dir / 'data2.jpg')

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
        ransac_threshold=args.ransac_threshold,
        ransac_max_iter=args.ransac_max_iter,
        ransac_confidence=args.ransac_confidence,
        usac_threshold=args.usac_threshold,
        usac_max_iter=args.usac_max_iter,
        usac_confidence=args.usac_confidence,
        use_prosac=not args.no_prosac,
        use_sprt=not args.no_sprt,
        use_magsac=not args.no_magsac,
        use_lo=not args.no_lo,
        use_adaptive=not args.no_adaptive,
    )
