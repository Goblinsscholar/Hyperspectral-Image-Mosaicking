"""SIFT 特征点检测与匹配主流程。

管道概述：
    data1.jpg ──┐
                 ├── 1. 高斯金字塔 ── 2. DoG 金字塔 ── 3. 尺度空间极值检测
    data2.jpg ──┘
                 │
                 ├── 4. 亚像素定位 ── 5. 低对比度/边缘剔除 ── 6. 主方向赋值
                 │
                 ├── 7. 128 维描述子构建（旋转归一 + 三线性插值 + L2 归一化）
                 │
                 ├── 8. 暴力匹配 + Ratio Test 筛选
                 │
                 └── 9. 可视化输出
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from gaussian_pyramid import build_gaussian_pyramid
from dog_pyramid import build_dog_pyramid
from scale_space_extrema import detect_extrema
from keypoint_refinement import refine_keypoints
from orientation import assign_orientation
from descriptor import build_descriptor
from matching import brute_force_match, ratio_test
from visualize import (plot_gaussian_pyramid, plot_dog_pyramid,
                       plot_keypoints, plot_matches, plot_pipeline_diagram)


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

    返回 (gray, described_keypoints, gp, dp) 元组。
    """
    gray = _to_grayscale(image)

    # 1. 高斯金字塔
    gp, sigs = build_gaussian_pyramid(
        gray, num_octaves, num_intervals, sigma)

    # 2. DoG 金字塔
    dp = build_dog_pyramid(gp)

    # 3. 尺度空间极值检测
    candidates = detect_extrema(dp, num_intervals, contrast_threshold)

    # 4. 关键点精炼（亚像素定位 + 低对比度 + 边缘剔除）
    refined = refine_keypoints(
        dp, candidates, num_intervals, contrast_threshold, edge_threshold)

    # 5. 主方向赋值
    oriented = assign_orientation(
        refined, gp, sigs, num_bins=num_ori_bins, scale_factor=scale_factor)

    # 6. 描述子构建
    described = build_descriptor(oriented, gp)

    return gray, described, gp, dp, sigs


def main(image1_path, image2_path, output_dir,
         num_octaves, num_intervals, sigma,
         contrast_threshold, edge_threshold,
         ratio_threshold,
         num_ori_bins, scale_factor):
    """运行完整的 SIFT 检测与匹配演示。"""
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 50)
    print('SIFT 特征点检测与匹配')
    print('=' * 50)

    # ---- 加载图像 ----
    print(f'\n[加载图像]')
    img1 = _load_image(image1_path)
    img2 = _load_image(image2_path)
    print(f'  图像 1: {image1_path} ({img1.shape[1]}×{img1.shape[0]})')
    print(f'  图像 2: {image2_path} ({img2.shape[1]}×{img2.shape[0]})')

    # ---- 处理图像 1 ----
    print(f'\n[处理图像 1]')
    gray1, desc1, gp1, dp1, sigs1 = process_single_image(
        img1, num_octaves, num_intervals, sigma,
        contrast_threshold, edge_threshold,
        num_ori_bins, scale_factor)
    print(f'  检测到 {len(desc1)} 个 SIFT 特征点')

    # ---- 处理图像 2 ----
    print(f'\n[处理图像 2]')
    gray2, desc2, gp2, dp2, sigs2 = process_single_image(
        img2, num_octaves, num_intervals, sigma,
        contrast_threshold, edge_threshold,
        num_ori_bins, scale_factor)
    print(f'  检测到 {len(desc2)} 个 SIFT 特征点')

    # ---- 特征匹配 ----
    print(f'\n[特征匹配]')
    if len(desc1) > 0 and len(desc2) > 0:
        descs1 = np.array([kp['descriptor'] for kp in desc1])
        descs2 = np.array([kp['descriptor'] for kp in desc2])

        all_matches = brute_force_match(descs1, descs2)
        good_matches = ratio_test(all_matches, threshold=ratio_threshold)
        print(f'  暴力匹配: {len(all_matches)} 对')
        print(f'  Ratio Test (阈值={ratio_threshold}): {len(good_matches)} 对')


    else:
        all_matches = []
        good_matches = []
        print(f'  特征点不足，跳过匹配')

    # ---- 可视化输出 ----
    print(f'\n[生成可视化结果]')

    # 根据图像2文件名生成后缀，区分不同图对
    img2_name = os.path.splitext(os.path.basename(image2_path))[0]
    suffix = f'_{img2_name}' if 'data2' not in img2_name else ''

    # 管道图
    plot_pipeline_diagram(os.path.join(output_dir, 'pipeline.png'))
    print(f'  pipeline.png — 管道图')

    # 高斯金字塔
    plot_gaussian_pyramid(gp1, sigs1, os.path.join(output_dir, 'gaussian_pyramid.png'))
    print(f'  gaussian_pyramid.png — 高斯金字塔')

    # DoG 金字塔
    plot_dog_pyramid(dp1, os.path.join(output_dir, 'dog_pyramid.png'))
    print(f'  dog_pyramid.png — DoG 金字塔')

    # 关键点检测结果
    plot_keypoints(img1, desc1, os.path.join(output_dir, 'sift_keypoints_img1.png'),
                   title=f'SIFT 特征点 — 图像 1（共 {len(desc1)} 个）')
    print(f'  sift_keypoints_img1.png — 图像 1 关键点')

    plot_keypoints(img2, desc2, os.path.join(output_dir, f'sift_keypoints{suffix}_img2.png'),
                   title=f'SIFT 特征点 — 图像 2（共 {len(desc2)} 个）')
    print(f'  sift_keypoints{suffix}_img2.png — 图像 2 关键点')

    # 匹配结果
    if good_matches:
        match_file = f'sift_matches{suffix}.png'
        plot_matches(img1, img2, desc1, desc2, good_matches,
                     os.path.join(output_dir, match_file),
                     title=f'SIFT 匹配（Ratio Test < {ratio_threshold}, {len(good_matches)} 对）')
        print(f'  {match_file} — 匹配连线图 ({len(good_matches)} 对)')

    print(f'\n所有结果已保存至 {output_dir}/')
    print('=' * 50)


def run_sweep(image1_path, image2_path, output_dir,
              param_name, param_values,
              num_octaves, num_intervals, sigma,
              contrast_threshold, edge_threshold,
              ratio_threshold,
              num_ori_bins, scale_factor):
    """参数扫描模式。"""
    os.makedirs(output_dir, exist_ok=True)

    img1 = _load_image(image1_path)
    img2 = _load_image(image2_path)

    results1 = []
    results2 = []

    print(f'\n[参数扫描] {param_name}')
    for val in param_values:
        _sigma = float(val) if param_name == 'sigma' else sigma
        _contrast = float(val) if param_name == 'contrast-threshold' else contrast_threshold
        _edge = float(val) if param_name == 'edge-threshold' else edge_threshold
        _ratio_t = float(val) if param_name == 'ratio-threshold' else ratio_threshold
        _intervals = int(val) if param_name == 'num-intervals' else num_intervals

        _, desc1, _, _, _ = process_single_image(
            img1, num_octaves, _intervals, _sigma,
            _contrast, _edge, num_ori_bins, scale_factor)
        _, desc2, _, _, _ = process_single_image(
            img2, num_octaves, _intervals, _sigma,
            _contrast, _edge, num_ori_bins, scale_factor)

        results1.append(desc1)
        results2.append(desc2)
        print(f'  {param_name}={val}: 图像1={len(desc1)} 个, 图像2={len(desc2)} 个')

    # 生成扫描对比图
    n = len(param_values)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, val in enumerate(param_values):
        kps1 = results1[col]
        kps2 = results2[col]

        ax1 = axes[0, col]
        ax1.imshow(img1)
        for kp in kps1:
            cx = kp['x'] * (2 ** kp['octave'])
            cy = kp['y'] * (2 ** kp['octave'])
            s = kp['s'] * 3
            circle = plt.Circle((cx, cy), s, color='red', fill=False, linewidth=0.5)
            ax1.add_patch(circle)
        ax1.set_title(f'{param_name}={val}\n({len(kps1)} 个)', fontsize=10)
        ax1.axis('off')

        ax2 = axes[1, col]
        ax2.imshow(img2)
        for kp in kps2:
            cx = kp['x'] * (2 ** kp['octave'])
            cy = kp['y'] * (2 ** kp['octave'])
            s = kp['s'] * 3
            circle = plt.Circle((cx, cy), s, color='red', fill=False, linewidth=0.5)
            ax2.add_patch(circle)
        ax2.set_title(f'({len(kps2)} 个)', fontsize=10)
        ax2.axis('off')

    axes[0, 0].set_ylabel('图像 1', fontsize=12)
    axes[1, 0].set_ylabel('图像 2', fontsize=12)

    save_path = os.path.join(output_dir, f'sweep_{param_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n扫描图已保存至 {save_path}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='SIFT 特征点检测与匹配教学演示工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--image1', default=None,
                        help='第一张输入图片路径（默认使用项目目录下的 data1.jpg）')
    parser.add_argument('--image2', default=None,
                        help='第二张输入图片路径（默认使用项目目录下的 data2.jpg）')
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
                        help='对比度阈值（原论文 0.03，本实现推荐 0.01）')
    parser.add_argument('--edge-threshold', type=float, default=10.0,
                        help='边缘响应阈值 r（原论文默认 10）')
    parser.add_argument('--ratio-threshold', type=float, default=0.7,
                        help='Ratio Test 阈值（原论文推荐 0.7-0.8）')
    parser.add_argument('--num-ori-bins', type=int, default=36,
                        help='方向直方图 bin 数')
    parser.add_argument('--scale-factor', type=float, default=1.5,
                        help='方向赋值高斯权重尺度因子')

    # 参数扫描
    parser.add_argument('--sweep', nargs='+', default=None,
                        metavar=('PARAM', 'VALUE'),
                        help=(
                            '参数扫描模式：指定参数名和多个取值，'
                            '如 --sweep sigma 1.0 1.6 2.0 2.5。'
                            '支持的参数：sigma, contrast-threshold, '
                            'edge-threshold, ratio-threshold, num-intervals'
                        ))

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    # 默认图片路径
    script_dir = Path(__file__).parent
    if args.image1 is None:
        args.image1 = str(script_dir / 'data1.jpg')
    if args.image2 is None:
        args.image2 = str(script_dir / 'data2.jpg')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep is not None:
        if len(args.sweep) < 2:
            print('错误：--sweep 需要至少指定一个参数名和两个取值')
            sys.exit(1)
        param_name = args.sweep[0]
        param_values = args.sweep[1:]
        run_sweep(
            image1_path=args.image1,
            image2_path=args.image2,
            output_dir=args.output_dir,
            param_name=param_name,
            param_values=param_values,
            num_octaves=args.num_octaves,
            num_intervals=args.num_intervals,
            sigma=args.sigma,
            contrast_threshold=args.contrast_threshold,
            edge_threshold=args.edge_threshold,
            ratio_threshold=args.ratio_threshold,
            num_ori_bins=args.num_ori_bins,
            scale_factor=args.scale_factor,
        )
    else:
        main(
            image1_path=args.image1,
            image2_path=args.image2,
            output_dir=args.output_dir,
            num_octaves=args.num_octaves,
            num_intervals=args.num_intervals,
            sigma=args.sigma,
            contrast_threshold=args.contrast_threshold,
            edge_threshold=args.edge_threshold,
            ratio_threshold=args.ratio_threshold,
            num_ori_bins=args.num_ori_bins,
            scale_factor=args.scale_factor,
        )
