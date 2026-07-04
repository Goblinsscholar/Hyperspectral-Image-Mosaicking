"""RANSAC 误匹配剔除与单应性矩阵求解主流程。

管道概述：
    SIFT 特征检测与匹配（调用 ../SIFT/ 模块）
        │
        ├── 1. RANSAC 随机采样 4 对匹配点
        ├── 2. DLT 求解单应性矩阵
        ├── 3. 重投影误差判定内外点
        ├── 4. 迭代 N 次，取最优模型
        ├── 5. 最小二乘精炼单应性矩阵
        │
        └── 6. 可视化输出（匹配对比图、误差分布、内点标注等）
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# ---- 本模块导入（先于 SIFT 路径添加，避免命名冲突） ----
_CURR_DIR = str(Path(__file__).parent)
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

# ---- 导入 SIFT 管道（不修改 SIFT 文件夹） ----
_SIFT_DIR = str(Path(__file__).parent.parent / 'SIFT')
if _SIFT_DIR not in sys.path:
    sys.path.insert(1, _SIFT_DIR)

from ransac_core import (
    ransac, refine_homography, compute_reprojection_error,
)
from visualize import (
    plot_ransac_matches, plot_error_histogram, plot_pipeline_diagram,
    plot_keypoints_with_inliers, plot_inlier_ratio_bar,
    plot_ratio_test_matches, plot_ransac_inliers_only,
)

# 直接从 SIFT 模块导入
from gaussian_pyramid import build_gaussian_pyramid
from dog_pyramid import build_dog_pyramid
from scale_space_extrema import detect_extrema
from keypoint_refinement import refine_keypoints
from orientation import assign_orientation
from descriptor import build_descriptor
from matching import brute_force_match, ratio_test


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

    返回 (gray, described_keypoints, gp, dp, sigs) 元组。
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
         ransac_threshold, ransac_max_iter, ransac_confidence):
    """运行完整的 SIFT + RANSAC + 全景拼接演示。"""
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 55)
    print('RANSAC 误匹配剔除教学演示')
    print('=' * 55)

    # ---- 加载图像 ----
    print(f'\n[加载图像]')
    img1 = _load_image(image1_path)
    img2 = _load_image(image2_path)
    print(f'  图像 1: {image1_path} ({img1.shape[1]}×{img1.shape[0]})')
    print(f'  图像 2: {image2_path} ({img2.shape[1]}×{img2.shape[0]})')

    # ---- SIFT 特征检测与匹配 ----
    print(f'\n[Step 1] SIFT 特征点检测与匹配')

    print(f'\n  [处理图像 1]')
    gray1, desc1, gp1, dp1, sigs1 = process_single_image(
        img1, num_octaves, num_intervals, sigma,
        contrast_threshold, edge_threshold,
        num_ori_bins, scale_factor)
    print(f'    检测到 {len(desc1)} 个 SIFT 特征点')

    print(f'\n  [处理图像 2]')
    gray2, desc2, gp2, dp2, sigs2 = process_single_image(
        img2, num_octaves, num_intervals, sigma,
        contrast_threshold, edge_threshold,
        num_ori_bins, scale_factor)
    print(f'    检测到 {len(desc2)} 个 SIFT 特征点')

    print(f'\n  [特征匹配]')
    if len(desc1) > 0 and len(desc2) > 0:
        descs1 = np.array([kp['descriptor'] for kp in desc1])
        descs2 = np.array([kp['descriptor'] for kp in desc2])

        all_matches = brute_force_match(descs1, descs2)
        good_matches = ratio_test(all_matches, threshold=ratio_threshold)
        print(f'    暴力匹配: {len(all_matches)} 对')
        print(f'    Ratio Test (阈值={ratio_threshold}): {len(good_matches)} 对')

        if len(good_matches) < 4:
            print(f'\n  [警告] Ratio Test 后匹配点少于 4 对，无法进行 RANSAC。')
            print(f'  请尝试放宽 ratio-threshold 或增加更多特征点。')
            good_matches = []
    else:
        all_matches = []
        good_matches = []
        print(f'    特征点不足，跳过匹配')

    # ---- RANSAC 误匹配剔除 ----
    print(f'\n[Step 2] RANSAC 误匹配剔除')

    if len(good_matches) >= 4:
        # 提取匹配点坐标（原图分辨率）
        pts1 = []
        pts2 = []
        for m in good_matches:
            kp1 = desc1[m['idx1']]
            kp2 = desc2[m['idx2']]
            x1 = kp1['x'] * (2 ** kp1['octave'])
            y1 = kp1['y'] * (2 ** kp1['octave'])
            x2 = kp2['x'] * (2 ** kp2['octave'])
            y2 = kp2['y'] * (2 ** kp2['octave'])
            pts1.append([x1, y1])
            pts2.append([x2, y2])

        pts1 = np.array(pts1, dtype=np.float64)
        pts2 = np.array(pts2, dtype=np.float64)

        print(f'  RANSAC 输入: {len(good_matches)} 对匹配点')
        print(f'  内点阈值: {ransac_threshold} px, 最大迭代: {ransac_max_iter}, '
              f'置信度: {ransac_confidence}')

        # 执行 RANSAC
        result = ransac(
            pts1, pts2,
            threshold=ransac_threshold,
            max_iter=ransac_max_iter,
            confidence=ransac_confidence,
        )

        H_raw = result['H']
        inlier_mask = result['inlier_mask']
        inlier_count = result['inlier_count']
        iterations_used = result['iterations']

        print(f'  RANSAC 迭代次数: {iterations_used}')
        print(f'  内点: {inlier_count} 对 / {len(good_matches)} 对 '
              f'({inlier_count / len(good_matches) * 100:.1f}%)')
        print(f'  外点: {len(good_matches) - inlier_count} 对')

        if inlier_count >= 4:
            # 最小二乘精炼
            print(f'\n[Step 3] 单应性矩阵最小二乘精炼')
            H = refine_homography(pts1, pts2, inlier_mask)
            print(f'  精炼后单应性矩阵 H:')
            print(f'    [[{H[0, 0]:.6f}, {H[0, 1]:.6f}, {H[0, 2]:.4f}]')
            print(f'     [{H[1, 0]:.6f}, {H[1, 1]:.6f}, {H[1, 2]:.4f}]')
            print(f'     [{H[2, 0]:.6f}, {H[2, 1]:.6f}, {H[2, 2]:.4f}]')

            # 计算精炼后的重投影误差
            refined_errors = compute_reprojection_error(H, pts1, pts2)
            mean_error = np.mean(refined_errors[inlier_mask])
            max_error = np.max(refined_errors[inlier_mask])
            print(f'  内点平均重投影误差: {mean_error:.3f} px')
            print(f'  内点最大重投影误差: {max_error:.3f} px')

            refined_errors = compute_reprojection_error(H, pts1, pts2)

        else:
            print(f'\n  [警告] 内点不足 4 对，无法进行可靠的单应性矩阵精炼')
            H = H_raw
            refined_errors = compute_reprojection_error(H, pts1, pts2)

    else:
        print(f'  匹配点不足 4 对，跳过 RANSAC')
        inlier_mask = np.array([], dtype=bool)
        inlier_count = 0
        iterations_used = 0
        H = np.eye(3)
        pts1 = np.array([])
        pts2 = np.array([])
        refined_errors = np.array([])

    # ---- 可视化输出 ----
    print(f'\n[Step 5] 生成可视化结果')

    # 管道图
    plot_pipeline_diagram(os.path.join(output_dir, 'pipeline.png'))
    print(f'  pipeline.png — 含 RANSAC 的完整管道图')

    # Ratio Test 后的所有匹配（RANSAC 之前）
    if len(good_matches) > 0:
        plot_ratio_test_matches(
            img1, img2, desc1, desc2, good_matches,
            os.path.join(output_dir, 'ratio_test_matches.png'),
            title='Ratio Test 筛选后的匹配结果')
        print(f'  ratio_test_matches.png — Ratio Test 后匹配连线图 ({len(good_matches)} 对)')

    # RANSAC 匹配对比图
    if len(good_matches) >= 4:
        inl, outl = plot_ransac_matches(
            img1, img2, desc1, desc2, good_matches, inlier_mask,
            os.path.join(output_dir, 'ransac_matches.png'),
            title='RANSAC 误匹配剔除结果')
        print(f'  ransac_matches.png — RANSAC 匹配对比图 (内点 {inl}, 外点 {outl})')

        # 内点/外点比例图
        plot_inlier_ratio_bar(
            inl, outl,
            os.path.join(output_dir, 'inlier_ratio.png'))
        print(f'  inlier_ratio.png — 内点/外点比例图')

        # RANSAC 内点匹配结果（仅内点，清晰显示）
        plot_ransac_inliers_only(
            img1, img2, desc1, desc2, good_matches, inlier_mask,
            os.path.join(output_dir, 'ransac_inliers.png'),
            title='RANSAC 内点匹配结果')
        print(f'  ransac_inliers.png — RANSAC 内点匹配连线图（仅内点）')

        # 重投影误差直方图
        if len(refined_errors) > 0:
            plot_error_histogram(
                refined_errors, inlier_mask, ransac_threshold,
                os.path.join(output_dir, 'error_histogram.png'))
            print(f'  error_histogram.png — 重投影误差分布直方图')

        # 分别在两幅图像上标注内点/外点
        if len(desc1) > 0 and len(good_matches) > 0:
            inlier_indices1 = set()
            for i, m in enumerate(good_matches):
                if i < len(inlier_mask) and inlier_mask[i]:
                    inlier_indices1.add(m['idx1'])

            plot_keypoints_with_inliers(
                img1, desc1, inlier_indices=inlier_indices1,
                save_path=os.path.join(output_dir, 'inliers_img1.png'),
                title='图像 1 — 内点（绿色）/ 外点（红色）')
            print(f'  inliers_img1.png — 图像 1 内点/外点标注')

        if len(desc2) > 0 and len(good_matches) > 0:
            inlier_indices2 = set()
            for i, m in enumerate(good_matches):
                if i < len(inlier_mask) and inlier_mask[i]:
                    inlier_indices2.add(m['idx2'])

            plot_keypoints_with_inliers(
                img2, desc2, inlier_indices=inlier_indices2,
                save_path=os.path.join(output_dir, 'inliers_img2.png'),
                title='图像 2 — 内点（绿色）/ 外点（红色）')
            print(f'  inliers_img2.png — 图像 2 内点/外点标注')

    else:
        print(f'  匹配点不足，跳过 RANSAC 相关可视化')

    print(f'\n所有结果已保存至 {output_dir}/')
    print('=' * 55)


def run_sweep(image1_path, image2_path, output_dir,
              param_name, param_values,
              num_octaves, num_intervals, sigma,
              contrast_threshold, edge_threshold,
              ratio_threshold,
              num_ori_bins, scale_factor,
              ransac_threshold, ransac_max_iter, ransac_confidence):
    """参数扫描模式 — 扫描 RANSAC 相关参数。"""
    os.makedirs(output_dir, exist_ok=True)

    img1 = _load_image(image1_path)
    img2 = _load_image(image2_path)

    print(f'\n[参数扫描] {param_name}')
    results = []

    for val in param_values:
        _threshold = float(val) if param_name == 'threshold' else ransac_threshold
        _max_iter = int(val) if param_name == 'max-iter' else ransac_max_iter
        _confidence = float(val) if param_name == 'confidence' else ransac_confidence

        # SIFT 检测与匹配
        _, desc1, _, _, _ = process_single_image(
            img1, num_octaves, num_intervals, sigma,
            contrast_threshold, edge_threshold,
            num_ori_bins, scale_factor)
        _, desc2, _, _, _ = process_single_image(
            img2, num_octaves, num_intervals, sigma,
            contrast_threshold, edge_threshold,
            num_ori_bins, scale_factor)

        if len(desc1) == 0 or len(desc2) == 0:
            results.append({'val': val, 'inlier': 0, 'total': 0, 'ratio': 0.0})
            continue

        descs1 = np.array([kp['descriptor'] for kp in desc1])
        descs2 = np.array([kp['descriptor'] for kp in desc2])
        all_m = brute_force_match(descs1, descs2)
        good_m = ratio_test(all_m, threshold=ratio_threshold)

        if len(good_m) < 4:
            results.append({'val': val, 'inlier': 0, 'total': len(good_m), 'ratio': 0.0})
            continue

        pts1 = np.array([[desc1[m['idx1']]['x'] * (2 ** desc1[m['idx1']]['octave']),
                          desc1[m['idx1']]['y'] * (2 ** desc1[m['idx1']]['octave'])]
                         for m in good_m], dtype=np.float64)
        pts2 = np.array([[desc2[m['idx2']]['x'] * (2 ** desc2[m['idx2']]['octave']),
                          desc2[m['idx2']]['y'] * (2 ** desc2[m['idx2']]['octave'])]
                         for m in good_m], dtype=np.float64)

        r = ransac(pts1, pts2, threshold=_threshold,
                   max_iter=_max_iter, confidence=_confidence)

        results.append({
            'val': val,
            'inlier': int(r['inlier_count']),
            'total': len(good_m),
            'ratio': r['inlier_count'] / len(good_m) if len(good_m) > 0 else 0.0,
        })
        print(f'  {param_name}={val}: 内点 {r["inlier_count"]}/{len(good_m)} '
              f'({r["inlier_count"] / len(good_m) * 100:.1f}%)')

    # 生成扫描对比图
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, r in enumerate(results):
        ax1 = axes[0, col]
        ax1.bar(['内点', '外点'], [r['inlier'], r['total'] - r['inlier']],
                color=['green', 'red'], alpha=0.7)
        ax1.set_title(f'{param_name}={r["val"]}\n{r["inlier"]}/{r["total"]} '
                      f'({r["ratio"] * 100:.1f}%)', fontsize=10)
        ax1.set_ylim(0, max(r['total'] * 1.3, 5))
        ax1.grid(axis='y', alpha=0.3)

        ax2 = axes[1, col]
        ratio_val = r['ratio'] * 100
        ax2.bar(['内点比例'], [ratio_val], color='green', alpha=0.7)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel('%', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

    axes[0, 0].set_ylabel('匹配对数量', fontsize=12)
    axes[1, 0].set_ylabel('内点比例 (%)', fontsize=12)
    fig.suptitle(f'RANSAC 参数扫描 — {param_name}', fontsize=14, fontweight='bold')

    save_path = os.path.join(output_dir, f'sweep_{param_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n扫描图已保存至 {save_path}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='RANSAC 误匹配剔除教学演示工具',
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
    parser.add_argument('--ratio-threshold', type=float, default=0.7,
                        help='Ratio Test 阈值')
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

    # 参数扫描
    parser.add_argument('--sweep', nargs='+', default=None,
                        metavar=('PARAM', 'VALUE'),
                        help=(
                            '参数扫描模式：指定参数名和多个取值，'
                            '如 --sweep threshold 1.0 3.0 5.0。'
                            '支持的参数: threshold, max-iter, confidence'
                        ))

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()

    # 默认图片路径
    script_dir = Path(__file__).parent
    sift_dir = Path(__file__).parent.parent / 'SIFT'

    if args.image1 is None:
        args.image1 = str(sift_dir / 'data1.jpg')
    if args.image2 is None:
        args.image2 = str(sift_dir / 'data2.jpg')

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
            ransac_threshold=args.ransac_threshold,
            ransac_max_iter=args.ransac_max_iter,
            ransac_confidence=args.ransac_confidence,
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
            ransac_threshold=args.ransac_threshold,
            ransac_max_iter=args.ransac_max_iter,
            ransac_confidence=args.ransac_confidence,
        )
