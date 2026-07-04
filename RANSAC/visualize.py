"""RANSAC 可视化工具。

生成多种教学对比图：
1. Ratio Test 后所有匹配连线图
2. RANSAC 前后匹配对比图 — 显示内点/外点
3. RANSAC 内点匹配结果图（仅内点，清晰显示）
4. 内点误差分布直方图
5. RANSAC 算法管道图
6. 关键点在各自图像上的标注
7. 内点/外点比例柱状图
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

# ---- 中文字体检测 ----
_CHINESE_FONTS = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC',
    'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    'Noto Sans SC', 'Source Han Sans SC',
]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = [f for f in _CHINESE_FONTS if f in _available]
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen[0]]
else:
    import warnings
    warnings.warn("未找到中文字体，图表标签可能无法正确显示。", stacklevel=2)
plt.rcParams['axes.unicode_minus'] = False


def plot_ransac_matches(image1, image2, keypoints1, keypoints2,
                         all_matches, inlier_mask,
                         save_path, title='RANSAC 误匹配剔除结果'):
    """绘制 RANSAC 筛选前后的匹配对比图。

    绿色连线 = 内点（正确匹配），红色连线 = 外点（误匹配）。

    参数:
        image1: (H1, W1, 3) 图像 1。
        image2: (H2, W2, 3) 图像 2。
        keypoints1: 图像 1 的关键点列表（需含 'x','y','octave'）。
        keypoints2: 图像 2 的关键点列表。
        all_matches: 所有匹配列表（含 idx1, idx2）。
        inlier_mask: (N,) bool 数组，内点标记。
        save_path: 保存路径。
        title: 图标题。
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w_total = w1 + w2

    canvas = np.zeros((h, w_total, 3), dtype=np.float64)

    img1_display = image1[:, :, :3] if image1.ndim == 3 else np.stack([image1] * 3, axis=-1)
    img2_display = image2[:, :, :3] if image2.ndim == 3 else np.stack([image2] * 3, axis=-1)

    canvas[:h1, :w1] = img1_display
    canvas[:h2, w1:w_total] = img2_display

    fig, ax = plt.subplots(1, 1, figsize=(18, 9))
    ax.imshow(canvas)

    inlier_count = 0
    outlier_count = 0

    for idx, m in enumerate(all_matches):
        kp1 = keypoints1[m['idx1']]
        kp2 = keypoints2[m['idx2']]

        x1 = kp1['x'] * (2 ** kp1['octave'])
        y1 = kp1['y'] * (2 ** kp1['octave'])
        x2 = kp2['x'] * (2 ** kp2['octave']) + w1
        y2 = kp2['y'] * (2 ** kp2['octave'])

        is_inlier = inlier_mask[idx] if idx < len(inlier_mask) else False

        color = 'green' if is_inlier else 'red'
        alpha = 0.8 if is_inlier else 0.3
        linewidth = 1.2 if is_inlier else 0.5

        if is_inlier:
            inlier_count += 1
        else:
            outlier_count += 1

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
        ax.scatter([x1], [y1], s=10, c=color, edgecolors='white', linewidths=0.3, alpha=alpha)
        ax.scatter([x2], [y2], s=10, c=color, edgecolors='white', linewidths=0.3, alpha=alpha)

    ax.set_title(f'{title}\n内点（绿色）: {inlier_count} 对 | 外点（红色）: {outlier_count} 对',
                 fontsize=14)
    ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return inlier_count, outlier_count


def plot_ratio_test_matches(image1, image2, keypoints1, keypoints2,
                             matches, save_path,
                             title='Ratio Test 筛选后的匹配结果'):
    """绘制 Ratio Test 筛选后的所有匹配点连线图（RANSAC 之前）。

    所有匹配点用统一颜色显示，每条线标注最近/次近邻距离比。

    参数:
        image1: (H1, W1, 3) 图像 1。
        image2: (H2, W2, 3) 图像 2。
        keypoints1: 图像 1 的关键点列表。
        keypoints2: 图像 2 的关键点列表。
        matches: Ratio Test 筛选后的匹配列表（含 idx1, idx2, ratio）。
        save_path: 保存路径。
        title: 图标题。
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w_total = w1 + w2

    canvas = np.zeros((h, w_total, 3), dtype=np.float64)

    img1_display = image1[:, :, :3] if image1.ndim == 3 else np.stack([image1] * 3, axis=-1)
    img2_display = image2[:, :, :3] if image2.ndim == 3 else np.stack([image2] * 3, axis=-1)

    canvas[:h1, :w1] = img1_display
    canvas[:h2, w1:w_total] = img2_display

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(canvas)

    for m in matches:
        kp1 = keypoints1[m['idx1']]
        kp2 = keypoints2[m['idx2']]

        x1 = kp1['x'] * (2 ** kp1['octave'])
        y1 = kp1['y'] * (2 ** kp1['octave'])
        x2 = kp2['x'] * (2 ** kp2['octave']) + w1
        y2 = kp2['y'] * (2 ** kp2['octave'])

        # 根据比率着色（比率越小匹配越好）
        ratio = m.get('ratio', 1.0)
        color = plt.cm.Blues(0.3 + 0.7 * (1 - ratio))

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.8, alpha=0.7)
        ax.scatter([x1], [y1], s=8, c='red', edgecolors='white', linewidths=0.2, alpha=0.6)
        ax.scatter([x2], [y2], s=8, c='blue', edgecolors='white', linewidths=0.2, alpha=0.6)

    ax.set_title(f'{title}（共 {len(matches)} 对）', fontsize=14)
    ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ransac_inliers_only(image1, image2, keypoints1, keypoints2,
                              matches, inlier_mask, save_path,
                              title='RANSAC 内点匹配结果'):
    """绘制 RANSAC 筛选后的内点匹配连线图（仅内点）。

    只显示通过 RANSAC 验证的正确匹配，绿色连线，画面清晰简洁。

    参数:
        image1: (H1, W1, 3) 图像 1。
        image2: (H2, W2, 3) 图像 2。
        keypoints1: 图像 1 的关键点列表。
        keypoints2: 图像 2 的关键点列表。
        matches: Ratio Test 筛选后的匹配列表。
        inlier_mask: (N,) bool 数组，内点标记。
        save_path: 保存路径。
        title: 图标题。
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w_total = w1 + w2

    canvas = np.zeros((h, w_total, 3), dtype=np.float64)

    img1_display = image1[:, :, :3] if image1.ndim == 3 else np.stack([image1] * 3, axis=-1)
    img2_display = image2[:, :, :3] if image2.ndim == 3 else np.stack([image2] * 3, axis=-1)

    canvas[:h1, :w1] = img1_display
    canvas[:h2, w1:w_total] = img2_display

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(canvas)

    inlier_count = 0
    for idx, m in enumerate(matches):
        if idx >= len(inlier_mask) or not inlier_mask[idx]:
            continue
        inlier_count += 1

        kp1 = keypoints1[m['idx1']]
        kp2 = keypoints2[m['idx2']]

        x1 = kp1['x'] * (2 ** kp1['octave'])
        y1 = kp1['y'] * (2 ** kp1['octave'])
        x2 = kp2['x'] * (2 ** kp2['octave']) + w1
        y2 = kp2['y'] * (2 ** kp2['octave'])

        ax.plot([x1, x2], [y1, y2], color='green', linewidth=1.2, alpha=0.9)
        ax.scatter([x1], [y1], s=15, c='green', edgecolors='white', linewidths=0.5)
        ax.scatter([x2], [y2], s=15, c='green', edgecolors='white', linewidths=0.5)

    ax.set_title(f'{title}（内点 {inlier_count} 对）', fontsize=14)
    ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_error_histogram(errors, inlier_mask, threshold, save_path):
    """绘制重投影误差分布直方图。

    参数:
        errors: (N,) numpy 数组，重投影误差。
        inlier_mask: (N,) bool 数组，内点标记。
        threshold: 内点阈值（像素）。
        save_path: 保存路径。
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    inlier_errors = errors[inlier_mask]
    outlier_errors = errors[~inlier_mask]

    bins = np.linspace(0, min(max(errors) + 1, 50), 50)

    if len(inlier_errors) > 0:
        ax.hist(inlier_errors, bins=bins, color='green', alpha=0.7,
                label=f'内点 (n={len(inlier_errors)})')
    if len(outlier_errors) > 0:
        ax.hist(outlier_errors, bins=bins, color='red', alpha=0.5,
                label=f'外点 (n={len(outlier_errors)})')

    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
               label=f'阈值 = {threshold} px')

    ax.set_xlabel('重投影误差（像素）', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('RANSAC 重投影误差分布', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pipeline_diagram(save_path):
    """绘制含 RANSAC 的完整拼接算法管道图。"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6)
    ax.axis('off')

    steps = [
        ('输入图像', 0.5),
        ('高斯金字塔', 1.8),
        ('DoG 金字塔', 3.0),
        ('尺度空间\n极值检测', 4.2),
        ('亚像素定位\n低对比度/边缘\n剔除', 5.5),
        ('主方向赋值', 6.8),
        ('128 维\n描述子构建', 8.0),
        ('暴力匹配\nRatio Test', 9.5),
        ('RANSAC\n误匹配剔除', 11.0),
        ('单应性矩阵\n最小二乘精炼', 12.5),
        ('内点/外点\n可视化输出', 14.0),
    ]

    y_center = 3.0
    box_w = 1.1
    box_h = 0.9

    for i, (label, x) in enumerate(steps):
        # RANSAC 高亮
        if i <= 7:
            color = 'lightblue'
        elif i == 8:
            color = 'lightcoral'  # RANSAC 高亮
        else:
            color = 'lightgreen'

        rect = plt.Rectangle((x - box_w / 2, y_center - box_h / 2),
                             box_w, box_h,
                             facecolor=color,
                             edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)

        lines = label.split('\n')
        for li, line in enumerate(lines):
            ax.text(x, y_center + (len(lines) - 1) * 0.12 - li * 0.24,
                    line, ha='center', va='center', fontsize=6.5)

    # 箭头
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + box_w / 2 + 0.05
        x2 = steps[i + 1][1] - box_w / 2 - 0.05
        ax.annotate('', xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title('RANSAC 误匹配剔除管道图',
                 fontsize=16, fontweight='bold')

    # 图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, label='SIFT 特征提取'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.8, label='RANSAC 几何验证'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.8, label='结果可视化'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_keypoints_with_inliers(image, keypoints, inlier_indices=None,
                                 save_path=None, title='特征点分布'):
    """在图像上标注特征点，可选标注哪些是内点。

    参数:
        image: 输入图像。
        keypoints: 关键点列表。
        inlier_indices: 内点在 keypoints 中的索引集合（set）。
        save_path: 保存路径。
        title: 图标题。
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image[:, :, :3])

    inlier_set = set(inlier_indices) if inlier_indices is not None else set()

    inlier_count = 0
    outlier_count = 0

    for idx, kp in enumerate(keypoints):
        x = kp['x'] * (2 ** kp['octave'])
        y = kp['y'] * (2 ** kp['octave'])
        s = kp['s'] * 3

        is_inlier = idx in inlier_set

        if is_inlier:
            color = 'green'
            inlier_count += 1
            circle = plt.Circle((x, y), s, color=color, fill=False, linewidth=1.5)
            ax.add_patch(circle)
        else:
            color = 'red'
            outlier_count += 1
            circle = plt.Circle((x, y), s * 0.5, color=color, fill=False, linewidth=0.5)
            ax.add_patch(circle)

    ax.set_title(f'{title}\n内点（绿色）: {inlier_count} | 外点（红色）: {outlier_count}',
                 fontsize=14)
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_inlier_ratio_bar(inlier_count, outlier_count, save_path):
    """绘制内点/外点比例柱状图。

    参数:
        inlier_count: 内点数量。
        outlier_count: 外点数量。
        save_path: 保存路径。
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    categories = ['内点（正确匹配）', '外点（误匹配）']
    counts = [inlier_count, outlier_count]
    colors = ['green', 'red']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

    total = inlier_count + outlier_count
    if total > 0:
        ax.set_title(f'RANSAC 内点/外点分布\n内点比例: {inlier_count / total * 100:.1f}%',
                     fontsize=14)
    else:
        ax.set_title('RANSAC 内点/外点分布', fontsize=14)

    ax.set_ylabel('匹配对数量', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
