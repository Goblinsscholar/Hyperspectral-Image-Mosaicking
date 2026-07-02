"""SIFT 可视化工具。

生成多种教学对比图：
1. 高斯金字塔可视化 — 多 Octave 并列显示
2. DoG 金字塔可视化 — 多 Octave 并列显示
3. 关键点检测结果 — 标注关键点位置、尺度、方向
4. 匹配连线图 — Ratio Test 前后的匹配对比
5. 参数扫描对比图
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


def _normalize(img):
    """将二维数组缩放到 [0, 1] 范围用于显示。"""
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin)


def plot_gaussian_pyramid(gaussian_pyramid, pyramid_sigmas, save_path,
                          max_octaves=3, max_layers=6):
    """可视化高斯金字塔。

    每 Octave 一行，展示各层 Gaussian 图像。
    """
    n_oct = min(len(gaussian_pyramid), max_octaves)
    n_layers = min(len(gaussian_pyramid[0]), max_layers)

    fig, axes = plt.subplots(n_oct, n_layers, figsize=(3 * n_layers, 3 * n_oct))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.3, wspace=0.2)

    if n_oct == 1:
        axes = [axes]
    if n_layers == 1:
        axes = [[a] for a in axes]

    for oi in range(n_oct):
        for li in range(n_layers):
            ax = axes[oi][li]
            if li < len(gaussian_pyramid[oi]):
                ax.imshow(gaussian_pyramid[oi][li], cmap='gray')
                sigma_val = pyramid_sigmas[oi][li]
                ax.set_title(f'Oct{oi} L{li}\nσ={sigma_val:.2f}', fontsize=8)
            ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dog_pyramid(dog_pyramid, save_path, max_octaves=3, max_layers=5):
    """可视化 DoG 金字塔。

    每 Octave 一行，展示各层 DoG 图像。
    """
    n_oct = min(len(dog_pyramid), max_octaves)
    n_layers = min(len(dog_pyramid[0]), max_layers)

    fig, axes = plt.subplots(n_oct, n_layers, figsize=(3 * n_layers, 3 * n_oct))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.3, wspace=0.2)

    if n_oct == 1:
        axes = [axes]
    if n_layers == 1:
        axes = [[a] for a in axes]

    for oi in range(n_oct):
        for li in range(n_layers):
            ax = axes[oi][li]
            if li < len(dog_pyramid[oi]):
                ax.imshow(_normalize(dog_pyramid[oi][li]), cmap='gray')
                ax.set_title(f'Oct{oi} D{li}', fontsize=8)
            ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_keypoints(image, keypoints, save_path, title='SIFT 关键点检测结果',
                   show_orientation=True):
    """在图像上标注关键点位置。

    圆半径表示特征尺度，线段方向表示主方向。
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    if keypoints:
        for kp in keypoints:
            x = kp['x']
            y = kp['y']
            # 转换坐标到原图分辨率
            oct_idx = kp['octave']
            scale_factor = 2 ** oct_idx
            x_orig = x * scale_factor
            y_orig = y * scale_factor
            s = kp['s'] * 3  # 可视化放大

            # 绘制尺度圆
            circle = plt.Circle((x_orig, y_orig), s, color='red',
                                fill=False, linewidth=1)
            ax.add_patch(circle)

            if show_orientation and 'orientation' in kp:
                # 绘制方向线段
                ori = kp['orientation']
                dx = s * np.cos(ori)
                dy = s * np.sin(ori)
                ax.arrow(x_orig, y_orig, dx, dy,
                         head_width=s * 0.3, head_length=s * 0.3,
                         fc='green', ec='green', linewidth=1.5)

    ax.set_title(title, fontsize=14)
    ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_matches(image1, image2, keypoints1, keypoints2, matches,
                 save_path, title='SIFT 特征匹配结果'):
    """绘制两幅图像之间的匹配连线。

    将两图左右拼接，匹配点之间用线段连接。
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w_total = w1 + w2

    canvas = np.zeros((h, w_total, 3), dtype=np.float64)

    if image1.ndim == 2:
        canvas[:h1, :w1] = np.stack([image1] * 3, axis=-1)
    else:
        canvas[:h1, :w1] = image1[:, :, :3]

    if image2.ndim == 2:
        canvas[:h2, w1:w_total] = np.stack([image2] * 3, axis=-1)
    else:
        canvas[:h2, w1:w_total] = image2[:, :, :3]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(canvas)

    # 绘制匹配线
    for m in matches:
        idx1 = m['idx1']
        idx2 = m['idx2']

        if idx1 >= len(keypoints1) or idx2 >= len(keypoints2):
            continue

        kp1 = keypoints1[idx1]
        kp2 = keypoints2[idx2]

        # 转换坐标到原图分辨率
        x1 = kp1['x'] * (2 ** kp1['octave'])
        y1 = kp1['y'] * (2 ** kp1['octave'])
        x2 = kp2['x'] * (2 ** kp2['octave']) + w1
        y2 = kp2['y'] * (2 ** kp2['octave'])

        # 根据比率着色（绿色好，红色差）
        ratio = m.get('ratio', 1.0)
        color = plt.cm.RdYlGn(1 - ratio) if ratio < 1.0 else 'red'

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.6)
        ax.scatter([x1], [y1], s=15, c='red', edgecolors='white', linewidths=0.3)
        ax.scatter([x2], [y2], s=15, c='blue', edgecolors='white', linewidths=0.3)

    ax.set_title(title, fontsize=14)
    ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pipeline_diagram(save_path):
    """绘制 SIFT 算法管道图。"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    steps = [
        ('输入图像', 0.5),
        ('高斯金字塔', 2.0),
        ('DoG 金字塔', 3.5),
        ('尺度空间\n极值检测', 5.0),
        ('亚像素定位\n低对比度剔除\n边缘响应剔除', 7.0),
        ('主方向赋值', 9.0),
        ('128 维\n描述子构建', 10.5),
        ('暴力匹配\nRatio Test', 12.5),
        ('匹配结果', 14.5),
    ]

    y_center = 3.0
    box_w = 1.3
    box_h = 1.0

    for i, (label, x) in enumerate(steps):
        rect = plt.Rectangle((x - box_w / 2, y_center - box_h / 2),
                             box_w, box_h,
                             facecolor='lightblue' if i > 0 else 'lightgray',
                             edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        # 支持多行标签
        lines = label.split('\n')
        for li, line in enumerate(lines):
            ax.text(x, y_center + (len(lines) - 1) * 0.12 - li * 0.24,
                    line, ha='center', va='center', fontsize=7)

    # 箭头
    for i in range(len(steps) - 1):
        x1 = steps[i][1] + box_w / 2
        x2 = steps[i + 1][1] - box_w / 2
        ax.annotate('', xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title('SIFT 特征检测与匹配管道图', fontsize=16, fontweight='bold')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
