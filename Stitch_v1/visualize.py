"""图像拼接可视化。

生成拼接管道的各阶段可视化：
1. 管道图 — 拼接算法流水线示意
2. 画布可视化 — 画布构建过程（角点投影 + 包围框）
3. 映射过程 — 两幅图像在画布上的映射状态
4. 拼接结果 — 最终全景图 + 重叠区高亮
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DengXian', 'Microsoft YaHei', 'SimHei',
                                    'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'


# ============================================================
# 1. 管道图
# ============================================================

def plot_pipeline_diagram(save_path):
    """生成拼接算法管道示意图。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 定义管道节点
    nodes = [
        (1.0, 4.5, '输入\n图像 A', '#E8F5E9'),
        (1.0, 1.5, '输入\n图像 B', '#E8F5E9'),
        (4.0, 3.0, 'SIFT\n特征检测', '#BBDEFB'),
        (6.5, 3.0, '特征匹配\nRatio Test', '#C5CAE9'),
        (9.0, 3.0, 'USAC\n鲁棒估计', '#FFE0B2'),
        (11.5, 3.0, '图像拼接\n画布→映射→融合', '#F8BBD0'),
        (13.0, 3.0, '全景图\n输出', '#C8E6C9'),
    ]

    # 绘制节点
    for x, y, label, color in nodes:
        rect = plt.Rectangle(
            (x - 0.8, y - 0.6), 1.6, 1.2,
            facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(rect)
        for i, line in enumerate(label.split('\n')):
            ax.text(x, y - 0.1 + i * 0.25, line,
                    ha='center', va='center', fontsize=9, fontweight='bold')

    # 绘制箭头
    for start, end in [
        ((2.6, 4.5), (3.2, 3.4)),
        ((2.6, 1.5), (3.2, 2.6)),
        ((4.8, 3.0), (5.7, 3.0)),
        ((7.3, 3.0), (8.2, 3.0)),
        ((9.8, 3.0), (10.7, 3.0)),
        ((12.3, 3.0), (12.2, 3.0)),
    ]:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#666666'))

    # 补充标注
    ax.text(4.0, 2.15, '尺度空间\n极值检测', ha='center', fontsize=7.5,
            color='#555555', style='italic')
    ax.text(6.5, 2.15, '暴力匹配\nLowe 比率', ha='center', fontsize=7.5,
            color='#555555', style='italic')
    ax.text(9.0, 2.15, 'PROSAC+SPRT\n+MAGSAC+LO', ha='center', fontsize=7.5,
            color='#555555', style='italic')
    ax.text(11.5, 2.15, '画布构建\n反向映射+融合', ha='center', fontsize=7.5,
            color='#555555', style='italic')

    ax.set_title('高光谱图像拼接管道 — SIFT → USAC → Stitching',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================
# 2. 画布构建可视化
# ============================================================

def plot_canvas_visualization(img1, img2, H, canvas_bounds, save_path):
    """可视化画布构建过程。

    显示：
    - 原始图像 A 和 B
    - 图像 A 变换后的角点位置
    - 最小包围画布边界
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 计算变换后的角点
    corners1 = np.array([
        [0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]
    ], dtype=np.float64)

    ones = np.ones((4, 1), dtype=np.float64)
    corners1_h = np.hstack([corners1, ones])
    transformed = (H @ corners1_h.T).T
    transformed = transformed / (transformed[:, 2:3] + 1e-10)
    corners1_trans = transformed[:, :2]

    corners2 = np.array([
        [0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]
    ], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 图 1: 原始图像
    ax = axes[0]
    ax.imshow(img1)
    ax.plot(corners1[:, 0], corners1[:, 1], 'ro', markersize=6)
    ax.plot([corners1[0, 0], corners1[1, 0], corners1[2, 0], corners1[3, 0], corners1[0, 0]],
            [corners1[0, 1], corners1[1, 1], corners1[2, 1], corners1[3, 1], corners1[0, 1]],
            'r--', linewidth=1, label='图像 A 角点')
    ax.set_title(f'图像 A ({w1}×{h1})', fontsize=12)
    ax.axis('off')

    # 图 2: 变换后的角点位置
    ax = axes[1]
    ax.imshow(img2, extent=[0, w2, h2, 0], alpha=0.7)

    # 图像 B 的角点（蓝色）
    ax.plot(corners2[:, 0], corners2[:, 1], 'bs', markersize=6, label='图像 B 角点')
    ax.plot([corners2[0, 0], corners2[1, 0], corners2[2, 0], corners2[3, 0], corners2[0, 0]],
            [corners2[0, 1], corners2[1, 1], corners2[2, 1], corners2[3, 1], corners2[0, 1]],
            'b--', linewidth=1)

    # 图像 A 变换后的角点（红色）
    ax.plot(corners1_trans[:, 0], corners1_trans[:, 1], 'ro', markersize=6, label='图像 A 变换后')
    ax.plot([corners1_trans[0, 0], corners1_trans[1, 0],
             corners1_trans[2, 0], corners1_trans[3, 0], corners1_trans[0, 0]],
            [corners1_trans[0, 1], corners1_trans[1, 1],
             corners1_trans[2, 1], corners1_trans[3, 1], corners1_trans[0, 1]],
            'r--', linewidth=1)

    # 画布边界
    rect = plt.Rectangle(
        (canvas_bounds['x_min'], canvas_bounds['y_min']),
        canvas_bounds['canvas_width'], canvas_bounds['canvas_height'],
        fill=False, edgecolor='green', linewidth=2, linestyle='-',
        label='最小画布')
    ax.add_patch(rect)

    ax.set_title('角点投影与画布边界', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('equal')
    ax.invert_yaxis()

    # 图 3: 最终画布示意
    ax = axes[2]
    canvas_preview = np.zeros((canvas_bounds['canvas_height'],
                               canvas_bounds['canvas_width'], 3), dtype=np.float64)

    # 在画布上绘制图像 B（平移后）
    ox, oy = canvas_bounds['offset_x'], canvas_bounds['offset_y']
    canvas_preview[oy:oy + h2, ox:ox + w2] = img2

    ax.imshow(canvas_preview)
    ax.set_title(f'最终画布 ({canvas_bounds["canvas_width"]}×'
                 f'{canvas_bounds["canvas_height"]})', fontsize=12)
    ax.text(canvas_bounds['canvas_width'] // 2, 15,
            f'偏移: ({ox}, {oy})', ha='center', fontsize=9,
            color='white', bbox=dict(facecolor='black', alpha=0.6))
    ax.axis('off')

    fig.suptitle('画布构建过程', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================
# 3. 映射过程可视化
# ============================================================

def plot_warping_result(mask1, mask2, result, save_path):
    """可视化两幅图像分别映射到画布的状态。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ch, cw = mask1.shape

    # 图 1: 图像 A 映射结果
    ax = axes[0]
    display = np.zeros((ch, cw, 3), dtype=np.float64)
    display[mask1] = [1.0, 0.4, 0.4]  # 红色调
    ax.imshow(display)
    inlier_count = np.sum(mask1)
    ax.set_title(f'图像 A 映射到画布\n有效像素: {inlier_count}', fontsize=11)
    ax.axis('off')

    # 图 2: 图像 B 映射结果
    ax = axes[1]
    display = np.zeros((ch, cw, 3), dtype=np.float64)
    display[mask2] = [0.4, 0.4, 1.0]  # 蓝色调
    ax.imshow(display)
    inlier_count = np.sum(mask2)
    ax.set_title(f'图像 B 映射到画布\n有效像素: {inlier_count}', fontsize=11)
    ax.axis('off')

    # 图 3: 重叠区域
    ax = axes[2]
    overlap = mask1 & mask2
    display = np.zeros((ch, cw, 3), dtype=np.float64)
    display[mask1 & ~overlap] = [1.0, 0.3, 0.3]  # 仅 A: 红
    display[mask2 & ~overlap] = [0.3, 0.3, 1.0]  # 仅 B: 蓝
    display[overlap] = [1.0, 1.0, 0.3]  # 重叠: 黄
    ax.imshow(display)
    ax.set_title(f'重叠区域\n重叠像素: {np.sum(overlap)}', fontsize=11)
    ax.axis('off')

    fig.suptitle('图像映射与重叠区域', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================
# 4. 最终拼接结果
# ============================================================

def plot_stitch_result(img1, img2, result, overlap,
                       matches, usac_result, save_path):
    """生成最终拼接结果四合一展示。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = usac_result['H']

    # ---- 图 1: 原始图像对 ----
    ax = axes[0, 0]
    gap = 10
    total_w = w1 + gap + w2
    composite = np.ones((max(h1, h2), total_w, 3), dtype=np.float64)
    composite[:h1, :w1] = img1
    composite[:h2, w1 + gap:w1 + gap + w2] = img2

    ax.imshow(composite)
    ax.axvline(w1 + gap / 2, color='gray', linewidth=1, linestyle=':')
    ax.text(w1 // 2, 15, '图像 A', ha='center', fontsize=10,
            color='white', bbox=dict(facecolor='red', alpha=0.6))
    ax.text(w1 + gap + w2 // 2, 15, '图像 B', ha='center', fontsize=10,
            color='white', bbox=dict(facecolor='blue', alpha=0.6))
    ax.set_title(f'原始图像对\n{w1}×{h1} + {w2}×{h2}', fontsize=11)
    ax.axis('off')

    # ---- 图 2: 匹配连线 ----
    ax = axes[0, 1]
    composite2 = np.ones((max(h1, h2), total_w, 3), dtype=np.float64)
    composite2[:h1, :w1] = img1
    composite2[:h2, w1 + gap:w1 + gap + w2] = img2
    ax.imshow(composite2)

    inlier_mask = usac_result['inlier_mask']
    for i, m in enumerate(matches):
        kp1 = m['kp1']
        kp2 = m['kp2']
        x1 = kp1['x'] * (2 ** kp1['octave'])
        y1 = kp1['y'] * (2 ** kp1['octave'])
        x2 = kp2['x'] * (2 ** kp2['octave']) + w1 + gap
        y2 = kp2['y'] * (2 ** kp2['octave'])

        color = 'green' if inlier_mask[i] else 'red'
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.4, alpha=0.6)

    ax.plot([], [], 'g-', linewidth=1, label=f'内点 ({usac_result["inlier_count"]})')
    ax.plot([], [], 'r-', linewidth=1, label=f'外点 ({usac_result["outlier_count"]})')
    ax.legend(fontsize=8)
    ax.set_title(f'匹配连线\nUSAC 内点/外点', fontsize=11)
    ax.axis('off')

    # ---- 图 3: 拼接结果 ----
    ax = axes[1, 0]
    ax.imshow(result)
    ax.set_title(f'拼接结果\n画布: {result.shape[1]}×{result.shape[0]}', fontsize=11)
    ax.axis('off')

    # ---- 图 4: 重叠区高亮 ----
    ax = axes[1, 1]
    display = result.copy()
    if overlap.any():
        # 在重叠区域叠加半透明黄色
        overlay = np.zeros_like(result)
        overlay[overlap] = [1.0, 1.0, 0.0]
        display = 0.7 * display + 0.3 * overlay
        display = np.clip(display, 0, 1)

    ax.imshow(display)
    ax.set_title(f'拼接结果 — 重叠区高亮\n'
                 f'重叠像素: {np.sum(overlap)}', fontsize=11)
    ax.axis('off')

    fig.suptitle('图像拼接结果展示', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================
# 5. 批量可视化
# ============================================================

def plot_all_results(stitch_result, img1, img2, matches, usac_result, output_dir):
    """一键生成所有可视化结果。"""
    os.makedirs(output_dir, exist_ok=True)

    plot_pipeline_diagram(os.path.join(output_dir, 'pipeline.png'))

    plot_canvas_visualization(
        img1, img2, usac_result['H'], stitch_result['canvas_bounds'],
        os.path.join(output_dir, 'canvas_visualization.png'))

    plot_warping_result(
        stitch_result['mask1'], stitch_result['mask2'],
        stitch_result['result'],
        os.path.join(output_dir, 'warping_visualization.png'))

    plot_stitch_result(
        img1, img2, stitch_result['result'], stitch_result['overlap'],
        matches, usac_result,
        os.path.join(output_dir, 'stitched_result.png'))

    return output_dir
