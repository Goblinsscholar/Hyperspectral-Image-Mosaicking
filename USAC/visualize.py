"""USAC vs RANSAC 可视化对比工具。

生成对比图：
1. 内点分布对比 — 三栏显示 USAC（绿）vs RANSAC（红）vs 差异（共有/独有）
2. 重投影误差分布直方图对比（自适应分桶 + 阈值线）
3. 性能仪表盘：时间 / 内点 / 拒绝次数 / 收敛曲线 四合一
4. USAC 算法管道示意图
5. 内点比例对比柱状图
"""

import sys
from pathlib import Path

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


def plot_comparison_matches(image1, image2, keypoints1, keypoints2,
                            all_matches, usac_mask, ransac_mask,
                            save_path):
    """USAC vs RANSAC 内点分布对比图。

    三栏：USAC 内点（绿）| RANSAC 内点（红）| 差异对比（共有黄 / USAC独有绿 / RANSAC独有红）
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h = max(h1, h2)
    w_total = w1 + w2

    canvas = np.zeros((h, w_total, 3), dtype=np.float64)
    img1_d = image1[:, :, :3] if image1.ndim == 3 else np.stack([image1] * 3, axis=-1)
    img2_d = image2[:, :, :3] if image2.ndim == 3 else np.stack([image2] * 3, axis=-1)
    canvas[:h1, :w1] = img1_d
    canvas[:h2, w1:w_total] = img2_d

    usac_only = usac_mask & ~ransac_mask
    ransac_only = ransac_mask & ~usac_mask
    both = usac_mask & ransac_mask

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    titles = [
        f'USAC 内点（绿色）\n{int(np.sum(usac_mask))} 对',
        f'RANSAC 内点（红色）\n{int(np.sum(ransac_mask))} 对',
        f'USAC 独有（绿）{int(np.sum(usac_only))} | '
        f'RANSAC 独有（红）{int(np.sum(ransac_only))} | '
        f'共有（黄）{int(np.sum(both))}'
    ]

    for ax_idx, ax in enumerate(axes):
        ax.imshow(canvas)

        if ax_idx == 0:
            # USAC 内点
            for idx, m in enumerate(all_matches):
                if idx >= len(usac_mask) or not usac_mask[idx]:
                    continue
                kp1 = keypoints1[m['idx1']]
                kp2 = keypoints2[m['idx2']]
                x1 = kp1['x'] * (2 ** kp1['octave'])
                y1 = kp1['y'] * (2 ** kp1['octave'])
                x2 = kp2['x'] * (2 ** kp2['octave']) + w1
                y2 = kp2['y'] * (2 ** kp2['octave'])
                ax.plot([x1, x2], [y1, y2], color='green', linewidth=1.0, alpha=0.7)
                ax.scatter([x1], [y1], s=8, c='green', edgecolors='white', linewidths=0.2)
                ax.scatter([x2], [y2], s=8, c='green', edgecolors='white', linewidths=0.2)
        elif ax_idx == 1:
            # RANSAC 内点
            for idx, m in enumerate(all_matches):
                if idx >= len(ransac_mask) or not ransac_mask[idx]:
                    continue
                kp1 = keypoints1[m['idx1']]
                kp2 = keypoints2[m['idx2']]
                x1 = kp1['x'] * (2 ** kp1['octave'])
                y1 = kp1['y'] * (2 ** kp1['octave'])
                x2 = kp2['x'] * (2 ** kp2['octave']) + w1
                y2 = kp2['y'] * (2 ** kp2['octave'])
                ax.plot([x1, x2], [y1, y2], color='red', linewidth=1.0, alpha=0.7)
                ax.scatter([x1], [y1], s=8, c='red', edgecolors='white', linewidths=0.2)
                ax.scatter([x2], [y2], s=8, c='red', edgecolors='white', linewidths=0.2)
        else:
            # 差异对比
            for idx, m in enumerate(all_matches):
                kp1 = keypoints1[m['idx1']]
                kp2 = keypoints2[m['idx2']]
                x1 = kp1['x'] * (2 ** kp1['octave'])
                y1 = kp1['y'] * (2 ** kp1['octave'])
                x2 = kp2['x'] * (2 ** kp2['octave']) + w1
                y2 = kp2['y'] * (2 ** kp2['octave'])

                if idx < len(usac_mask) and idx < len(ransac_mask):
                    if usac_mask[idx] and ransac_mask[idx]:
                        color, lw = 'gold', 1.2
                    elif usac_mask[idx]:
                        color, lw = 'green', 1.2
                    elif ransac_mask[idx]:
                        color, lw = 'red', 1.0
                    else:
                        continue
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.8)
                    ax.scatter([x1], [y1], s=6, c=color, edgecolors='white', linewidths=0.2)
                    ax.scatter([x2], [y2], s=6, c=color, edgecolors='white', linewidths=0.2)

        ax.set_title(titles[ax_idx], fontsize=12)
        ax.axis('off')

    fig.suptitle('USAC vs RANSAC 内点分布对比', fontsize=16, fontweight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_error_comparison(usac_errors, ransac_errors,
                          usac_mask, ransac_mask,
                          save_path):
    """重投影误差分布直方图对比（双图并列，带阈值线）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    max_error = max(
        np.max(usac_errors[usac_errors < np.inf]) if np.any(usac_errors < np.inf) else 10,
        np.max(ransac_errors[ransac_errors < np.inf]) if np.any(ransac_errors < np.inf) else 10,
    ) + 1
    adaptive_bins = max(30, int(max_error * 2))
    bins = np.linspace(0, max_error, adaptive_bins)

    for ax_idx, (errors, mask, title) in enumerate([
        (usac_errors, usac_mask, 'USAC'),
        (ransac_errors, ransac_mask, 'RANSAC'),
    ]):
        ax = axes[ax_idx]
        inlier_err = errors[mask]
        outlier_err = errors[~mask] if np.any(~mask) else np.array([])

        if len(inlier_err) > 0:
            ax.hist(inlier_err, bins=bins, color='green', alpha=0.7,
                    label=f'内点 (n={len(inlier_err)}, '
                    f'均值={np.mean(inlier_err):.2f}px)')
        if len(outlier_err) > 0:
            ax.hist(outlier_err, bins=bins, color='red', alpha=0.4,
                    label=f'外点 (n={len(outlier_err)})')

        # 阈值线
        ax.axvline(3.0, color='blue', linestyle='--', linewidth=1.5, alpha=0.8,
                   label='阈值 = 3px')
        ax.set_title(f'{title} 重投影误差分布', fontsize=13)
        ax.set_xlabel('误差（像素）', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('重投影误差分布对比', fontsize=15, fontweight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_convergence_comparison(usac_stats, ransac_stats, save_path):
    """性能仪表盘：四合一 — 时间 / 内点 / 拒绝次数 / 收敛曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========== 左上：执行时间 + 迭代次数 ==========
    ax = axes[0, 0]
    metrics = ['执行时间 (s)', '迭代次数']
    usac_vals = [
        usac_stats.get('time_sec', 0),
        usac_stats.get('total_iterations', 0),
    ]
    ransac_vals = [
        ransac_stats.get('time_sec', 0),
        ransac_stats.get('total_iterations', 0),
    ]

    x = np.arange(len(metrics))
    width = 0.3
    bars1 = ax.bar(x - width / 2, usac_vals, width, color='green', alpha=0.75,
                   label='USAC', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, ransac_vals, width, color='red', alpha=0.6,
                   label='RANSAC', edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars1, usac_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(usac_vals + ransac_vals) * 0.02,
                f'{val:.3f}' if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
    for bar, val in zip(bars2, ransac_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(usac_vals + ransac_vals) * 0.02,
                f'{val:.3f}' if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

    speedup = ransac_vals[0] / usac_vals[0] if usac_vals[0] > 0 else 1
    ax.set_title(f'时间与迭代次数\nUSAC 耗时 {usac_vals[0]:.3f}s vs RANSAC {ransac_vals[0]:.3f}s', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ========== 右上：内点 / 外点 / 拒绝次数 ==========
    ax = axes[0, 1]
    metrics2 = ['内点', '外点', '拒绝次数']
    usac_in = usac_stats.get('inlier_count', 0)
    usac_out = usac_stats.get('outlier_count', 0)
    usac_rej = usac_stats.get('sprt_early_rejections', 0)
    ransac_in = ransac_stats.get('inlier_count', 0)
    ransac_out = ransac_stats.get('outlier_count', 0)
    ransac_rej = 0  # RANSAC 无拒绝机制

    usac_vals2 = [usac_in, usac_out, usac_rej]
    ransac_vals2 = [ransac_in, ransac_out, ransac_rej]

    x2 = np.arange(len(metrics2))
    bars3 = ax.bar(x2 - width / 2, usac_vals2, width, color='green', alpha=0.75,
                   label='USAC', edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x2 + width / 2, ransac_vals2, width, color='red', alpha=0.6,
                   label='RANSAC', edgecolor='black', linewidth=0.5)

    max_val = max(usac_vals2 + ransac_vals2) * 1.1
    for bar, val in zip(bars3, usac_vals2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.02,
                str(val), ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')
    for bar, val in zip(bars4, ransac_vals2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val * 0.02,
                str(val), ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    ax.set_title(f'内点 / 外点 / 拒绝次数\nUSAC 内点率 {usac_stats.get("final_inlier_ratio", 0) * 100:.1f}%'
                 f' vs RANSAC {ransac_stats.get("final_inlier_ratio", 0) * 100:.1f}%', fontsize=11)
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics2, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # ========== 左下：收敛曲线 ==========
    ax = axes[1, 0]
    usac_iters_over_time = usac_stats.get('iterations_over_time', [])
    usac_ratio_over_time = usac_stats.get('inlier_ratio_over_time', [])

    if len(usac_iters_over_time) > 0 and len(usac_ratio_over_time) > 0:
        ax.plot(usac_iters_over_time,
                [r * 100 for r in usac_ratio_over_time],
                'g-', linewidth=2, marker='o', markersize=4,
                label='USAC 收敛曲线')
        # 标注最终内点率
        ax.axhline(y=usac_ratio_over_time[-1] * 100, color='green',
                   linestyle=':', alpha=0.5)
        ax.text(len(usac_iters_over_time) - 1, usac_ratio_over_time[-1] * 100,
                f'  {usac_ratio_over_time[-1] * 100:.1f}%',
                color='green', fontsize=9, fontweight='bold')

    # RANSAC 无收敛曲线，用水平虚线标注最终内点率
    ransac_final_ratio = ransac_stats.get('final_inlier_ratio', 0) * 100
    ax.axhline(y=ransac_final_ratio, color='red', linestyle='--', linewidth=1.5,
               label=f'RANSAC 最终 ({ransac_final_ratio:.1f}%)')
    ax.text(0, ransac_final_ratio, f'  {ransac_final_ratio:.1f}%',
            color='red', fontsize=9, fontweight='bold')

    ax.set_xlabel('迭代次数（发现更优模型时）', fontsize=10)
    ax.set_ylabel('内点率 (%)', fontsize=10)
    ax.set_title('收敛过程对比\nUSAC 记录每次发现更优模型的时刻', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ========== 右下：平均误差 + 评估模型数 ==========
    ax = axes[1, 1]
    metrics3 = ['平均误差 (px)', '评估模型数']
    usac_vals3 = [
        usac_stats.get('mean_error', 0),
        usac_stats.get('models_evaluated', 0),
    ]
    ransac_vals3 = [
        ransac_stats.get('mean_error', 0),
        ransac_stats.get('models_evaluated', 0),
    ]

    x3 = np.arange(len(metrics3))
    bars5 = ax.bar(x3 - width / 2, usac_vals3, width, color='green', alpha=0.75,
                   label='USAC', edgecolor='black', linewidth=0.5)
    bars6 = ax.bar(x3 + width / 2, ransac_vals3, width, color='red', alpha=0.6,
                   label='RANSAC', edgecolor='black', linewidth=0.5)

    max_val3 = max(usac_vals3 + ransac_vals3) * 1.15 if max(usac_vals3 + ransac_vals3) > 0 else 1
    for bar, val in zip(bars5, usac_vals3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val3 * 0.02,
                f'{val:.3f}' if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
    for bar, val in zip(bars6, ransac_vals3):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_val3 * 0.02,
                f'{val:.3f}' if isinstance(val, float) else str(val),
                ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

    ax.set_title(f'精度与效率\n误差越小越优，评估数反映计算量', fontsize=11)
    ax.set_xticks(x3)
    ax.set_xticklabels(metrics3, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('USAC vs RANSAC 性能对比仪表盘', fontsize=15, fontweight='bold')
    fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.93)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pipeline_diagram(save_path):
    """绘制 USAC 算法管道示意图。"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis('off')

    stages = [
        ('输入图像', 0.8, 'lightgray', ''),
        ('高斯金字塔\nDoG 金字塔', 2.3, 'lightblue', 'SIFT 特征提取'),
        ('尺度空间极值\n亚像素定位', 3.8, 'lightblue', ''),
        ('主方向赋值\n128D 描述子', 5.3, 'lightblue', ''),
        ('暴力匹配\nRatio Test', 7.0, 'lightsteelblue', '特征匹配'),
        ('PROSAC\n渐进采样', 9.0, 'lightcoral', 'USAC 核心'),
        ('SPRT\n快速验证', 10.8, 'lightsalmon', ''),
        ('MAGSAC\n多阈值评分', 12.6, 'lightcoral', ''),
        ('LO-RANSAC\n局部优化', 14.4, 'lightsalmon', ''),
        ('自适应终止', 16.0, 'lightcoral', ''),
        ('最优模型\n可视化输出', 18.0, 'lightgreen', '结果'),
    ]

    y_center = 4.0
    box_w = 1.4
    box_h = 1.2

    for i, (label, x, color, desc) in enumerate(stages):
        rect = plt.Rectangle((x - box_w / 2, y_center - box_h / 2),
                             box_w, box_h,
                             facecolor=color, edgecolor='black',
                             linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)

        lines = label.split('\n')
        for li, line in enumerate(lines):
            ax.text(x, y_center + (len(lines) - 1) * 0.14 - li * 0.28,
                    line, ha='center', va='center', fontsize=7)

        if desc:
            ax.text(x, y_center - box_h / 2 - 0.3, desc,
                    ha='center', va='top', fontsize=8,
                    style='italic', color='gray')

    for i in range(len(stages) - 1):
        x1 = stages[i][1] + box_w / 2 + 0.05
        x2 = stages[i + 1][1] - box_w / 2 - 0.05
        ax.annotate('', xy=(x2, y_center), xytext=(x1, y_center),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.set_title('USAC 算法管道图', fontsize=16, fontweight='bold')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.8, label='SIFT 特征提取'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightsteelblue', alpha=0.8, label='特征匹配'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.8, label='USAC 核心组件'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightsalmon', alpha=0.8, label='USAC 辅助组件'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.8, label='结果输出'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=9)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_inlier_ratio_comparison(usac_mask, ransac_mask, save_path):
    """内点/外点比例对比柱状图（分组显示）。"""
    N = len(usac_mask)
    usac_in = int(np.sum(usac_mask))
    ransac_in = int(np.sum(ransac_mask))
    usac_out = N - usac_in
    ransac_out = N - ransac_in

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.arange(2)
    width = 0.3

    bars_in = ax.bar(x - width / 2, [usac_in, ransac_in], width,
                     color=['green', 'red'], alpha=0.75,
                     label='内点', edgecolor='black', linewidth=1)
    bars_out = ax.bar(x + width / 2, [usac_out, ransac_out], width,
                      color=['lightgreen', 'lightcoral'], alpha=0.5,
                      label='外点', edgecolor='black', linewidth=1)

    for bar, val in zip(bars_in, [usac_in, ransac_in]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')

    # 内点率标注
    usac_pct = usac_in / N * 100 if N > 0 else 0
    ransac_pct = ransac_in / N * 100 if N > 0 else 0
    ax.text(0, max(usac_in, usac_out) * 0.6, f'{usac_pct:.1f}%',
            ha='center', fontsize=14, fontweight='bold', color='green')
    ax.text(1, max(ransac_in, ransac_out) * 0.6, f'{ransac_pct:.1f}%',
            ha='center', fontsize=14, fontweight='bold', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(['USAC', 'RANSAC'], fontsize=12)
    ax.set_ylabel('匹配对数量', fontsize=12)
    ax.set_title(f'内点/外点比例对比', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
