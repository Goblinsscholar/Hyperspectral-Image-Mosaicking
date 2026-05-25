import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 动态检测可用的中文字体，找不到时自动回退
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
    # 无中文字体，标题将显示为英文回退
    import warnings
    warnings.warn(
        "未找到中文字体，图表标签可能无法正确显示。"
        "安装 Noto Sans SC 等字体可解决此问题。",
        stacklevel=2,
    )
plt.rcParams['axes.unicode_minus'] = False


def _normalize(img):
    """将二维数组缩放到 [0, 1] 范围用于显示。"""
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin)


def plot_harris_steps(original, Ix, Iy, products, M, corners, save_path):
    """创建灰度图 Harris 检测对比图并保存。

    布局（5 行 × 3 列）：
        第 1 行：原始灰度图
        第 2 行：I_x | I_y
        第 3 行：I_x² | I_x I_y | I_y²
        第 4 行：S_xx | S_xy | S_yy
        第 5 行：原始图上标注检测到的角点
    """
    fig, axes = plt.subplots(5, 3, figsize=(10, 14))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.97,
                        hspace=0.4, wspace=0.3)

    _title = lambda t: t

    # 第 1 行：原始灰度图（居中）
    axes[0, 1].imshow(original, cmap='gray')
    axes[0, 1].set_title(_title('原始灰度图'), fontsize=10)
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')

    # 第 2 行：Sobel 梯度
    axes[1, 0].imshow(_normalize(np.abs(Ix)), cmap='gray')
    axes[1, 0].set_title(r'$I_x$' + _title('（Sobel X 方向）'), fontsize=10)
    axes[1, 1].imshow(_normalize(np.abs(Iy)), cmap='gray')
    axes[1, 1].set_title(r'$I_y$' + _title('（Sobel Y 方向）'), fontsize=10)
    axes[1, 2].axis('off')

    # 第 3 行：梯度乘积（结构张量分量）
    axes[2, 0].imshow(_normalize(products['Ix2']), cmap='gray')
    axes[2, 0].set_title(r'$I_x^2$', fontsize=10)
    axes[2, 1].imshow(_normalize(products['Ixy']), cmap='gray')
    axes[2, 1].set_title(r'$I_x I_y$', fontsize=10)
    axes[2, 2].imshow(_normalize(products['Iy2']), cmap='gray')
    axes[2, 2].set_title(r'$I_y^2$', fontsize=10)

    # 第 4 行：M 矩阵
    axes[3, 0].imshow(_normalize(M['Sxx']), cmap='gray')
    axes[3, 0].set_title(r'$S_{xx}$', fontsize=10)
    axes[3, 1].imshow(_normalize(M['Sxy']), cmap='gray')
    axes[3, 1].set_title(r'$S_{xy}$', fontsize=10)
    axes[3, 2].imshow(_normalize(M['Syy']), cmap='gray')
    axes[3, 2].set_title(r'$S_{yy}$', fontsize=10)

    # 第 5 行：角点标注
    if len(corners) > 0:
        rgb = np.stack([original] * 3, axis=-1)
        axes[4, 1].imshow(rgb)
        axes[4, 1].scatter(corners[:, 1], corners[:, 0],
                           s=15, c='red', marker='o', edgecolors='white', linewidths=0.5)
        axes[4, 1].set_title(
            _title('角点检测结果（共 ') +
            f'{len(corners)}' +
            _title(' 个）'), fontsize=10)
    else:
        axes[4, 1].imshow(original, cmap='gray')
        axes[4, 1].set_title(_title('未检测到角点'), fontsize=10)
    axes[4, 0].axis('off')
    axes[4, 2].axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_multichannel_harris_steps(original_rgb, Rx, Ry, Gx, Gy, Bx, By,
                                   products, M, corners, save_path):
    """创建多通道（RGB）Harris 检测对比图并保存。

    布局（6 行 × 3 列）：
        第 1 行：原始彩色图
        第 2 行：R_x | G_x | B_x（各通道 Sobel X）
        第 3 行：R_y | G_y | B_y（各通道 Sobel Y）
        第 4 行：融合 I_x² | 融合 I_x I_y | 融合 I_y²
        第 5 行：S_xx | S_xy | S_yy
        第 6 行：原始图上标注检测到的角点
    """
    fig, axes = plt.subplots(6, 3, figsize=(10, 16))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.97,
                        hspace=0.4, wspace=0.3)

    _title = lambda t: t

    # 第 1 行：原始彩色图
    axes[0, 1].imshow(original_rgb)
    axes[0, 1].set_title(_title('原始彩色图'), fontsize=10)
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')

    # 第 2 行：各通道 Sobel X
    axes[1, 0].imshow(_normalize(np.abs(Rx)), cmap='gray')
    axes[1, 0].set_title(r'$R_x$', fontsize=10)
    axes[1, 1].imshow(_normalize(np.abs(Gx)), cmap='gray')
    axes[1, 1].set_title(r'$G_x$', fontsize=10)
    axes[1, 2].imshow(_normalize(np.abs(Bx)), cmap='gray')
    axes[1, 2].set_title(r'$B_x$', fontsize=10)

    # 第 3 行：各通道 Sobel Y
    axes[2, 0].imshow(_normalize(np.abs(Ry)), cmap='gray')
    axes[2, 0].set_title(r'$R_y$', fontsize=10)
    axes[2, 1].imshow(_normalize(np.abs(Gy)), cmap='gray')
    axes[2, 1].set_title(r'$G_y$', fontsize=10)
    axes[2, 2].imshow(_normalize(np.abs(By)), cmap='gray')
    axes[2, 2].set_title(r'$B_y$', fontsize=10)

    # 第 4 行：融合梯度乘积
    axes[3, 0].imshow(_normalize(products['Ix2']), cmap='gray')
    axes[3, 0].set_title(_title('融合 ') + r'$I_x^2$', fontsize=10)
    axes[3, 1].imshow(_normalize(products['Ixy']), cmap='gray')
    axes[3, 1].set_title(_title('融合 ') + r'$I_x I_y$', fontsize=10)
    axes[3, 2].imshow(_normalize(products['Iy2']), cmap='gray')
    axes[3, 2].set_title(_title('融合 ') + r'$I_y^2$', fontsize=10)

    # 第 5 行：M 矩阵
    axes[4, 0].imshow(_normalize(M['Sxx']), cmap='gray')
    axes[4, 0].set_title(r'$S_{xx}$', fontsize=10)
    axes[4, 1].imshow(_normalize(M['Sxy']), cmap='gray')
    axes[4, 1].set_title(r'$S_{xy}$', fontsize=10)
    axes[4, 2].imshow(_normalize(M['Syy']), cmap='gray')
    axes[4, 2].set_title(r'$S_{yy}$', fontsize=10)

    # 第 6 行：角点标注
    if len(corners) > 0:
        axes[5, 1].imshow(original_rgb)
        axes[5, 1].scatter(corners[:, 1], corners[:, 0],
                           s=15, c='red', marker='o', edgecolors='white', linewidths=0.5)
        axes[5, 1].set_title(
            _title('角点检测结果（共 ') +
            f'{len(corners)}' +
            _title(' 个）'), fontsize=10)
    else:
        axes[5, 1].imshow(original_rgb)
        axes[5, 1].set_title(_title('未检测到角点'), fontsize=10)
    axes[5, 0].axis('off')
    axes[5, 2].axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_figure(gray, original_rgb,
                           corners_g_no_nms, corners_g,
                           corners_c_no_nms, corners_c,
                           save_path):
    """创建灰度图 vs 多通道 RGB 总对比图。

    布局（3 行 × 2 列），对称对比两条流水线：
        第 1 行：原始图（灰度 | 彩色）
        第 2 行：角点叠加 — 无 NMS（大量响应） | 有 NMS（稀疏精准）
        第 3 行：同上，针对 RGB 路径
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.04, top=0.95,
                        hspace=0.35, wspace=0.3)

    _title = lambda t: t
    col_titles = [
        _title('灰度图路径'),
        _title('多通道 RGB 路径'),
    ]
    gray_rgb = np.stack([gray] * 3, axis=-1)

    # 第 1 行：原始图
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title(f'{col_titles[0]} — {_title("原始图")}', fontsize=10)
    axes[0, 1].imshow(original_rgb)
    axes[0, 1].set_title(f'{col_titles[1]} — {_title("原始图")}', fontsize=10)

    # 第 2 行：角点叠加 — 无 NMS vs 有 NMS（灰度图）
    axes[1, 0].imshow(gray_rgb)
    axes[1, 0].set_title(
        _title('灰度图路径 — ') +
        _title('无 NMS（') + f'{len(corners_g_no_nms)}' +
        _title(' 个）'), fontsize=10)
    if len(corners_g_no_nms) > 0:
        axes[1, 0].scatter(corners_g_no_nms[:, 1], corners_g_no_nms[:, 0],
                           s=3, c='red', alpha=0.3)
    axes[1, 1].imshow(gray_rgb)
    axes[1, 1].set_title(
        _title('灰度图路径 — ') +
        _title('有 NMS（') + f'{len(corners_g)}' +
        _title(' 个）'), fontsize=10)
    if len(corners_g) > 0:
        axes[1, 1].scatter(corners_g[:, 1], corners_g[:, 0],
                           s=15, c='red', marker='o', edgecolors='white', linewidths=0.5)

    # 第 3 行：角点叠加 — 无 NMS vs 有 NMS（RGB 路径）
    axes[2, 0].imshow(original_rgb)
    axes[2, 0].set_title(
        _title('RGB 路径 — ') +
        _title('无 NMS（') + f'{len(corners_c_no_nms)}' +
        _title(' 个）'), fontsize=10)
    if len(corners_c_no_nms) > 0:
        axes[2, 0].scatter(corners_c_no_nms[:, 1], corners_c_no_nms[:, 0],
                           s=3, c='red', alpha=0.3)
    axes[2, 1].imshow(original_rgb)
    axes[2, 1].set_title(
        _title('RGB 路径 — ') +
        _title('有 NMS（') + f'{len(corners_c)}' +
        _title(' 个）'), fontsize=10)
    if len(corners_c) > 0:
        axes[2, 1].scatter(corners_c[:, 1], corners_c[:, 0],
                           s=15, c='red', marker='o', edgecolors='white', linewidths=0.5)

    for ax in axes.flat:
        ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
