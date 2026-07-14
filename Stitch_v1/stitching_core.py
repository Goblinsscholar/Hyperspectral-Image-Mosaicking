"""图像拼接核心算法。

实现完整的双图拼接流程：
1. 画布构建（Canvas Building）— 根据单应性矩阵计算目标画布尺寸
2. 反向映射（Backward/Inverse Warping）— 带双线性插值的像素采样
3. 图像融合（Image Blending）— 加权融合处理重叠区域

数据复用：通过 sys.path 导入其他模块，不修改其他文件夹内容。
"""

import numpy as np


# ============================================================
# 1. 画布构建
# ============================================================

def _transform_corners(H, width, height):
    """将图像四个角点通过单应性矩阵 H 变换到目标坐标系。

    参数:
        H: 3x3 单应性矩阵。
        width: 图像宽度。
        height: 图像高度。

    返回:
        (4, 2) numpy 数组，变换后的四个角点坐标 (x, y)。
    """
    corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float64)

    ones = np.ones((4, 1), dtype=np.float64)
    corners_h = np.hstack([corners, ones])

    transformed = (H @ corners_h.T).T
    transformed = transformed / (transformed[:, 2:3] + 1e-10)

    return transformed[:, :2]


def compute_canvas_bounds(H, img1_shape, img2_shape):
    """计算能够同时容纳两幅图像的最小画布边界。

    流程：
    1. 保持参考图像 img2 位置不变。
    2. 利用 H 计算 img1 四个顶点变换后的坐标。
    3. 将所有顶点（img2 的四个角 + img1 变换后的四个角）
       放到同一坐标系中，统计 x_min, x_max, y_min, y_max。
    4. 返回边界信息供后续平移和画布构建使用。

    参数:
        H: 3x3 单应性矩阵，描述 img1 → img2 的变换。
        img1_shape: (H1, W1) 或 (H1, W1, C)，图像 A 的尺寸。
        img2_shape: (H2, W2) 或 (H2, W2, C)，图像 B（参考图像）的尺寸。

    返回:
        dict:
            - 'x_min', 'x_max', 'y_min', 'y_max': 原始坐标范围
            - 'canvas_width', 'canvas_height': 画布宽高
            - 'offset_x', 'offset_y': 平移量（负坐标修正）
            - 'T': 3x3 平移矩阵
    """
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]

    # 图像 B（参考图像）的四个角点
    corners2 = np.array([
        [0, 0],
        [w2 - 1, 0],
        [w2 - 1, h2 - 1],
        [0, h2 - 1],
    ], dtype=np.float64)

    # 图像 A 变换后的四个角点
    corners1_trans = _transform_corners(H, w1, h1)

    # 合并所有顶点坐标
    all_corners = np.vstack([corners2, corners1_trans])

    x_min = np.floor(all_corners[:, 0].min()).astype(int)
    x_max = np.ceil(all_corners[:, 0].max()).astype(int)
    y_min = np.floor(all_corners[:, 1].min()).astype(int)
    y_max = np.ceil(all_corners[:, 1].max()).astype(int)

    # 平移量：确保无负坐标
    offset_x = -x_min
    offset_y = -y_min

    # 平移矩阵 T: 将所有点平移至非负区域
    T = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1],
    ], dtype=np.float64)

    canvas_width = x_max - x_min + 1
    canvas_height = y_max - y_min + 1

    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'canvas_width': canvas_width,
        'canvas_height': canvas_height,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'T': T,
    }


# ============================================================
# 2. 反向映射（Backward/Inverse Warping）
# ============================================================

def _bilinear_interpolate(image, x, y):
    """双线性插值采样。

    在原始图像 (x, y) 位置采样像素值。x, y 为浮点数，
    通过双线性插值获得平滑的像素值。

    参数:
        image: (H, W) 或 (H, W, C) numpy 数组。
        x: float，列坐标。
        y: float，行坐标。

    返回:
        插值后的像素值（标量或 (C,) 向量）。
    """
    h, w = image.shape[:2]
    is_color = image.ndim == 3

    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        if is_color:
            return np.zeros(image.shape[2], dtype=image.dtype)
        return 0.0

    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    dx = x - x0
    dy = y - y0

    if is_color:
        top = (1 - dx) * image[y0, x0] + dx * image[y0, x1]
        bottom = (1 - dx) * image[y1, x0] + dx * image[y1, x1]
        return (1 - dy) * top + dy * bottom
    else:
        top = (1 - dx) * image[y0, x0] + dx * image[y0, x1]
        bottom = (1 - dx) * image[y1, x0] + dx * image[y1, x1]
        return (1 - dy) * top + dy * bottom


def backward_warp(image, H, canvas_bounds):
    """反向映射：将图像通过单应性矩阵映射到画布上。

    对于画布中的每个整数像素位置，使用 H_inv 计算其在
    原始图像中的对应浮点坐标，然后通过双线性插值采样。

    H' = T @ H：将图像 A 先做单应性变换，再平移到画布位置。

    参数:
        image: (H, W) 或 (H, W, C) 原始图像。
        H: 3x3 单应性矩阵（图像 → 参考坐标系）。
        canvas_bounds: compute_canvas_bounds 返回的字典。

    返回:
        canvas: (canvas_height, canvas_width) 或
                (canvas_height, canvas_width, C) numpy 数组。
        mask: (canvas_height, canvas_width) bool 数组。
    """
    is_color = image.ndim == 3
    cw = canvas_bounds['canvas_width']
    ch = canvas_bounds['canvas_height']
    T = canvas_bounds['T']

    H_combined = T @ H
    H_inv = np.linalg.inv(H_combined)

    if is_color:
        num_channels = image.shape[2]
        canvas = np.zeros((ch, cw, num_channels), dtype=image.dtype)
    else:
        canvas = np.zeros((ch, cw), dtype=image.dtype)
    mask = np.zeros((ch, cw), dtype=bool)

    ys, xs = np.meshgrid(np.arange(ch), np.arange(cw), indexing='ij')
    ones = np.ones_like(xs)
    coords_h = np.stack([xs, ys, ones], axis=-1)

    mapped = H_inv @ coords_h[..., None]
    mapped = mapped[..., 0]
    mapped = mapped / (mapped[..., 2:3] + 1e-10)

    src_x = mapped[..., 0]
    src_y = mapped[..., 1]

    h_img, w_img = image.shape[:2]
    valid = (src_x >= 0) & (src_x < w_img - 1) & (src_y >= 0) & (src_y < h_img - 1)

    valid_y, valid_x = np.where(valid)
    for i in range(len(valid_y)):
        yi, xi = valid_y[i], valid_x[i]
        sx, sy = src_x[yi, xi], src_y[yi, xi]
        if is_color:
            canvas[yi, xi] = _bilinear_interpolate(image, sx, sy)
        else:
            canvas[yi, xi] = _bilinear_interpolate(image, sx, sy)
        mask[yi, xi] = True

    return canvas, mask


def translate_image(image, canvas_bounds):
    """平移参考图像到画布上。

    参考图像 B 不需要几何变换，仅需根据平移矩阵 T
    整体平移后复制像素值。

    参数:
        image: (H, W) 或 (H, W, C) 参考图像。
        canvas_bounds: compute_canvas_bounds 返回的字典。

    返回:
        canvas: 平移后的画布。
        mask: 有效区域掩码。
    """
    h, w = image.shape[:2]
    is_color = image.ndim == 3
    cw = canvas_bounds['canvas_width']
    ch = canvas_bounds['canvas_height']
    offset_x = canvas_bounds['offset_x']
    offset_y = canvas_bounds['offset_y']

    if is_color:
        num_channels = image.shape[2]
        canvas = np.zeros((ch, cw, num_channels), dtype=image.dtype)
    else:
        canvas = np.zeros((ch, cw), dtype=image.dtype)
    mask = np.zeros((ch, cw), dtype=bool)

    y_start = offset_y
    x_start = offset_x
    y_end = y_start + h
    x_end = x_start + w

    y_start_c = max(0, y_start)
    x_start_c = max(0, x_start)
    y_end_c = min(ch, y_end)
    x_end_c = min(cw, x_end)

    img_y_start = y_start_c - y_start
    img_x_start = x_start_c - x_start
    img_y_end = img_y_start + (y_end_c - y_start_c)
    img_x_end = img_x_start + (x_end_c - x_start_c)

    if is_color:
        canvas[y_start_c:y_end_c, x_start_c:x_end_c] = \
            image[img_y_start:img_y_end, img_x_start:img_x_end]
    else:
        canvas[y_start_c:y_end_c, x_start_c:x_end_c] = \
            image[img_y_start:img_y_end, img_x_start:img_x_end]

    mask[y_start_c:y_end_c, x_start_c:x_end_c] = True
    return canvas, mask


# ============================================================
# 3. 图像融合
# ============================================================

def feather_blend(canvas1, mask1, canvas2, mask2):
    """加权融合（Feather Blending）。

    重叠区域使用距离权重线性融合，非重叠区域直接复制。

    参数:
        canvas1: (H, W, C) 变换图像 A 的画布。
        mask1: (H, W) bool 数组，图像 A 的有效区域。
        canvas2: (H, W, C) 参考图像 B 的画布。
        mask2: (H, W) bool 数组，图像 B 的有效区域。

    返回:
        blended: (H, W, C) 融合后的最终图像。
        overlap_mask: (H, W) bool 数组，重叠区域为 True。
    """
    is_color = canvas1.ndim == 3
    blended = np.zeros_like(canvas1)
    overlap = mask1 & mask2

    if is_color:
        blended[mask1 & ~mask2] = canvas1[mask1 & ~mask2]
        blended[~mask1 & mask2] = canvas2[~mask1 & mask2]
    else:
        blended[mask1 & ~mask2] = canvas1[mask1 & ~mask2]
        blended[~mask1 & mask2] = canvas2[~mask1 & mask2]

    if overlap.any():
        dist1 = _compute_distance_to_boundary(mask1.astype(np.float64))
        dist2 = _compute_distance_to_boundary(mask2.astype(np.float64))
        weight_sum = dist1 + dist2 + 1e-10
        w1 = dist1 / weight_sum
        w2 = dist2 / weight_sum

        if is_color:
            for c in range(canvas1.shape[2]):
                blended[overlap, c] = (
                    w1[overlap] * canvas1[overlap, c] +
                    w2[overlap] * canvas2[overlap, c]
                )
        else:
            blended[overlap] = (
                w1[overlap] * canvas1[overlap] +
                w2[overlap] * canvas2[overlap]
            )

    return blended, overlap


def _compute_distance_to_boundary(mask_float, max_dist=50):
    """计算掩码中每个像素到边界的距离（腐蚀近似）。

    参数:
        mask_float: (H, W) float 数组，1.0=有效，0.0=无效。
        max_dist: 最大距离（腐蚀次数）。

    返回:
        dist: (H, W) float 数组，距离值归一化到 [0, 1]。
    """
    h, w = mask_float.shape
    dist = np.zeros((h, w), dtype=np.float64)
    current = mask_float.copy()

    for d in range(1, max_dist + 1):
        eroded = current.copy()
        eroded[1:, :] = np.minimum(eroded[1:, :], current[:-1, :])
        eroded[:-1, :] = np.minimum(eroded[:-1, :], current[1:, :])
        eroded[:, 1:] = np.minimum(eroded[:, 1:], current[:, :-1])
        eroded[:, :-1] = np.minimum(eroded[:, :-1], current[:, 1:])

        changed = (current > 0) & (eroded == 0)
        dist[changed] = d
        current = eroded
        if current.sum() == 0:
            break

    dist[current > 0] = max_dist
    dist = dist / max_dist
    return dist


def direct_copy_blend(canvas1, mask1, canvas2, mask2):
    """直接覆盖融合：重叠区域直接采用图像 A 的像素。"""
    blended = canvas2.copy()
    blended[mask1] = canvas1[mask1]
    overlap = mask1 & mask2
    return blended, overlap


# ============================================================
# 4. 单应性矩阵验证与鲁棒回退
# ============================================================

def is_homography_degenerate(H, eps=1e-3):
    """检查单应性矩阵是否退化（不可用）。

    退化判定条件：
    1. 左上 2×2 子矩阵的行列式绝对值过小（映射收缩到近零区域）
    2. 左上 2×2 子矩阵的行列式绝对值过大（映射爆炸）
    3. H[2,0] 或 H[2,1] 过大（过度透视扭曲）

    参数:
        H: 3x3 单应性矩阵。
        eps: 行列式阈值。

    返回:
        (bool, str): (是否退化, 原因描述)。
    """
    det_2x2 = np.linalg.det(H[:2, :2])
    if abs(det_2x2) < eps:
        return True, f'H 2x2行列式过小: {det_2x2:.6f}'
    if abs(det_2x2) > 1.0 / eps:
        return True, f'H 2x2行列式过大: {det_2x2:.6f}'
    # 检查透视分量（非零表示强透视扭曲）
    perspective_norm = np.sqrt(H[2, 0]**2 + H[2, 1]**2)
    if perspective_norm > 0.1:
        return True, f'H 透视分量过大: {perspective_norm:.6f}'
    return False, 'ok'


def _check_match_consistency(pts1, pts2, inlier_mask=None, spread_threshold=50):
    """检查匹配点的几何一致性。

    当匹配点在参考图像中高度集中时，说明匹配质量差，H不可靠。
    例如：多个 img1 点映射到 img2 中同一个像素 => 匹配无效。

    参数:
        pts1: (N, 2) 图像1中的点。
        pts2: (N, 2) 图像2中的点。
        inlier_mask: (N,) bool 数组，或 None。
        spread_threshold: 有效点云的扩散最小阈值（像素）。

    返回:
        (bool, str): (是否一致, 描述)。
    """
    if inlier_mask is not None and inlier_mask.sum() > 0:
        p1 = pts1[inlier_mask]
        p2 = pts2[inlier_mask]
    else:
        p1 = pts1
        p2 = pts2

    if len(p1) < 4:
        return False, f'有效匹配点不足: {len(p1)}'

    # 检查两个图像中点的空间扩散
    spread1 = np.sqrt(np.var(p1[:, 0]) + np.var(p1[:, 1]))
    spread2 = np.sqrt(np.var(p2[:, 0]) + np.var(p2[:, 1]))

    if spread1 < spread_threshold:
        return False, f'img1 内点过于集中 (spread={spread1:.1f}px)'
    if spread2 < spread_threshold:
        return False, f'img2 内点过于集中 (spread={spread2:.1f}px)'

    # 检查位移一致性
    shifts = p2 - p1
    shift_std = np.sqrt(np.var(shifts[:, 0]) + np.var(shifts[:, 1]))
    if shift_std > 300:
        return False, f'位移不一致 (std={shift_std:.1f}px)'

    return True, f'一致: spread1={spread1:.0f}px, spread2={spread2:.0f}px, shift_std={shift_std:.0f}px'



def estimate_homography_robust(pts1, pts2, inlier_mask=None):
    """从匹配点对通过 DLT 估计单应性矩阵（带归一化）。

    参数:
        pts1: (N, 2) 图像1中的点。
        pts2: (N, 2) 图像2中的点。
        inlier_mask: (N,) bool 数组。

    返回:
        H: 3x3 单应性矩阵，或 None（失败时）。
    """
    if inlier_mask is not None and inlier_mask.sum() >= 4:
        p1 = pts1[inlier_mask]
        p2 = pts2[inlier_mask]
    elif inlier_mask is None and pts1.shape[0] >= 4:
        p1 = pts1
        p2 = pts2
    else:
        return None

    N = p1.shape[0]

    # 归一化
    centroid1 = np.mean(p1, axis=0)
    centroid2 = np.mean(p2, axis=0)
    p1_centered = p1 - centroid1
    p2_centered = p2 - centroid2
    s1 = np.sqrt(2) / (np.mean(np.sqrt(np.sum(p1_centered**2, axis=1))) + 1e-10)
    s2 = np.sqrt(2) / (np.mean(np.sqrt(np.sum(p2_centered**2, axis=1))) + 1e-10)

    T1 = np.array([[s1, 0, -s1 * centroid1[0]],
                   [0, s1, -s1 * centroid1[1]],
                   [0, 0, 1]], dtype=np.float64)
    T2 = np.array([[s2, 0, -s2 * centroid2[0]],
                   [0, s2, -s2 * centroid2[1]],
                   [0, 0, 1]], dtype=np.float64)

    p1_norm = (T1 @ np.hstack([p1, np.ones((N, 1))]).T).T[:, :2]
    p2_norm = (T2 @ np.hstack([p2, np.ones((N, 1))]).T).T[:, :2]

    # DLT
    A = []
    for i in range(N):
        x, y = p1_norm[i]
        xp, yp = p2_norm[i]
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1, :].reshape(3, 3)

    H = np.linalg.inv(T2) @ H_norm @ T1
    H = H / H[2, 2]
    return H


def _estimate_translation_by_pixel_diff(img1, img2, max_shift=1000, step=5):
    """通过像素差异最小化来估计两图之间的平移量。

    当 SIFT 匹配失败时，直接比较图像像素值寻找最佳对齐。
    适用于两图为同一场景平移拍摄的情况。

    参数:
        img1: (H, W, C) 图像 A（变换图像）。
        img2: (H, W, C) 图像 B（参考图像）。
        max_shift: 最大搜索位移（像素）。
        step: 搜索步长。

    返回:
        H: 3x3 平移矩阵。
        score: 最佳匹配得分（越小越好）。
    """
    h, w = img1.shape[:2]
    g1 = np.dot(img1[..., :3], [0.299, 0.587, 0.114])
    g2 = np.dot(img2[..., :3], [0.299, 0.587, 0.114])

    best_score = float('inf')
    best_dx = 0

    # 只搜索水平方向（典型拼接场景）
    for dx in range(-max_shift, 0, step):
        # img1 右侧 vs img2 左侧
        overlap_w = w + dx  # dx negative → overlap = w - |dx|
        if overlap_w < 100:
            continue

        p1 = g1[:, -overlap_w:]
        p2 = g2[:, :overlap_w]
        score = np.abs(p1 - p2).mean()

        if score < best_score:
            best_score = score
            best_dx = dx

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = best_dx
    print(f'  [像素对齐] 最佳平移 dx={best_dx}, 平均像素差={best_score:.4f}')
    return H, best_score


# ============================================================
# 5. 拼接主函数（含鲁棒回退）
# ============================================================

def stitch(img1, img2, H, blend_mode='feather',
           pts1=None, pts2=None, inlier_mask=None):
    """执行完整的双图拼接流水线（含 H 退化检测与回退）。

    完整流程：
    1. 验证 H 是否退化，若退化则自动回退
    2. 根据 H 和两图像尺寸计算画布边界
    3. 将参考图像 img2 平移至画布
    4. 将变换图像 img1 通过反向映射投影到画布
    5. 根据融合策略融合两幅图像

    参数:
        img1: (H1, W1, C) numpy 数组，变换图像 A。
        img2: (H2, W2, C) numpy 数组，参考图像 B。
        H: 3x3 单应性矩阵（img1 → img2）。
        blend_mode: 融合模式，'feather'（默认）或 'direct'。
        pts1: (N, 2) 可选，图像1匹配点（用于 H 退化回退）。
        pts2: (N, 2) 可选，图像2匹配点（用于 H 退化回退）。
        inlier_mask: (N,) bool 可选，内点掩码。

    返回:
        dict:
            - 'result': (H, W, C) 拼接结果图像。
            - 'canvas_bounds': 画布边界信息。
            - 'mask1': 图像 A 的有效区域掩码。
            - 'mask2': 图像 B 的有效区域掩码。
            - 'overlap': 重叠区域掩码。
            - 'blend_mode': 使用的融合模式。
            - 'H_used': 实际使用的 H 矩阵。
            - 'fallback': 是否使用了回退 H。
    """
    # Step 0: 验证匹配点一致性（始终执行）
    matches_bad = False
    if pts1 is not None and pts2 is not None:
        consistent, cons_reason = _check_match_consistency(
            pts1, pts2, inlier_mask)
        if not consistent:
            print(f'  [警告] 匹配点不一致: {cons_reason}')
            matches_bad = True

    # Step 0b: 验证 H 是否退化
    degenerate, reason = is_homography_degenerate(H)
    fallback = False

    if degenerate or matches_bad:
        if degenerate:
            print(f'  [警告] H 退化: {reason}')
        if matches_bad:
            print(f'  [警告] 匹配点无有效几何关系')

        # 尝试基于像素差异的对齐（不依赖 SIFT 匹配）
        H_px, score = _estimate_translation_by_pixel_diff(img1, img2)
        _, px_reason = is_homography_degenerate(H_px)
        # 只有像素差异显著改善时才使用
        if score < 0.25 and abs(H_px[0, 2]) > 10:
            print(f'  [回退] 使用像素对齐 tx={H_px[0,2]:.1f}')
            H = H_px
            fallback = True
        else:
            print(f'  [提示] SIFT 匹配点间不存在有效的单应变换')
            print(f'  [回退] 使用单位矩阵（两图保持原位）')
            H = np.eye(3, dtype=np.float64)
            fallback = True

    # Step 1: 计算画布边界
    canvas_bounds = compute_canvas_bounds(H, img1.shape, img2.shape)

    # Step 2: 平移参考图像 img2
    canvas2, mask2 = translate_image(img2, canvas_bounds)

    # Step 3: 反向映射图像 img1
    canvas1, mask1 = backward_warp(img1, H, canvas_bounds)

    # Step 4: 融合
    if blend_mode == 'direct':
        result, overlap = direct_copy_blend(canvas1, mask1, canvas2, mask2)
    else:
        result, overlap = feather_blend(canvas1, mask1, canvas2, mask2)

    return {
        'result': result,
        'canvas_bounds': canvas_bounds,
        'mask1': mask1,
        'mask2': mask2,
        'overlap': overlap,
        'blend_mode': blend_mode,
        'H_used': H,
        'fallback': fallback,
    }
