"""Quick test for USAC."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent / 'SIFT'))

import numpy as np
from matplotlib import pyplot as plt
from gaussian_pyramid import build_gaussian_pyramid
from dog_pyramid import build_dog_pyramid
from scale_space_extrema import detect_extrema
from keypoint_refinement import refine_keypoints
from orientation import assign_orientation
from descriptor import build_descriptor
from matching import match_keypoints, extract_point_pairs


def _to_grayscale(image):
    if image.ndim == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return image


img1 = plt.imread('../SIFT/data1.jpg')
if img1.max() > 1.0: img1 = img1.astype(np.float64) / 255.0
if img1.ndim == 3 and img1.shape[2] == 4: img1 = img1[:, :, :3]

img2 = plt.imread('../SIFT/data2.jpg')
if img2.max() > 1.0: img2 = img2.astype(np.float64) / 255.0
if img2.ndim == 3 and img2.shape[2] == 4: img2 = img2[:, :, :3]

g1, sigs1 = build_gaussian_pyramid(_to_grayscale(img1), 4, 3, 1.6)
d1 = build_dog_pyramid(g1)
c1 = detect_extrema(d1, 3, 0.01)
r1 = refine_keypoints(d1, c1, 3, 0.01, 10.0)
o1 = assign_orientation(r1, g1, sigs1)
desc1 = build_descriptor(o1, g1)

g2, sigs2 = build_gaussian_pyramid(_to_grayscale(img2), 4, 3, 1.6)
d2 = build_dog_pyramid(g2)
c2 = detect_extrema(d2, 3, 0.01)
r2 = refine_keypoints(d2, c2, 3, 0.01, 10.0)
o2 = assign_orientation(r2, g2, sigs2)
desc2 = build_descriptor(o2, g2)

print(f'SIFT: {len(desc1)} + {len(desc2)} keypoints')

matches = match_keypoints(desc1, desc2, ratio_threshold=0.75)
print(f'Matches: {len(matches)}')

pts1, pts2, quals = extract_point_pairs(matches)
print(f'Points: {pts1.shape[0]}')

from usac_core import usac, MAGSACScorer

# Test MAGSAC threshold derivation
ms = MAGSACScorer(base_threshold=3.0)
print(f'MAGSAC sigmas: {ms.sigmas}')
print(f'MAGSAC thresholds: {ms.thresholds}')

s = time.time()
result = usac(pts1, pts2, qualities=quals, threshold=3.0, max_iter=500, confidence=0.99)
t = time.time() - s
print(f'USAC: {t:.3f}s, iters={result["iterations_used"]}, inliers={result["inlier_count"]}, mean_err={result["mean_error"]:.3f}px')
print(f'Stats: evaluated={result["stats"]["models_evaluated"]}, sprt_rej={result["stats"]["sprt_early_rejections"]}')
