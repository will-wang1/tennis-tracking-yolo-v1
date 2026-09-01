"""Sub-pixel refinement of a coarse court-keypoint estimate.

Port of yastrebksv/TennisCourtDetector's postprocess.py: the CNN's heatmap
localizes a keypoint to within a few pixels via a Hough-circle peak, then
this crops a small window around that estimate, detects the two court LINES
crossing it (Hough line transform), and refines the point to their exact
intersection - lines are thin, sharp, and antialiasing/heatmap blur affects
them less than it affects the blob-of-pixels the circle detector saw.
"""

import cv2
import numpy as np
from scipy.spatial import distance
from sympy import Line
from sympy.geometry.point import Point2D


def line_intersection(line1, line2) -> "tuple[float, float] | None":
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))
    intersection = l1.intersection(l2)
    if intersection and isinstance(intersection[0], Point2D):
        return intersection[0].coordinates
    return None


def detect_lines(image: np.ndarray) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines) if lines is not None else np.array([])
    if len(lines.shape) > 0 and len(lines) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines


def merge_lines(lines: list) -> list:
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []
    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1 :]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array(
                            [int((x1 + x3) / 2), int((y1 + y3) / 2), int((x2 + x4) / 2), int((y2 + y4) / 2)]
                        )
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines


def refine_kps(img: np.ndarray, y_ct: int, x_ct: int, crop_size: int = 40) -> tuple[float, float]:
    """`(y_ct, x_ct)` in, `(x_refined, y_refined)` out - matches the
    upstream repo's (row, col) in / (x, y) out convention exactly, since
    that mismatch is easy to reintroduce by "fixing" it."""
    refined_x, refined_y = x_ct, y_ct

    img_height, img_width = img.shape[:2]
    y_min, y_max = max(y_ct - crop_size, 0), min(img_height, y_ct + crop_size)
    x_min, x_max = max(x_ct - crop_size, 0), min(img_width, x_ct + crop_size)
    img_crop = img[y_min:y_max, x_min:x_max]

    lines = detect_lines(img_crop)
    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x, new_y = int(inters[0]), int(inters[1])
                if 0 < new_x < img_crop.shape[1] and 0 < new_y < img_crop.shape[0]:
                    refined_x = x_min + new_x
                    refined_y = y_min + new_y
    return float(refined_x), float(refined_y)
