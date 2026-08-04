"""Accumulate raw detections into a spatial density map."""

import numpy as np


def accumulate_heatmap(
    points: list[tuple[float, float]],
    weights: list[float],
    width: int,
    height: int,
    radius: int = 8,
) -> np.ndarray:
    """Splat each (x, y) point as a filled circle of `weight` into a
    `height` x `width` float array, so repeated hits at the same screen
    location accumulate into a bright spot - a persistent false-positive
    source shows up as a hotspot, a genuinely moving ball as a faint smear
    spread across the court.
    """
    heat = np.zeros((height, width), dtype=np.float64)
    yy, xx = np.ogrid[:height, :width]

    for (x, y), w in zip(points, weights):
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        heat[mask] += w

    return heat
