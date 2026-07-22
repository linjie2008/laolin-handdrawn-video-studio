"""Image preprocessing and stroke extraction."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


Point = tuple[float, float]
NEIGHBORS_8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
DEFAULT_BORDER_MARGIN_RATIO = 0.012
MIN_BORDER_MARGIN_PX = 4
STROKE_DETAIL_PRESETS = {
    "balanced": {
        "threshold": 235,
        "min_points": 8,
        "min_length": 8.0,
        "smooth_spacing": None,
        "merge_gap": None,
        "merge": True,
        "max_merge_strokes": 1000,
    },
    "rich": {
        "threshold": 235,
        "min_points": 4,
        "min_length": "rich",
        "smooth_spacing": "rich",
        "merge_gap": "rich",
        "merge": True,
        "max_merge_strokes": 3200,
    },
    "max": {
        "threshold": 248,
        "min_points": 2,
        "min_length": 3.0,
        "smooth_spacing": "max",
        "merge_gap": "max",
        "merge": False,
        "max_merge_strokes": 0,
    },
}


@dataclass
class Stroke:
    """A drawable stroke path.

    ``widths`` is an optional per-point stroke diameter in canvas pixels.
    When set, the renderer draws variable-width ink so thick doodle marks and
    solid filled blocks reconstruct instead of collapsing into thin centerlines.
    ``width`` is a scalar fallback used when every point shares one thickness.

    Example:
        >>> Stroke(points=[(0, 0), (1, 1)]).source
        'raster'
    """

    points: list[Point]
    color: tuple[int, int, int] = (64, 60, 62)
    source: str = "raster"
    # Optional pen width in canvas pixels (uniform along the stroke).
    width: float | None = None
    # Optional per-point diameters (same length as points). Preferred over width.
    widths: list[float] | None = None

    def mean_width(self, default: float = 2.0) -> float:
        if self.widths:
            return float(sum(self.widths) / max(1, len(self.widths)))
        if self.width is not None and self.width > 0:
            return float(self.width)
        return default

    def max_width(self, default: float = 2.0) -> float:
        if self.widths:
            return float(max(self.widths))
        if self.width is not None and self.width > 0:
            return float(self.width)
        return default


def load_on_canvas(image_path: Path, canvas_size: tuple[int, int]) -> Image.Image:
    """Resize an image into a white canvas while preserving aspect ratio.

    Example:
        >>> load_on_canvas(Path("missing.png"), (640, 360))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
    """

    canvas, _ = load_on_canvas_with_bounds(image_path, canvas_size)
    return canvas


def load_on_canvas_with_bounds(image_path: Path, canvas_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Resize an image into a white canvas and return its placed bounds."""

    width, height = canvas_size
    src = Image.open(image_path).convert("RGBA")
    src.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    left, top = (width - src.width) // 2, (height - src.height) // 2
    canvas.alpha_composite(src, (left, top))
    return canvas.convert("RGB"), (left, top, left + src.width, top + src.height)


def binarize(img: np.ndarray, threshold: int = 235) -> np.ndarray:
    """Convert a grayscale image array into a boolean foreground mask.

    Example:
        >>> binarize(np.array([[255, 0]], dtype=np.uint8)).tolist()
        [[False, True]]
    """

    return img < threshold


def _border_margin_px(canvas_size: tuple[int, int], margin_px: int | None = None) -> int:
    if margin_px is not None:
        return max(0, int(margin_px))
    width, height = canvas_size
    return max(MIN_BORDER_MARGIN_PX, int(round(min(width, height) * DEFAULT_BORDER_MARGIN_RATIO)))


def _normalized_bounds(bounds: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = bounds
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _clear_frame_band(mask: np.ndarray, bounds: tuple[int, int, int, int], margin_px: int) -> None:
    x0, y0, x1, y1 = _normalized_bounds(bounds, mask.shape)
    if x1 <= x0 or y1 <= y0:
        return
    margin = min(margin_px, max(1, (x1 - x0) // 4), max(1, (y1 - y0) // 4))
    if margin <= 0:
        return

    mask[y0 : min(y1, y0 + margin), x0:x1] = False
    mask[max(y0, y1 - margin) : y1, x0:x1] = False
    mask[y0:y1, x0 : min(x1, x0 + margin)] = False
    mask[y0:y1, max(x0, x1 - margin) : x1] = False

    search = max(margin * 3, 16)
    pad = max(1, margin // 2)
    density_threshold = 0.75

    def clear_rows(row_indices: np.ndarray) -> None:
        for row in row_indices:
            mask[max(y0, row - pad) : min(y1, row + pad + 1), x0:x1] = False

    def clear_cols(col_indices: np.ndarray) -> None:
        for col in col_indices:
            mask[y0:y1, max(x0, col - pad) : min(x1, col + pad + 1)] = False

    top_end = min(y1, y0 + search)
    if top_end > y0 and x1 > x0:
        rows = np.where(mask[y0:top_end, x0:x1].mean(axis=1) >= density_threshold)[0] + y0
        clear_rows(rows)
    bottom_start = max(y0, y1 - search)
    if y1 > bottom_start and x1 > x0:
        rows = np.where(mask[bottom_start:y1, x0:x1].mean(axis=1) >= density_threshold)[0] + bottom_start
        clear_rows(rows)
    left_end = min(x1, x0 + search)
    if left_end > x0 and y1 > y0:
        cols = np.where(mask[y0:y1, x0:left_end].mean(axis=0) >= density_threshold)[0] + x0
        clear_cols(cols)
    right_start = max(x0, x1 - search)
    if x1 > right_start and y1 > y0:
        cols = np.where(mask[y0:y1, right_start:x1].mean(axis=0) >= density_threshold)[0] + right_start
        clear_cols(cols)


def suppress_canvas_border(
    mask: np.ndarray,
    bounds: tuple[int, int, int, int] | None = None,
    margin_px: int | None = None,
) -> np.ndarray:
    """Remove frame-like lines hugging the canvas or placed-image edges."""

    cleaned = mask.copy()
    if cleaned.size == 0:
        return cleaned
    h, w = cleaned.shape
    margin = _border_margin_px((w, h), margin_px)
    _clear_frame_band(cleaned, (0, 0, w, h), margin)
    if bounds is not None:
        _clear_frame_band(cleaned, bounds, margin)
    return cleaned


def quality_check(png_path: Path, canvas_size: tuple[int, int] = (1024, 1024), threshold: int = 235) -> float:
    """Return foreground pixel ratio for a candidate line-art image.

    Example:
        >>> 0 <= quality_check(Path("missing.png")) <= 1  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
    """

    source, bounds = load_on_canvas_with_bounds(png_path, canvas_size)
    mask = suppress_canvas_border(binarize(np.asarray(source.convert("L")), threshold), bounds=bounds)
    return float(np.mean(mask))


def zhang_suen_skeleton(mask: np.ndarray, max_iterations: int = 160) -> np.ndarray:
    """Thin a binary mask into a 1px skeleton using Zhang-Suen thinning.

    Example:
        >>> zhang_suen_skeleton(np.ones((3, 3), dtype=bool)).shape
        (3, 3)
    """

    img = np.pad(mask.astype(np.uint8), 1, mode="constant")
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            center = img[1:-1, 1:-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            transitions = sum((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8))
            count = sum(neighbors)
            if step == 0:
                marker = (
                    (center == 1)
                    & (count >= 2)
                    & (count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0)
                    & ((p4 * p6 * p8) == 0)
                )
            else:
                marker = (
                    (center == 1)
                    & (count >= 2)
                    & (count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0)
                    & ((p2 * p6 * p8) == 0)
                )
            if np.any(marker):
                center[marker] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary dilate without wrap-around (unlike np.roll)."""

    if radius <= 0 or mask.size == 0:
        return mask.copy()
    try:
        import cv2

        k = 2 * int(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        h, w = mask.shape
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        span = radius * 2 + 1
        for dy in range(span):
            for dx in range(span):
                out |= padded[dy : dy + h, dx : dx + w]
        return out


def _erode_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary erode without wrap-around."""

    if radius <= 0 or mask.size == 0:
        return mask.copy()
    try:
        import cv2

        k = 2 * int(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        inverted = ~mask
        grown = _dilate_mask(inverted, radius=radius)
        return ~grown & mask


def ink_radius_at(mask: np.ndarray, x: float, y: float, max_radius: int = 128) -> float:
    """Chebyshev distance from a point to the nearest background pixel.

    Used as local half-width for variable brush strokes. Skeleton points of a
    solid disk of radius R report ~R, so a diameter of 2R reconstructs the fill.
    """

    h, w = mask.shape
    cx = int(round(x))
    cy = int(round(y))
    if not (0 <= cx < w and 0 <= cy < h) or not mask[cy, cx]:
        return 0.5
    limit = max(1, min(int(max_radius), max(h, w)))
    for radius in range(1, limit + 1):
        y0, y1 = cy - radius, cy + radius
        x0, x1 = cx - radius, cx + radius
        if y0 < 0 or x0 < 0 or y1 >= h or x1 >= w:
            return float(radius - 1) + 0.5
        top = mask[y0, x0 : x1 + 1]
        bottom = mask[y1, x0 : x1 + 1]
        left = mask[y0 : y1 + 1, x0]
        right = mask[y0 : y1 + 1, x1]
        if not (bool(top.all()) and bool(bottom.all()) and bool(left.all()) and bool(right.all())):
            return float(radius - 1) + 0.5
    return float(limit)


def _widths_for_path(path: np.ndarray | list[Point], mask: np.ndarray, min_width: float = 1.5) -> list[float]:
    """Map path points to stroke diameters via 2 * local ink radius."""

    widths: list[float] = []
    for point in path:
        x, y = float(point[0]), float(point[1])
        width = max(min_width, min(240.0, ink_radius_at(mask, x, y) * 2.0))
        widths.append(width)
    return widths


def _widths_for_path_from_dist(
    path: np.ndarray | list[Point],
    dist: np.ndarray,
    min_width: float = 1.5,
    max_width: float = 240.0,
) -> list[float]:
    """Fast per-point diameters from a precomputed Euclidean distance map."""

    heights, width = dist.shape
    widths: list[float] = []
    for point in path:
        xi, yi = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= xi < width and 0 <= yi < heights:
            diameters = float(dist[yi, xi]) * 2.0
            widths.append(max(min_width, min(max_width, diameters)))
        else:
            widths.append(min_width)
    return widths


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels. Returns (labels, count). Background is 0."""

    try:
        import cv2

        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return labels.astype(np.int32), int(count) - 1
    except Exception:
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        count = 0
        ys, xs = np.nonzero(mask)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if labels[y, x]:
                continue
            count += 1
            stack = [(x, y)]
            labels[y, x] = count
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = count
                        stack.append((nx, ny))
        return labels, count


def _skel_neighbors(skel: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = point
    h, w = skel.shape
    result = []
    for dx, dy in NEIGHBORS_8:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < w and 0 <= ny < h and skel[ny, nx]):
            continue
        if dx != 0 and dy != 0 and (skel[y, nx] or skel[ny, x]):
            # An orthogonal bridge already connects these pixels. Ignoring the
            # redundant diagonal avoids tiny triangular strokes around T and
            # cross junctions while preserving true diagonal centerlines.
            continue
        result.append((nx, ny))
    return result


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _choose_next(
    prev: tuple[int, int],
    cur: tuple[int, int],
    candidates: list[tuple[int, int]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[int, int] | None:
    fresh = [p for p in candidates if _edge_key(cur, p) not in visited_edges and p != prev]
    if not fresh:
        return None
    vx, vy = cur[0] - prev[0], cur[1] - prev[1]
    return max(
        fresh,
        key=lambda p: ((vx * (p[0] - cur[0]) + vy * (p[1] - cur[1])) / (math.hypot(vx, vy) * math.hypot(p[0] - cur[0], p[1] - cur[1]) or 1.0)),
    )


def trace_8connected(skel: np.ndarray, min_points: int = 8) -> list[np.ndarray]:
    """Trace 8-connected skeleton pixels into continuous stroke arrays.

    At junctions, keep following the straightest unused edge instead of
    splitting every branch into a separate short stroke. Remaining branches
    are picked up by later starts. This better matches how a person would draw
    thick or jagged doodle contours with one continuous pen movement.

    Example:
        >>> trace_8connected(np.eye(8, dtype=bool), min_points=2)[0].shape[1]
        2
    """

    ys, xs = np.nonzero(skel)
    points = [(int(x), int(y)) for x, y in zip(xs, ys)]
    if not points:
        return []
    degrees = {p: len(_skel_neighbors(skel, p)) for p in points}
    starts = [p for p in points if degrees[p] == 1] + [p for p in points if degrees[p] > 2] + points
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    strokes: list[np.ndarray] = []
    for start in starts:
        for nb in _skel_neighbors(skel, start):
            edge = _edge_key(start, nb)
            if edge in visited_edges:
                continue
            path = [start]
            prev, cur = start, nb
            visited_edges.add(edge)
            while True:
                path.append(cur)
                next_pt = _choose_next(prev, cur, _skel_neighbors(skel, cur), visited_edges)
                if next_pt is None:
                    break
                visited_edges.add(_edge_key(cur, next_pt))
                prev, cur = cur, next_pt
            if len(path) >= min_points:
                strokes.append(np.asarray(path, dtype=np.int32))
    return strokes


def _freehand_fill_points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rng: np.random.Generator,
    *,
    wobble: float = 1.2,
    sample_step: float = 4.5,
) -> list[Point]:
    """Polyline along a segment with casual hand wobble (not a ruler line)."""

    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1.5:
        return [(x0, y0), (x1, y1)]
    n = max(2, int(round(length / max(2.0, sample_step))) + 1)
    # Gentle low-frequency drift + high-frequency jitter so it feels scribbled.
    phase = float(rng.uniform(0.0, math.tau))
    freq = float(rng.uniform(0.7, 1.6))
    amp = wobble * float(rng.uniform(0.55, 1.15))
    # Slight arc bias (wrist sweep) instead of pure sine.
    arc = float(rng.uniform(-0.35, 0.35)) * wobble
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -dy, dx  # unit normal
    points: list[Point] = []
    for i in range(n + 1):
        t = i / n
        # Ease ends so endpoints stay closer to the ink run.
        edge = math.sin(math.pi * t)
        along = t * length
        # Mild speed variation (hand pauses / rushes).
        along += amp * 0.15 * math.sin(phase + t * math.pi * 2.0)
        base_x = x0 + dx * (along if along <= length else length * t)
        base_y = y0 + dy * (along if along <= length else length * t)
        # Re-lerp for stability at ends.
        base_x = x0 + (x1 - x0) * t
        base_y = y0 + (y1 - y0) * t
        wob = amp * edge * math.sin(phase + t * math.pi * 2.0 * freq)
        wob += arc * edge * (4.0 * t * (1.0 - t))
        wob += float(rng.normal(0.0, amp * 0.18)) * edge
        points.append((base_x + nx * wob, base_y + ny * wob))
    # Dedup near-duplicates.
    cleaned: list[Point] = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 0.35:
            cleaned.append(p)
    if len(cleaned) < 2:
        return [(x0, y0), (x1, y1)]
    return cleaned


def _freehand_widths(
    n: int,
    base: float,
    rng: np.random.Generator,
    *,
    max_brush: float = 9.0,
) -> list[float]:
    """Per-point pressure: thicker mid-stroke, lighter ends, small jitter."""

    if n < 2:
        return [max(1.4, min(max_brush, base))]
    widths: list[float] = []
    pulse = float(rng.uniform(0.0, math.pi))
    for i in range(n):
        t = i / max(1, n - 1)
        # Calligraphic envelope: thin→fat→thin, a bit asymmetric.
        env = 0.78 + 0.28 * math.sin(math.pi * t) + 0.06 * math.sin(pulse + t * math.pi * 3.0)
        env *= float(rng.uniform(0.9, 1.1))
        w = max(1.4, min(max_brush, base * env))
        widths.append(w)
    return widths


def _freehand_scribble_strokes(
    mask: np.ndarray,
    dist: np.ndarray | None = None,
    min_area: int = 12,
    source: str = "ink-freehand",
    color: tuple[int, int, int] = (52, 50, 52),
    max_brush: float = 7.5,
) -> list[Stroke]:
    """Cover ink with continuous freehand pen strokes — no skeleton, no scan-brush.

    Each stroke is a meandering path that stays inside the ink mass, turns casually,
    and lifts when coverage is good enough. Looks like 随笔, not 刷子横扫 / 中心线骨架.
    """

    if not np.any(mask):
        return []
    if dist is None:
        dist = _distance_transform(mask)

    labels, count = _label_components(mask.astype(bool))
    strokes: list[Stroke] = []
    h, w = mask.shape
    seed = int(mask.sum()) ^ (h * 10007 + w * 17) ^ (count * 131)
    rng = np.random.default_rng(seed & 0xFFFFFFFF)

    # Downsampled coverage so we know where still needs ink without O(n²).
    cell = 3
    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(component)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        local_r = float(dist[component].mean()) if area else 2.0
        base_brush = max(1.7, min(max_brush, local_r * 1.1 + 0.5))
        step = max(1.5, min(3.6, base_brush * 0.5))

        cov_h = max(1, (y1 - y0) // cell + 2)
        cov_w = max(1, (x1 - x0) // cell + 2)
        covered = np.zeros((cov_h, cov_w), dtype=np.uint8)

        def cov_idx(px: float, py: float) -> tuple[int, int]:
            return (
                max(0, min(cov_h - 1, int((py - y0) // cell))),
                max(0, min(cov_w - 1, int((px - x0) // cell))),
            )

        # Target how many coverage cells should get a visit.
        ink_cells = 0
        cell_seeds: list[tuple[int, int]] = []
        for cy in range(cov_h):
            for cx in range(cov_w):
                # Any ink in this cell?
                yy0 = y0 + cy * cell
                xx0 = x0 + cx * cell
                patch = component[yy0 : min(h, yy0 + cell), xx0 : min(w, xx0 + cell)]
                if patch.size and bool(patch.any()):
                    ink_cells += 1
                    # Seed near cell center if on ink.
                    sy = min(h - 1, yy0 + cell // 2)
                    sx = min(w - 1, xx0 + cell // 2)
                    if component[sy, sx]:
                        cell_seeds.append((sx, sy))
                    else:
                        pys, pxs = np.nonzero(patch)
                        if len(pxs):
                            cell_seeds.append((int(xx0 + pxs[0]), int(yy0 + pys[0])))

        if not cell_seeds:
            continue

        target_cells = max(1, int(ink_cells * 0.93))
        max_paths = max(4, min(240, int(ink_cells * 0.65) + 6))
        visited_cells = 0
        path_i = 0
        seed_order = list(range(len(cell_seeds)))
        rng.shuffle(seed_order)
        seed_cursor = 0

        def pick_start() -> tuple[float, float] | None:
            nonlocal seed_cursor
            # Prefer uncovered seeds, then any.
            for _ in range(len(seed_order)):
                sx, sy = cell_seeds[seed_order[seed_cursor % len(seed_order)]]
                seed_cursor += 1
                ci, cj = cov_idx(float(sx), float(sy))
                if covered[ci, cj] == 0 and component[sy, sx]:
                    return float(sx), float(sy)
            # Fallback: random ink pixel in bbox.
            for _ in range(24):
                xi = int(rng.integers(x0, x1 + 1))
                yi = int(rng.integers(y0, y1 + 1))
                if component[yi, xi]:
                    return float(xi), float(yi)
            return None

        while visited_cells < target_cells and path_i < max_paths:
            start = pick_start()
            if start is None:
                break
            x, y = start
            # Initial direction: random, slightly biased along longer bbox axis.
            if (x1 - x0) >= (y1 - y0):
                ang = float(rng.uniform(-0.45, 0.45)) + float(rng.choice([0.0, math.pi]))
            else:
                ang = float(rng.uniform(-0.45, 0.45)) + float(rng.choice([math.pi * 0.5, -math.pi * 0.5]))
            vx, vy = math.cos(ang), math.sin(ang)
            points: list[Point] = [(x, y)]
            # Variable stroke length — short flicks and longer runs mixed.
            max_steps = int(rng.integers(16, 60))
            stuck = 0
            for _ in range(max_steps):
                # Candidate turns: mostly forward, occasional big turn (wrist flick).
                turn = float(rng.normal(0.0, 0.35))
                if rng.random() < 0.08:
                    turn += float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.7, 1.6))
                c, s = math.cos(turn), math.sin(turn)
                nvx, nvy = vx * c - vy * s, vx * s + vy * c
                # Try a few step lengths / slight side slips to stay in ink.
                stepped = False
                for slip in (0.0, 0.55, -0.55, 1.1, -1.1):
                    sx = -nvy * slip * step * 0.35
                    sy = nvx * slip * step * 0.35
                    step_len = step * float(rng.uniform(0.78, 1.18))
                    nx = x + nvx * step_len + sx
                    ny = y + nvy * step_len + sy
                    xi, yi = int(round(nx)), int(round(ny))
                    if 0 <= xi < w and 0 <= yi < h and component[yi, xi]:
                        x, y = float(xi), float(yi)
                        vx, vy = nvx, nvy
                        points.append((x, y))
                        ci, cj = cov_idx(x, y)
                        if covered[ci, cj] == 0:
                            covered[ci, cj] = 1
                            visited_cells += 1
                        else:
                            # Light re-trace is ok; mark revisit so we prefer fresh areas.
                            covered[ci, cj] = min(3, covered[ci, cj] + 1)
                        stepped = True
                        stuck = 0
                        break
                if not stepped:
                    # Bounce: reverse or pick new angle toward uncovered ink.
                    stuck += 1
                    if stuck >= 3:
                        break
                    ang = float(rng.uniform(0.0, math.tau))
                    vx, vy = math.cos(ang), math.sin(ang)
                    continue
                # Lift pen early sometimes (short 随笔).
                if len(points) >= 8 and rng.random() < 0.028:
                    break

            if len(points) < 2:
                path_i += 1
                continue
            # Soft freehand wobble on the polyline (keep endpoints near ink).
            if len(points) >= 3:
                wobbled: list[Point] = [points[0]]
                for i in range(1, len(points) - 1):
                    px, py = points[i]
                    # Local tangent normal.
                    ax, ay = points[i - 1]
                    bx, by = points[i + 1]
                    tx, ty = bx - ax, by - ay
                    tl = math.hypot(tx, ty) or 1.0
                    nx_, ny_ = -ty / tl, tx / tl
                    amp = float(rng.uniform(0.2, 0.9))
                    wobbled.append((px + nx_ * amp * float(rng.normal(0, 0.4)), py + ny_ * amp * float(rng.normal(0, 0.4))))
                wobbled.append(points[-1])
                points = wobbled

            # Width from local thickness along path; stay modest.
            widths: list[float] = []
            for px, py in points:
                xi, yi = int(round(px)), int(round(py))
                if 0 <= xi < w and 0 <= yi < h:
                    r = float(dist[yi, xi])
                else:
                    r = local_r
                widths.append(max(1.4, min(max_brush, r * 1.08 + 0.45)))
            # Calligraphic pulse on top.
            pulse = _freehand_widths(len(points), float(sum(widths) / len(widths)), rng, max_brush=max_brush)
            widths = [max(1.4, min(max_brush, 0.58 * a + 0.42 * b)) for a, b in zip(widths, pulse)]

            strokes.append(
                Stroke(
                    points=points,
                    color=color,
                    source=source,
                    width=float(sum(widths) / len(widths)),
                    widths=widths,
                )
            )
            path_i += 1

    return strokes


def _hatch_fill_strokes(
    mask: np.ndarray,
    min_radius: float = 4.0,
    min_area: int = 80,
    spacing_factor: float = 0.9,
    dist: np.ndarray | None = None,
    source: str = "hatch-fill",
    color: tuple[int, int, int] = (64, 60, 62),
    fill_all: bool = False,
    casual: bool = True,
) -> list[Stroke]:
    """Deprecated scan-fill; routes to freehand scribble (no parallel brush rows)."""

    if not np.any(mask):
        return []
    if dist is None:
        dist = _distance_transform(mask)
    if not fill_all:
        erode_r = max(1, int(round(min_radius)))
        thick_core = _erode_mask(mask, radius=erode_r)
        if not np.any(thick_core):
            return []
        mask = _dilate_mask(thick_core, radius=max(1, erode_r - 1)) & mask
        dist = _distance_transform(mask)
    return _freehand_scribble_strokes(
        mask,
        dist=dist,
        min_area=min_area,
        source=source if source != "hatch-fill" else "ink-freehand",
        color=color,
        max_brush=8.0 if fill_all else 9.0,
    )




def _stroke_length(points: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _point_distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _unit_vector(a: Point, b: Point) -> Point:
    length = _point_distance(a, b)
    if length <= 1e-6:
        return (0.0, 0.0)
    return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _is_frame_stroke(points: list[Point], bounds: tuple[int, int, int, int], margin_px: int) -> bool:
    if len(points) < 2:
        return False
    x0, y0, x1, y1 = bounds
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span_x, span_y = max_x - min_x, max_y - min_y
    edge_margin = margin_px * 2
    touches = [
        min_x <= x0 + edge_margin,
        max_x >= x1 - edge_margin,
        min_y <= y0 + edge_margin,
        max_y >= y1 - edge_margin,
    ]
    if sum(touches) >= 3 and span_x >= width * 0.68 and span_y >= height * 0.68:
        return True
    if span_x >= width * 0.72 and (max_y <= y0 + edge_margin or min_y >= y1 - edge_margin):
        return True
    if span_y >= height * 0.72 and (max_x <= x0 + edge_margin or min_x >= x1 - edge_margin):
        return True
    near_edge = (xs <= x0 + margin_px) | (xs >= x1 - margin_px) | (ys <= y0 + margin_px) | (ys >= y1 - margin_px)
    return bool(float(np.mean(near_edge)) >= 0.82 and (span_x >= width * 0.55 or span_y >= height * 0.55))


def filter_canvas_border_strokes(
    strokes: list[Stroke],
    canvas_size: tuple[int, int],
    bounds: tuple[int, int, int, int] | None = None,
    margin_px: int | None = None,
) -> list[Stroke]:
    """Drop strokes that are likely image/canvas frame artifacts."""

    margin = _border_margin_px(canvas_size, margin_px)
    width, height = canvas_size
    frame_bounds = [(0, 0, width, height)]
    if bounds is not None and bounds != frame_bounds[0]:
        frame_bounds.append(bounds)
    return [stroke for stroke in strokes if not any(_is_frame_stroke(stroke.points, candidate, margin) for candidate in frame_bounds)]


def _merge_candidate(
    a: list[Point],
    b: list[Point],
    max_gap: float,
    min_dot: float,
    touch_gap: float,
    touch_min_dot: float,
    short_len: float,
    short_min_dot: float,
) -> list[Point] | None:
    if len(a) < 2 or len(b) < 2:
        return None
    variants = [
        (a, b, _unit_vector(a[-2], a[-1]), _unit_vector(b[0], b[1])),
        (a, list(reversed(b)), _unit_vector(a[-2], a[-1]), _unit_vector(b[-1], b[-2])),
        (b, a, _unit_vector(b[-2], b[-1]), _unit_vector(a[0], a[1])),
        (list(reversed(b)), a, _unit_vector(b[0], b[1]), _unit_vector(a[0], a[1])),
    ]
    best: tuple[float, list[Point]] | None = None
    for left, right, left_dir, right_dir in variants:
        gap = _point_distance(left[-1], right[0])
        dot = _dot(left_dir, right_dir)
        effective_min_dot = min_dot
        if gap <= touch_gap:
            effective_min_dot = touch_min_dot
        elif min(_stroke_length(left), _stroke_length(right)) <= short_len and gap <= max_gap * 0.55:
            effective_min_dot = short_min_dot
        if gap <= max_gap and dot >= effective_min_dot:
            merged = left + right[1:] if gap <= 1.25 else left + right
            if best is None or gap < best[0]:
                best = (gap, merged)
    return best[1] if best else None


def merge_nearby_strokes(
    strokes: list[Stroke],
    canvas_size: tuple[int, int],
    max_gap_px: float | None = None,
    max_angle_deg: float = 42.0,
    touch_angle_deg: float = 96.0,
    short_angle_deg: float = 72.0,
) -> list[Stroke]:
    """Merge short broken strokes when nearby endpoints have compatible tangents."""

    if len(strokes) < 2:
        return strokes
    max_gap = max_gap_px if max_gap_px is not None else max(9.0, min(canvas_size) * 0.016)
    min_dot = math.cos(math.radians(max_angle_deg))
    touch_gap = max(1.75, min(canvas_size) * 0.0022)
    touch_min_dot = math.cos(math.radians(touch_angle_deg))
    short_len = max(14.0, min(canvas_size) * 0.018)
    short_min_dot = math.cos(math.radians(short_angle_deg))
    remaining = [
        Stroke(
            points=list(stroke.points),
            color=stroke.color,
            source=stroke.source,
            width=stroke.width,
            widths=list(stroke.widths) if stroke.widths else None,
        )
        for stroke in strokes
        if len(stroke.points) > 1
    ]
    # Do not merge fat fill strokes into thin linework — widths would average poorly.
    remaining.sort(key=lambda stroke: _stroke_length(stroke.points), reverse=True)
    changed = True
    passes = 0
    while changed and passes < 5:
        changed = False
        passes += 1
        idx = 0
        while idx < len(remaining):
            base = remaining[idx]
            if base.max_width(default=0.0) >= 8.0 or str(base.source).startswith(("hatch", "ink-brush", "ink-direct", "blob")):
                idx += 1
                continue
            best_index: int | None = None
            best_points: list[Point] | None = None
            best_gain = float("inf")
            best_width: float | None = base.width
            best_widths: list[float] | None = list(base.widths) if base.widths else None
            for candidate_index, candidate in enumerate(remaining):
                if candidate_index == idx:
                    continue
                if candidate.max_width(default=0.0) >= 8.0 or str(candidate.source).startswith(
                    ("hatch", "ink-brush", "ink-direct", "blob")
                ):
                    continue
                merged = _merge_candidate(base.points, candidate.points, max_gap, min_dot, touch_gap, touch_min_dot, short_len, short_min_dot)
                if merged is None:
                    continue
                gain = _stroke_length(merged) - _stroke_length(base.points) - _stroke_length(candidate.points)
                if gain < best_gain:
                    best_gain = gain
                    best_index = candidate_index
                    best_points = merged
                    scalar = [w for w in (base.width, candidate.width) if w is not None and w > 0]
                    best_width = max(scalar) if scalar else None
                    # Drop per-point widths on merge — geometry changed.
                    best_widths = None
            if best_index is not None and best_points is not None:
                base.points = best_points
                base.source = f"{base.source}+merged"
                base.width = best_width
                base.widths = best_widths
                del remaining[best_index]
                changed = True
                if best_index < idx:
                    idx -= 1
            idx += 1
    return remaining


def stitch_touching_strokes(
    strokes: list[Stroke],
    max_gap_px: float = 1.5,
    max_turn_deg: float = 96.0,
) -> list[Stroke]:
    """Join genuinely touching geometry paths without inventing long bridges.

    Raster skeletons split a continuous line at every junction. A spatial
    endpoint index keeps this pass close to linear even for detailed artwork,
    unlike the general-purpose all-pairs merger.
    """

    usable = [stroke for stroke in strokes if len(stroke.points) > 1]
    if len(usable) < 2:
        return strokes
    cell = max(1.0, float(max_gap_px))
    min_dot = math.cos(math.radians(max_turn_deg))
    endpoint_grid: dict[tuple[int, int], list[tuple[int, bool]]] = {}

    def grid_key(point: Point) -> tuple[int, int]:
        return (int(math.floor(point[0] / cell)), int(math.floor(point[1] / cell)))

    for index, stroke in enumerate(usable):
        endpoint_grid.setdefault(grid_key(stroke.points[0]), []).append((index, False))
        endpoint_grid.setdefault(grid_key(stroke.points[-1]), []).append((index, True))

    unused = set(range(len(usable)))

    def oriented(index: int, from_end: bool) -> tuple[list[Point], list[float] | None]:
        stroke = usable[index]
        points = list(reversed(stroke.points)) if from_end else list(stroke.points)
        widths = list(reversed(stroke.widths)) if from_end and stroke.widths else (
            list(stroke.widths) if stroke.widths else None
        )
        return points, widths

    def next_path(point: Point, direction: Point) -> tuple[int, list[Point], list[float] | None] | None:
        gx, gy = grid_key(point)
        best: tuple[float, int, list[Point], list[float] | None] | None = None
        for cy in range(gy - 1, gy + 2):
            for cx in range(gx - 1, gx + 2):
                for index, from_end in endpoint_grid.get((cx, cy), []):
                    if index not in unused:
                        continue
                    points, widths = oriented(index, from_end)
                    gap = _point_distance(point, points[0])
                    if gap > max_gap_px:
                        continue
                    candidate_direction = _unit_vector(points[0], points[1])
                    alignment = _dot(direction, candidate_direction)
                    if alignment < min_dot:
                        continue
                    score = gap + (1.0 - alignment) * max_gap_px * 0.35
                    if best is None or score < best[0]:
                        best = (score, index, points, widths)
        return None if best is None else (best[1], best[2], best[3])

    joined: list[Stroke] = []
    seed_order = sorted(unused, key=lambda index: _stroke_length(usable[index].points), reverse=True)
    for seed_index in seed_order:
        if seed_index not in unused:
            continue
        unused.remove(seed_index)
        seed = usable[seed_index]
        chain = list(seed.points)
        chain_widths = list(seed.widths) if seed.widths else None

        for side in range(2):
            while len(chain) > 1:
                direction = _unit_vector(chain[-2], chain[-1])
                candidate = next_path(chain[-1], direction)
                if candidate is None:
                    break
                index, points, widths = candidate
                unused.remove(index)
                skip_first = _point_distance(chain[-1], points[0]) <= 1.25
                chain.extend(points[1:] if skip_first else points)
                if chain_widths is not None and widths is not None:
                    chain_widths.extend(widths[1:] if skip_first else widths)
                else:
                    chain_widths = None
            if side == 0:
                chain.reverse()
                if chain_widths is not None:
                    chain_widths.reverse()

        width = seed.width
        if chain_widths:
            width = float(sum(chain_widths) / len(chain_widths))
        joined.append(
            Stroke(
                points=chain,
                color=seed.color,
                source=seed.source,
                width=width,
                widths=chain_widths,
            )
        )
    return joined


def _resample_points(
    points: list[Point],
    spacing: float = 3.0,
    widths: list[float] | None = None,
) -> tuple[list[Point], list[float] | None]:
    if len(points) < 2:
        return points, list(widths) if widths else None
    use_widths = widths if widths and len(widths) == len(points) else None
    total = _stroke_length(points)
    if total <= spacing:
        if use_widths is None:
            return [points[0], points[-1]], None
        return [points[0], points[-1]], [use_widths[0], use_widths[-1]]
    targets = np.arange(0.0, total, spacing).tolist()
    if not targets or targets[-1] < total:
        targets.append(total)
    result: list[Point] = []
    result_w: list[float] | None = [] if use_widths is not None else None
    segment_index = 0
    traversed = 0.0
    for target in targets:
        while segment_index < len(points) - 2:
            seg_len = _point_distance(points[segment_index], points[segment_index + 1])
            if traversed + seg_len >= target:
                break
            traversed += seg_len
            segment_index += 1
        a, b = points[segment_index], points[segment_index + 1]
        seg_len = max(_point_distance(a, b), 1e-6)
        local = max(0.0, min(1.0, (target - traversed) / seg_len))
        result.append((a[0] + (b[0] - a[0]) * local, a[1] + (b[1] - a[1]) * local))
        if result_w is not None and use_widths is not None:
            wa = use_widths[segment_index]
            wb = use_widths[min(segment_index + 1, len(use_widths) - 1)]
            result_w.append(wa + (wb - wa) * local)
    return result, result_w


def _chaikin_smooth(
    points: list[Point],
    iterations: int = 1,
    widths: list[float] | None = None,
) -> tuple[list[Point], list[float] | None]:
    if len(points) < 4:
        return points, list(widths) if widths else None
    result = points
    result_w = list(widths) if widths and len(widths) == len(points) else None
    for _ in range(iterations):
        smoothed: list[Point] = [result[0]]
        smoothed_w: list[float] | None = [result_w[0]] if result_w is not None else None
        for idx, (a, b) in enumerate(zip(result, result[1:])):
            smoothed.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            smoothed.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
            if smoothed_w is not None and result_w is not None:
                wa, wb = result_w[idx], result_w[idx + 1]
                smoothed_w.append(wa * 0.75 + wb * 0.25)
                smoothed_w.append(wa * 0.25 + wb * 0.75)
        smoothed.append(result[-1])
        if smoothed_w is not None and result_w is not None:
            smoothed_w.append(result_w[-1])
        result = smoothed
        result_w = smoothed_w
    return result, result_w


def smooth_strokes(strokes: list[Stroke], spacing: float = 3.0) -> list[Stroke]:
    """Resample and lightly smooth stroke coordinates to reduce pixel stair-steps."""

    smoothed: list[Stroke] = []
    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        # Freehand fills already carry intentional wobble — do not Chaikin them flat.
        src = str(stroke.source)
        if src.startswith(("hatch", "ink-brush", "ink-direct", "ink-freehand", "blob")):
            smoothed.append(stroke)
            continue
        points, widths = _resample_points(stroke.points, spacing=spacing, widths=stroke.widths)
        points, widths = _chaikin_smooth(points, iterations=1, widths=widths)
        points, widths = _resample_points(points, spacing=spacing, widths=widths)
        is_thick = bool(widths and max(widths) >= 8.0) or (stroke.width is not None and stroke.width >= 8.0)
        min_keep = 1.0 if is_thick else 2.0
        if len(points) > 1 and _stroke_length(points) > min_keep:
            smoothed.append(
                Stroke(
                    points=points,
                    color=stroke.color,
                    source=f"{stroke.source}+smooth",
                    width=stroke.width if widths is None else float(sum(widths) / len(widths)),
                    widths=widths,
                )
            )
    return smoothed


def postprocess_strokes(
    strokes: list[Stroke],
    canvas_size: tuple[int, int],
    smooth: bool = True,
    merge: bool = True,
    smooth_spacing: float | None = None,
    merge_gap_px: float | None = None,
    min_length_px: float = 0.0,
    max_merge_strokes: int | None = 5000,
    draw_mode: str = "structure-then-ink",
) -> list[Stroke]:
    """Apply continuity improvements before final stroke ordering."""

    processed = strokes
    if merge and (max_merge_strokes is None or len(processed) <= max_merge_strokes):
        processed = merge_nearby_strokes(processed, canvas_size, max_gap_px=merge_gap_px)
    if smooth:
        spacing = smooth_spacing if smooth_spacing is not None else max(2.25, min(canvas_size) * 0.0035)
        processed = smooth_strokes(processed, spacing=spacing)
    if min_length_px > 0:
        processed = [
            stroke
            for stroke in processed
            if _stroke_length(stroke.points) >= min_length_px
            or str(stroke.source).startswith(("hatch", "ink-brush", "ink-direct", "blob"))
            or stroke.max_width(default=0.0) >= 8.0
        ]
    return order_strokes(processed, canvas_size, draw_mode=draw_mode)


def _stroke_bounds(points: list[Point]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _stroke_center(points: list[Point]) -> Point:
    x0, y0, x1, y1 = _stroke_bounds(points)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _canvas_from_strokes(strokes: list[Stroke]) -> tuple[int, int]:
    xs = [p[0] for s in strokes for p in s.points]
    ys = [p[1] for s in strokes for p in s.points]
    return int(max(xs)) + 8, int(max(ys)) + 8


def _order_strokes_internal(strokes: list[Stroke], canvas_size: tuple[int, int] | None = None) -> list[Stroke]:
    remaining = [s for s in strokes if len(s.points) > 1 and _stroke_length(s.points) > 1.0]
    ordered: list[Stroke] = []
    current: Point | None = None
    last_center: Point | None = None
    width, height = canvas_size or _canvas_from_strokes(remaining)
    row_height = max(48.0, min(160.0, height * 0.12))

    def row_of(stroke: Stroke) -> int:
        _x0, y0, _x1, _y1 = _stroke_bounds(stroke.points)
        return int(y0 // row_height)

    while remaining:
        if current is None:
            index = min(
                range(len(remaining)),
                key=lambda i: (
                    row_of(remaining[i]),
                    _stroke_bounds(remaining[i].points)[0],
                    _stroke_bounds(remaining[i].points)[1],
                    -_stroke_length(remaining[i].points),
                ),
            )
        else:
            def score(idx: int) -> float:
                stroke = remaining[idx]
                start, end = stroke.points[0], stroke.points[-1]
                x0, y0, x1, y1 = _stroke_bounds(stroke.points)
                cx, cy = _stroke_center(stroke.points)
                dist = min(math.hypot(start[0] - current[0], start[1] - current[1]), math.hypot(end[0] - current[0], end[1] - current[1]))
                length = _stroke_length(stroke.points)
                visual = x0 * 0.55 + y0 * 0.18
                continuity = dist * 0.32
                long_line_bonus = min(length, 900.0) * 0.055
                backtrack_penalty = 0.0
                if last_center is not None:
                    if cy < last_center[1] - row_height * 0.75:
                        backtrack_penalty += row_height * 4.0
                    if abs(cy - last_center[1]) <= row_height and cx < last_center[0] - width * 0.18:
                        backtrack_penalty += width * 0.35
                return visual + continuity + backtrack_penalty - long_line_bonus

            endpoint_distances = [
                min(_point_distance(current, stroke.points[0]), _point_distance(current, stroke.points[-1]))
                for stroke in remaining
            ]
            local_limit = max(36.0, min(width, height) * 0.055)
            candidates = [idx for idx, distance in enumerate(endpoint_distances) if distance <= local_limit]
            if not candidates:
                min_row = min(row_of(stroke) for stroke in remaining)
                candidates = [idx for idx, stroke in enumerate(remaining) if row_of(stroke) == min_row]
            index = min(candidates, key=score)
        stroke = remaining.pop(index)
        if current is not None:
            start, end = stroke.points[0], stroke.points[-1]
            if math.hypot(end[0] - current[0], end[1] - current[1]) < math.hypot(start[0] - current[0], start[1] - current[1]):
                stroke.points = list(reversed(stroke.points))
                if stroke.widths:
                    stroke.widths = list(reversed(stroke.widths))
        ordered.append(stroke)
        current = stroke.points[-1]
        last_center = _stroke_center(stroke.points)
    return ordered


def _is_fill_stroke(stroke: Stroke) -> bool:
    """All freehand ink is fill; no skeleton outlines remain."""
    src = str(stroke.source)
    return (
        src.startswith("hatch")
        or src.startswith("ink-brush")
        or src.startswith("ink-direct")
        or src.startswith("ink-freehand")
        or src.startswith("blob")
        or "fill" in src
        or "wash" in src
        or stroke.max_width(default=0.0) >= 8.0
    )


def _stroke_bbox_center(stroke: Stroke) -> Point:
    return _stroke_center(stroke.points)


def _region_order_strokes(strokes: list[Stroke], canvas_size: tuple[int, int] | None) -> list[Stroke]:
    """Keep nearby paths together so a local form finishes before moving on."""

    if len(strokes) < 2:
        return strokes
    width, height = canvas_size or _canvas_from_strokes(strokes)
    gap = max(12.0, min(width, height) * 0.035)
    boxes = [_stroke_bounds(stroke.points) for stroke in strokes]
    parent = list(range(len(strokes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    # Bbox proximity is deliberately used instead of stroke width: hair,
    # sleeves, and wash fragments should remain one local drawing region.
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        for j in range(i):
            a0, b0, a1, b1 = boxes[j]
            horizontal_gap = max(0.0, max(a0, x0) - min(a1, x1))
            vertical_gap = max(0.0, max(b0, y0) - min(b1, y1))
            if horizontal_gap <= gap and vertical_gap <= gap:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(strokes)):
        groups.setdefault(find(index), []).append(index)
    ordered_groups = sorted(
        groups.values(),
        key=lambda indexes: (
            min(boxes[index][1] for index in indexes),
            min(boxes[index][0] for index in indexes),
        ),
    )
    return [strokes[index] for indexes in ordered_groups for index in indexes]


def order_strokes(
    strokes: list[Stroke],
    canvas_size: tuple[int, int] | None = None,
    draw_mode: str = "structure-then-ink",
) -> list[Stroke]:
    """Order freehand strokes top-to-bottom / left-to-right with continuity.

    Geometry strokes are ordered before wash/fill strokes. structure-then-ink
    paints solid structure before wash; direct-ink keeps one spatial pass.
    """
    mode = _normalize_draw_mode(draw_mode)
    if mode == "direct-ink":
        return _region_order_strokes(_order_strokes_internal(strokes, canvas_size), canvas_size)

    # Solid structure first, wash second.
    solids = [s for s in strokes if not str(s.source).endswith("wash") and "wash" not in str(s.source)]
    washes = [s for s in strokes if str(s.source).endswith("wash") or "wash" in str(s.source)]
    return _order_strokes_internal(solids, canvas_size) + _order_strokes_internal(washes, canvas_size)


def _detail_preset(stroke_detail: str, canvas_size: tuple[int, int]) -> dict[str, float | int | bool]:
    if stroke_detail not in STROKE_DETAIL_PRESETS:
        raise ValueError(f"Unknown stroke detail preset: {stroke_detail}")
    preset = dict(STROKE_DETAIL_PRESETS[stroke_detail])
    min_side = min(canvas_size)
    if preset["smooth_spacing"] == "rich":
        preset["smooth_spacing"] = max(1.65, min_side * 0.0026)
    elif preset["smooth_spacing"] == "max":
        preset["smooth_spacing"] = max(1.2, min_side * 0.0019)
    if preset["merge_gap"] == "rich":
        preset["merge_gap"] = max(7.0, min_side * 0.014)
    elif preset["merge_gap"] == "max":
        preset["merge_gap"] = max(4.0, min_side * 0.0075)
    if preset["min_length"] == "rich":
        preset["min_length"] = max(9.0, min_side * 0.0115)
    return preset


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance to background for each foreground pixel."""
    try:
        import cv2

        return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    except Exception:
        # Lightweight fallback without OpenCV: iterative morphological distance.
        dist = np.zeros(mask.shape, dtype=np.float32)
        current = mask.astype(bool)
        radius = 0
        while np.any(current) and radius < 64:
            radius += 1
            dist[current] = float(radius)
            # erode by 1
            padded = np.pad(current, 1, mode="constant", constant_values=False)
            eroded = np.ones_like(current, dtype=bool)
            for dy in range(3):
                for dx in range(3):
                    eroded &= padded[dy : dy + current.shape[0], dx : dx + current.shape[1]]
            current = eroded
        return dist


def _stroke_widths_from_distance(
    paths: list[np.ndarray],
    dist: np.ndarray,
    min_width: float = 2.0,
    max_width: float = 28.0,
) -> list[float]:
    """Estimate a pen width for each skeleton path from local ink thickness."""
    heights, width = dist.shape
    widths: list[float] = []
    for path in paths:
        samples: list[float] = []
        for x, y in path:
            xi, yi = int(x), int(y)
            if 0 <= xi < width and 0 <= yi < heights:
                # Diameter ≈ 2 * radius-to-edge.
                samples.append(float(dist[yi, xi]) * 2.0)
        if not samples:
            widths.append(min_width)
            continue
        # Median is robust to junction spikes; slight boost so solid ink fills.
        med = float(np.median(samples))
        est = max(min_width, min(max_width, med * 1.15 + 0.5))
        widths.append(est)
    return widths


def _ink_brush_fill_strokes(
    mask: np.ndarray,
    min_area: int = 80,
    spacing: float = 3.5,
    brush_width: float | None = None,
    color: tuple[int, int, int] = (64, 60, 62),
) -> list[Stroke]:
    """Dense brush-scan fills that paint solid ink regions fluidly.

    Unlike sparse 45° hatch lines, spacing is tight relative to brush width so
    consecutive passes blend into continuous ink instead of visible stripes.
    """
    h, w = mask.shape
    if not np.any(mask):
        return []
    labels, num_labels = _label_components(mask)
    strokes: list[Stroke] = []
    bw = float(brush_width) if brush_width and brush_width > 0 else max(3.0, spacing * 1.35)
    step = max(1.5, min(spacing, bw * 0.55))

    for label in range(1, num_labels + 1):
        coords = np.argwhere(labels == label)
        area = int(len(coords))
        if area < min_area:
            continue
        y0 = int(coords[:, 0].min())
        y1 = int(coords[:, 0].max())
        x0 = int(coords[:, 1].min())
        x1 = int(coords[:, 1].max())
        bw_box = x1 - x0 + 1
        bh_box = y1 - y0 + 1
        # Prefer horizontal brush for wide regions, vertical for tall ones.
        horizontal = bw_box >= bh_box
        reverse = False
        if horizontal:
            for y in np.arange(y0 + step * 0.5, y0 + bh_box, step):
                yi = int(round(y))
                if yi < 0 or yi >= h:
                    continue
                row = labels[yi, x0 : x0 + bw_box] == label
                if not np.any(row):
                    continue
                xs = np.flatnonzero(row) + x0
                # Split into contiguous runs.
                breaks = np.where(np.diff(xs) > 1)[0]
                starts = np.r_[0, breaks + 1]
                ends = np.r_[breaks, len(xs) - 1]
                for s_i, e_i in zip(starts, ends):
                    x_start, x_end = int(xs[s_i]), int(xs[e_i])
                    if x_end - x_start < 2:
                        continue
                    span = x_end - x_start
                    sample_x = np.linspace(x_start, x_end, max(3, int(span / 18) + 2))
                    phase = label * 0.73 + yi * 0.11
                    wobble = min(1.8, bw * 0.16)
                    candidate = [
                        (float(x), float(yi + math.sin(phase + index * 1.7) * wobble))
                        for index, x in enumerate(sample_x)
                    ]
                    pts = candidate if all(
                        0 <= int(round(px)) < w
                        and 0 <= int(round(py)) < h
                        and labels[int(round(py)), int(round(px))] == label
                        for px, py in candidate
                    ) else [(float(x_start), float(yi)), (float(x_end), float(yi))]
                    if reverse:
                        pts = list(reversed(pts))
                    strokes.append(
                        Stroke(
                            points=pts,
                            color=color,
                            source="ink-brush-fill",
                            width=bw,
                            widths=[bw * (0.88 + 0.16 * math.sin(phase + index)) for index in range(len(pts))],
                        )
                    )
                    reverse = not reverse
        else:
            for x in np.arange(x0 + step * 0.5, x0 + bw_box, step):
                xi = int(round(x))
                if xi < 0 or xi >= w:
                    continue
                col = labels[y0 : y0 + bh_box, xi] == label
                if not np.any(col):
                    continue
                ys = np.flatnonzero(col) + y0
                breaks = np.where(np.diff(ys) > 1)[0]
                starts = np.r_[0, breaks + 1]
                ends = np.r_[breaks, len(ys) - 1]
                for s_i, e_i in zip(starts, ends):
                    y_start, y_end = int(ys[s_i]), int(ys[e_i])
                    if y_end - y_start < 2:
                        continue
                    span = y_end - y_start
                    sample_y = np.linspace(y_start, y_end, max(3, int(span / 18) + 2))
                    phase = label * 0.61 + xi * 0.13
                    wobble = min(1.8, bw * 0.16)
                    candidate = [
                        (float(xi + math.sin(phase + index * 1.7) * wobble), float(y))
                        for index, y in enumerate(sample_y)
                    ]
                    pts = candidate if all(
                        0 <= int(round(px)) < w
                        and 0 <= int(round(py)) < h
                        and labels[int(round(py)), int(round(px))] == label
                        for px, py in candidate
                    ) else [(float(xi), float(y_start)), (float(xi), float(y_end))]
                    if reverse:
                        pts = list(reversed(pts))
                    strokes.append(
                        Stroke(
                            points=pts,
                            color=color,
                            source="ink-brush-fill",
                            width=bw,
                            widths=[bw * (0.88 + 0.16 * math.sin(phase + index)) for index in range(len(pts))],
                        )
                    )
                    reverse = not reverse
    return strokes


def _mask_to_geometry_strokes(
    mask: np.ndarray,
    dist: np.ndarray,
    *,
    min_points: int,
    source: str,
    color: tuple[int, int, int],
    min_length: float = 0.0,
    max_width: float = 28.0,
) -> list[Stroke]:
    """Trace source geometry without adding random hand-drawn drift."""

    if not np.any(mask):
        return []
    skeleton = zhang_suen_skeleton(mask)
    paths = trace_8connected(skeleton, min_points=min_points)
    strokes: list[Stroke] = []
    for path in paths:
        points = [(float(x), float(y)) for x, y in path]
        if len(points) < 2 or _stroke_length(points) < min_length:
            continue
        widths = _widths_for_path_from_dist(path, dist, min_width=1.5, max_width=max_width)
        strokes.append(
            Stroke(
                points=points,
                color=color,
                source=source,
                width=float(sum(widths) / max(1, len(widths))),
                widths=widths,
            )
        )
    return strokes


def _apply_source_tones(strokes: list[Stroke], gray: np.ndarray) -> list[Stroke]:
    """Use the source raster tone for each extracted path instead of one ink color."""

    height, width = gray.shape
    toned: list[Stroke] = []
    for stroke in strokes:
        samples: list[float] = []
        for x, y in stroke.points:
            xi = max(0, min(width - 1, int(round(x))))
            yi = max(0, min(height - 1, int(round(y))))
            samples.append(float(gray[yi, xi]))
        tone = int(round(float(np.median(samples)))) if samples else 80
        # Keep the slight warm-black bias used by the renderer while retaining
        # the measured light/dark difference between ink layers.
        tone = max(18, min(235, tone))
        stroke.color = (tone, max(0, tone - 2), max(0, tone - 4))
        toned.append(stroke)
    return toned


def _normalize_draw_mode(draw_mode: str | None) -> str:
    mode = (draw_mode or "structure-then-ink").strip().lower()
    if mode in {"direct", "direct-ink", "ink-direct", "one-pass", "fluid", "一气呵成"}:
        return "direct-ink"
    return "structure-then-ink"


def to_strokes(
    png_path: Path,
    canvas_size: tuple[int, int],
    threshold: int | None = None,
    stroke_detail: str = "max",
    disable_hatching: bool = False,
    draw_mode: str = "structure-then-ink",
    ink_darkness: int = 90,
    ink_brush: float = 5.5,
) -> list[Stroke]:
    """Convert raster line art into ordered freehand stroke paths.

    Structure layers follow the source geometry; wash layers use organic
    coverage strokes. Never uses parallel scan-brush rows for line structure.
    Both draw modes paint ink mass as continuous freehand pen strokes (随笔):

      - structure-then-ink: darker solid freehand first, then soft wash freehand
      - direct-ink: solid + wash freehand in one spatial pass (一气呵成)

    ink_darkness: 0–100 (0=white, 100=pure black, default 90)
    ink_brush: max stroke width in pixels (default 5.5)
    """

    import os

    mode = _normalize_draw_mode(draw_mode)
    pure_ink = mode == "direct-ink"

    darkness = max(0, min(100, int(ink_darkness)))
    v = int(round(255 * (1.0 - darkness / 100.0)))
    ink_color = (max(0, v), max(0, v - 2), max(0, v - 4))
    ink_solid_color = (max(0, v + 6), max(0, v + 4), max(0, v + 2))
    brush = max(1.5, float(ink_brush))

    # Ink-wash pipeline sets one of these; prefer explicit provider markers.
    is_ink = any(
        key in os.environ
        for key in (
            "INKWASH_BONE_DELTA",
            "INKWASH_DARK_THRESHOLD",
            "INKWASH_MID_DELTA",
            "INKWASH_PALE_DELTA",
            "WHITEBOARD_INKWASH_MODE",
        )
    )
    detail = _detail_preset(stroke_detail, canvas_size)
    if threshold is None:
        if is_ink:
            # Structure tiers from run_inkwash_cv are 0 (bone) and ~55 (mid).
            threshold = int(os.getenv("INKWASH_DRAW_THRESH", "90"))
        else:
            threshold = int(detail["threshold"])
    else:
        threshold = int(threshold)

    source, bounds = load_on_canvas_with_bounds(png_path, canvas_size)
    gray = np.asarray(source.convert("L"))
    mask = suppress_canvas_border(binarize(gray, threshold), bounds=bounds)
    if not np.any(mask):
        return []

    min_side = min(canvas_size)
    strokes: list[Stroke] = []
    dist = _distance_transform(mask)

    # Bridge tiny gaps only when ink already has body (helps broken freehand).
    interior = _erode_mask(mask, radius=1)
    if np.any(interior) and float(interior.sum()) >= max(8.0, float(mask.sum()) * 0.08):
        mask = _dilate_mask(mask, radius=1)
        dist = _distance_transform(mask)

    # --- Geometry-preserving structure + organic wash; no scan-brush rows ---
    # disable_hatching is ignored for generation (we never hatch); kept for API compat.
    _ = disable_hatching
    if is_ink:
        solid_thresh = int(os.getenv("INKWASH_SOLID_THRESH", "100"))
        solid_mask = suppress_canvas_border(binarize(gray, solid_thresh), bounds=bounds)
        solid_dist = _distance_transform(solid_mask) if np.any(solid_mask) else dist

        # Structure lines follow the extracted geometry. Only the wash layer
        # below uses organic freehand coverage.
        solid_src = "ink-direct" if pure_ink else "ink-freehand"
        strokes.extend(
            _mask_to_geometry_strokes(
                solid_mask,
                dist=solid_dist,
                min_points=int(detail["min_points"]),
                source=solid_src,
                color=ink_solid_color if pure_ink else ink_color,
                min_length=0.0 if stroke_detail == "max" else float(detail["min_length"]),
                max_width=brush + 0.3,
            )
        )

        # Broad ink bodies use irregular, boundary-constrained brush passes
        # instead of evenly spaced scan rows. The paths mimic push/pull ink
        # movement while remaining inside the extracted mask.
        strokes.extend(
            _freehand_scribble_strokes(
                solid_mask,
                dist=solid_dist,
                min_area=max(18, int(min_side * 0.004)),
                source="ink-freehand-fill",
                color=ink_color,
                max_brush=max(brush + 1.5, brush * 1.7),
            )
        )

        # Soft wash follows the extracted raster geometry too. Random
        # meanders here made the final linework visibly different from the
        # preview, especially on broad or pale ink strokes.
        wash_thresh = int(os.getenv("INKWASH_WASH_DRAW_THRESH", "220"))
        wash_mask = suppress_canvas_border(binarize(gray, wash_thresh), bounds=bounds) & ~solid_mask
        if np.any(wash_mask):
            wash_dist = _distance_transform(wash_mask)
            wash_src = "ink-direct-wash" if pure_ink else "ink-freehand-wash"
            strokes.extend(
                _mask_to_geometry_strokes(
                    wash_mask,
                    dist=wash_dist,
                    min_points=int(detail["min_points"]),
                    source=wash_src,
                    color=ink_color,
                    min_length=0.0 if stroke_detail == "max" else float(detail["min_length"]),
                    max_width=brush + 1.0,
                )
            )
    else:
        # For ordinary line art, source geometry is authoritative. Random
        # meanders make a clean extracted line look rough in the final video.
        strokes.extend(
            _mask_to_geometry_strokes(
                mask,
                dist=dist,
                min_points=int(detail["min_points"]),
                source="raster-geometry",
                color=ink_color,
                min_length=0.0 if stroke_detail == "max" else float(detail["min_length"]),
                max_width=brush,
            )
        )

        # Skeleton tracing splits continuous contours at junctions. Reconnect
        # only endpoints that physically touch and have a plausible pen turn.
        strokes = stitch_touching_strokes(strokes, max_gap_px=1.5)

    if is_ink:
        # Preserve the extracted浓/中/浅墨 tier on every traced path so the
        # renderer can reproduce density differences instead of flattening
        # all water-ink regions into one black stroke color.
        strokes = _apply_source_tones(strokes, gray)

    strokes = filter_canvas_border_strokes(strokes, canvas_size, bounds=bounds)

    # Freehand paths are already organic — never merge into long centerline zigzags.
    min_len = min(float(detail["min_length"]), 2.0)
    return postprocess_strokes(
        strokes,
        canvas_size,
        smooth=True,
        merge=False,
        smooth_spacing=float(detail["smooth_spacing"]) if detail["smooth_spacing"] is not None else None,
        merge_gap_px=float(detail["merge_gap"]) if detail["merge_gap"] is not None else None,
        min_length_px=min_len,
        max_merge_strokes=0,
        draw_mode=mode,
    )



def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value)
    return float(match.group(0)) if match else default


def _parse_points(points_text: str | None) -> list[Point]:
    if not points_text:
        return []
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", points_text)]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _sample_cubic(p0: Point, p1: Point, p2: Point, p3: Point, steps: int = 24) -> list[Point]:
    pts = []
    for idx in range(1, steps + 1):
        t, mt = idx / steps, 1.0 - idx / steps
        pts.append(
            (
                mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0],
                mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1],
            )
        )
    return pts


def _sample_quad(p0: Point, p1: Point, p2: Point, steps: int = 18) -> list[Point]:
    pts = []
    for idx in range(1, steps + 1):
        t, mt = idx / steps, 1.0 - idx / steps
        pts.append((mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0], mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]))
    return pts


def _parse_svg_path(d: str) -> list[list[Point]]:
    tokens = re.findall(r"[MmLlHhVvCcQqAaZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", d)
    strokes: list[list[Point]] = []
    idx, cmd = 0, ""
    current = start = (0.0, 0.0)
    stroke: list[Point] = []

    def is_cmd(token: str) -> bool:
        return bool(re.match(r"^[A-Za-z]$", token))

    def read_num() -> float:
        nonlocal idx
        value = float(tokens[idx])
        idx += 1
        return value

    while idx < len(tokens):
        if is_cmd(tokens[idx]):
            cmd = tokens[idx]
            idx += 1
        if not cmd:
            break
        absolute, code = cmd.isupper(), cmd.upper()
        if code == "M":
            x, y = read_num(), read_num()
            if not absolute:
                x, y = x + current[0], y + current[1]
            if len(stroke) > 1:
                strokes.append(stroke)
            current = start = (x, y)
            stroke = [current]
            cmd = "L" if absolute else "l"
        elif code == "L":
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                x, y = read_num(), read_num()
                current = (x, y) if absolute else (x + current[0], y + current[1])
                stroke.append(current)
        elif code == "H":
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                x = read_num()
                current = (x if absolute else x + current[0], current[1])
                stroke.append(current)
        elif code == "V":
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                y = read_num()
                current = (current[0], y if absolute else y + current[1])
                stroke.append(current)
        elif code == "C":
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                p1, p2, p3 = (read_num(), read_num()), (read_num(), read_num()), (read_num(), read_num())
                if not absolute:
                    p1 = (p1[0] + current[0], p1[1] + current[1])
                    p2 = (p2[0] + current[0], p2[1] + current[1])
                    p3 = (p3[0] + current[0], p3[1] + current[1])
                stroke.extend(_sample_cubic(current, p1, p2, p3))
                current = p3
        elif code == "Q":
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                p1, p2 = (read_num(), read_num()), (read_num(), read_num())
                if not absolute:
                    p1 = (p1[0] + current[0], p1[1] + current[1])
                    p2 = (p2[0] + current[0], p2[1] + current[1])
                stroke.extend(_sample_quad(current, p1, p2))
                current = p2
        elif code == "A":
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                _rx, _ry, _rot, _large, _sweep = read_num(), read_num(), read_num(), read_num(), read_num()
                x, y = read_num(), read_num()
                current = (x, y) if absolute else (x + current[0], y + current[1])
                stroke.append(current)
        elif code == "Z":
            if stroke and stroke[-1] != start:
                stroke.append(start)
            if len(stroke) > 1:
                strokes.append(stroke)
            stroke, current, cmd = [], start, ""
        else:
            while idx < len(tokens) and not is_cmd(tokens[idx]):
                idx += 1
    if len(stroke) > 1:
        strokes.append(stroke)
    return strokes


def _sample_ellipse(cx: float, cy: float, rx: float, ry: float, steps: int = 80) -> list[Point]:
    return [(cx + math.cos(2 * math.pi * i / steps) * rx, cy + math.sin(2 * math.pi * i / steps) * ry) for i in range(steps + 1)]


def svg_to_strokes(svg_path: Path, canvas_size: tuple[int, int]) -> tuple[list[Stroke], Image.Image]:
    """Parse SVG primitives into stroke paths and a preview image."""

    width, height = canvas_size
    root = ET.parse(svg_path).getroot()
    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if view_box:
        nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", view_box)]
        vb_x, vb_y, vb_w, vb_h = nums[:4]
    else:
        vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, _parse_float(root.attrib.get("width"), 100.0), _parse_float(root.attrib.get("height"), 100.0)
    scale = min(width / max(vb_w, 1.0), height / max(vb_h, 1.0))
    off_x, off_y = (width - vb_w * scale) / 2.0, (height - vb_h * scale) / 2.0
    content_bounds = (
        int(round(off_x)),
        int(round(off_y)),
        int(round(off_x + vb_w * scale)),
        int(round(off_y + vb_h * scale)),
    )

    def map_pt(point: Point) -> Point:
        return ((point[0] - vb_x) * scale + off_x, (point[1] - vb_y) * scale + off_y)

    strokes: list[Stroke] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        raw: list[list[Point]] = []
        if tag == "path" and elem.attrib.get("d"):
            raw.extend(_parse_svg_path(elem.attrib["d"]))
        elif tag in {"polyline", "polygon"}:
            pts = _parse_points(elem.attrib.get("points"))
            raw.append(pts + ([pts[0]] if tag == "polygon" and pts else []))
        elif tag == "line":
            raw.append([(_parse_float(elem.attrib.get("x1")), _parse_float(elem.attrib.get("y1"))), (_parse_float(elem.attrib.get("x2")), _parse_float(elem.attrib.get("y2")))])
        elif tag == "rect":
            x, y = _parse_float(elem.attrib.get("x")), _parse_float(elem.attrib.get("y"))
            w, h = _parse_float(elem.attrib.get("width")), _parse_float(elem.attrib.get("height"))
            raw.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
        elif tag == "circle":
            cx, cy, r = _parse_float(elem.attrib.get("cx")), _parse_float(elem.attrib.get("cy")), _parse_float(elem.attrib.get("r"))
            raw.append(_sample_ellipse(cx, cy, r, r))
        elif tag == "ellipse":
            raw.append(_sample_ellipse(_parse_float(elem.attrib.get("cx")), _parse_float(elem.attrib.get("cy")), _parse_float(elem.attrib.get("rx")), _parse_float(elem.attrib.get("ry"))))
        for pts in raw:
            mapped = [map_pt(pt) for pt in pts]
            if len(mapped) > 1:
                strokes.append(Stroke(mapped, source="svg"))
    strokes = postprocess_strokes(filter_canvas_border_strokes(strokes, canvas_size, bounds=content_bounds), canvas_size, smooth=False, merge=True)
    preview = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(preview)
    for stroke in strokes:
        draw.line(stroke.points, fill=stroke.color, width=2, joint="curve")
    return strokes, preview
