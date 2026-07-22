#!/usr/bin/env python3
"""Ink-wash (水墨风) line-art extractor — pure PIL + NumPy, zero neural network.

Designed for Chinese ink painting characteristics that generic edge models destroy:

  * 晕染 (yūn rǎn)  — soft graduated washes with no hard boundary
  * 飞白 (fēi bái)  — intentional gaps / dry-brush gaps inside a stroke
  * 浓/中/淡墨      — three ink density tiers that carry semantic weight

Pipeline
--------
1. Load grayscale, auto-orient to dark-on-white, optional long-edge resize.
2. Estimate paper tone (multi-percentile + border blend; works on beige/gray).
3. Normalize onto a white canvas: ``gray / local_paper * 255`` + soft floor.
4. Relative ink amount: ``ink = max(0, paper - gray)`` on normalized gray.
5. Structure layer (骨法用笔): dark ink OR (mid ink + edge response).
6. Wash layer (晕染): softer ink with weak edges, excluding structure.
7. Brush-aware despeckle (keep elongated components).
8. Output grayscale tiers compatible with whiteboard skeleton tracing:
        浓墨 structure  → 0
        中墨 structure  → 55
        淡墨 wash       → 200   (above default DRAW/SOLID gates → no hatch mess)
        background      → 255

Environment variables (UI names take precedence when set)
---------------------------------------------------------
INKWASH_BONE_DELTA / INKWASH_DARK_THRESHOLD   浓墨相对纸白阈值
INKWASH_MID_DELTA  / INKWASH_MID_THRESHOLD    中墨/细线相对纸白
INKWASH_PALE_DELTA / INKWASH_PALE_THRESHOLD   淡墨/晕染相对纸白（越大越干净）
INKWASH_GRAD_THRESH                           结构边缘梯度门限
INKWASH_MIN_AREA                              最小连通域面积
INKWASH_MIN_ELON                              小连通域最小细长度
INKWASH_CLOSE_RADIUS                          结构线闭运算半径
INKWASH_SIZE                                  长边处理尺寸 (0=原图)
INKWASH_WASH_LEVEL                            输出淡墨灰阶 (默认 200)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _first_env_int(*names: str, default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip() != "":
            return int(raw)
    return default


def _first_env_float(*names: str, default: float) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip() != "":
            return float(raw)
    return default


# Defaults (env re-read inside extract_inkwash so UI/subprocess always apply).
_DEFAULTS = {
    "bone": 70,
    "mid": 32,
    "pale": 22,
    "grad": 6.0,
    "min_area": 6,
    "min_elon": 1.25,
    "close_r": 1,
    "max_size": 0,
    "wash_level": 200,
}
MID_LEVEL = 55


def _read_params() -> dict:
    return {
        "bone": _first_env_int("INKWASH_BONE_DELTA", "INKWASH_DARK_THRESHOLD", default=_DEFAULTS["bone"]),
        "mid": _first_env_int("INKWASH_MID_DELTA", "INKWASH_MID_THRESHOLD", default=_DEFAULTS["mid"]),
        "pale": _first_env_int("INKWASH_PALE_DELTA", "INKWASH_PALE_THRESHOLD", default=_DEFAULTS["pale"]),
        "grad": _first_env_float("INKWASH_GRAD_THRESH", default=_DEFAULTS["grad"]),
        "min_area": _first_env_int("INKWASH_MIN_AREA", default=_DEFAULTS["min_area"]),
        "min_elon": _first_env_float("INKWASH_MIN_ELON", default=_DEFAULTS["min_elon"]),
        "close_r": _first_env_int("INKWASH_CLOSE_RADIUS", default=_DEFAULTS["close_r"]),
        "max_size": _first_env_int("INKWASH_SIZE", default=_DEFAULTS["max_size"]),
        "wash_level": _first_env_int("INKWASH_WASH_LEVEL", default=_DEFAULTS["wash_level"]),
    }


def load_gray(path: Path, max_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (H×W uint8 grayscale, original_size) — always dark-on-white."""
    img = Image.open(path).convert("L")
    original_size = img.size
    if max_size > 0:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    # White-on-dark ink scans: invert so ink is dark.
    if float(np.mean(arr)) < 100.0:
        arr = (255 - arr).astype(np.uint8)
    return arr, original_size


def _box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Fast separable box blur returning float32."""
    if radius <= 0:
        return arr.astype(np.float32)
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    blurred = Image.fromarray(clipped, mode="L").filter(ImageFilter.BoxBlur(radius))
    return np.asarray(blurred, dtype=np.float32)


def _border_paper_sample(gray: np.ndarray) -> float | None:
    """Estimate paper from border/corner strips (common paper on scans).

    Returns None when borders look like heavy ink/frame rather than paper.
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 16 or w < 16:
        return None
    bw = max(2, min(h, w) // 20)
    strips = [
        g[:bw, :],
        g[-bw:, :],
        g[:, :bw],
        g[:, -bw:],
    ]
    border = np.concatenate([s.ravel() for s in strips])
    # Reject if border is mostly dark (ink-heavy frame / photo edge).
    med = float(np.median(border))
    if med < 70.0:
        return None
    # Bright side of border: ignore darkest 30% (possible frame/ink).
    floor = float(np.percentile(border, 30))
    bright = border[border >= floor]
    if bright.size < 32:
        bright = border
    # Mean of mid-high band is robust to occasional bright speculars.
    lo = float(np.percentile(bright, 60))
    hi = float(np.percentile(bright, 92))
    band = bright[(bright >= lo) & (bright <= hi)]
    if band.size < 16:
        band = bright
    return float(np.mean(band))


def estimate_paper_white(gray: np.ndarray) -> float:
    """Robust paper tone from the bright side of the histogram.

    Works for pure white *and* beige/gray/cream paper. Does not force a
    floor of 180 (that broke mid-gray paper). Uses multi-percentile global
    estimate blended with border/corner samples when they look like paper.
    """
    g = gray.astype(np.float32)
    # Ignore darkest ~40% (likely ink) when estimating paper.
    p40 = float(np.percentile(g, 40))
    bright = g[g >= p40]
    if bright.size < 64:
        bright = g

    # Multi-percentile robust estimate: mean of [p70, p90] on bright side,
    # with a p50-of-bright fallback blend for textured paper.
    p70 = float(np.percentile(bright, 70))
    p90 = float(np.percentile(bright, 90))
    band = bright[(bright >= p70) & (bright <= p90)]
    if band.size < 16:
        global_paper = float(np.percentile(bright, 85))
    else:
        global_paper = float(np.mean(band))
    # Soft blend with median of bright pixels (stable on textured scans).
    bright_med = float(np.median(bright))
    global_paper = 0.65 * global_paper + 0.35 * bright_med

    border_paper = _border_paper_sample(g)
    if border_paper is not None:
        # Prefer border when it is close to global bright estimate (same paper).
        # If border is much darker, it is likely a mat/frame — trust global.
        if border_paper >= global_paper * 0.85:
            paper = 0.55 * global_paper + 0.45 * border_paper
        else:
            paper = global_paper
    else:
        paper = global_paper

    # Clip only to a wide sensible range — mid-gray paper (~120-160) is valid.
    return float(np.clip(paper, 80.0, 255.0))


def normalize_to_white_paper(gray: np.ndarray) -> np.ndarray:
    """Map non-white paper onto a ~255 canvas while keeping true ink dark.

    Pipeline:
      1. Global paper estimate.
      2. Local paper field via large-radius blur of ink-suppressed gray
         (tracks uneven lighting without letting strokes darken the field).
      3. Divide-and-scale: out = gray / local_paper * 255.
      4. Soft paper-noise gate: near-white pixels forced to 255.
    """
    g = gray.astype(np.float32)
    paper = estimate_paper_white(g)

    # Local paper field: large blur tracks gradual shading / textured paper.
    # Radius scales with image size; clamp so tiny images still get smoothing.
    h, w = g.shape
    radius = max(16, min(h, w) // 12)

    # Two-pass ink-suppressed blur so thick strokes don't pull local paper down,
    # while still following real paper gradients (beige left → bright right).
    rough = _box_blur(g, radius)
    # Pixels much darker than local neighbourhood are ink → replace with rough paper.
    cleaned = np.where(g < (rough - 12.0), rough, g)
    local_blur = _box_blur(cleaned, radius)
    # Mild floor vs global paper: keep mid-gray / shaded paper valid, but avoid
    # near-zero division. Do NOT force paper*0.9 — that flattens lighting gradients.
    local_paper = np.maximum(local_blur, max(paper * 0.55, 1.0))

    # Normalize: paper → ~255, ink stays proportionally dark.
    out = g / local_paper * 255.0
    out = np.clip(out, 0.0, 255.0)

    # Preserve absolute darkness: pixels much darker than *local* paper stay dark.
    # Scale by local paper so gray<=60 maps near 0 on mid-gray and white paper.
    relative_local = g / local_paper
    very_dark = relative_local < 0.30
    dark_mapped = np.clip(relative_local * 255.0, 0.0, 255.0)
    out = np.where(very_dark, np.minimum(out, dark_mapped), out)
    out = np.clip(out, 0.0, 255.0)

    # Soft paper-noise gate: residual texture near paper becomes pure white.
    # After normalize, true paper sits near 255; texture often lands 240-254.
    paper_floor = 248.0
    out = np.where(out >= paper_floor, 255.0, out)
    # Gentle near-white lift: soft floor so faint texture doesn't become wash.
    soft = (out >= 240.0) & (out < paper_floor)
    # Blend toward 255 based on how close to floor.
    t = (out - 240.0) / (paper_floor - 240.0)
    out = np.where(soft, out + t * (255.0 - out) * 0.85, out)
    out = np.clip(out, 0.0, 255.0)

    return out.astype(np.uint8)


def relative_ink(gray: np.ndarray, paper: float) -> np.ndarray:
    """Ink amount relative to paper white, in [0, 255]."""
    ink = paper - gray.astype(np.float32)
    return np.clip(ink, 0.0, 255.0)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Simple Sobel gradient magnitude (float32)."""
    g = gray.astype(np.float32)
    # Pad to keep size.
    padded = np.pad(g, 1, mode="edge")
    # Kernels
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def local_contrast_ink(gray: np.ndarray, radius: int = 24) -> np.ndarray:
    """Local darkness vs neighbourhood — catches faint strokes on uneven paper."""
    g = gray.astype(np.float32)
    r = max(4, int(radius))
    local_mean = _box_blur(g, r)
    local_sq = _box_blur(g * g, r)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0)) + 1e-3
    # Positive when pixel is darker than local mean.
    z = (local_mean - g) / local_std
    # Map typical stroke z-scores into ~[0, 255] ink units.
    return np.clip(z * 28.0, 0.0, 255.0)


def adaptive_ink_scale(ink: np.ndarray, pale_delta: float) -> float:
    """Estimate a per-image scale for pale, medium, and dark ink tiers."""

    values = ink[ink >= max(3.0, float(pale_delta) * 0.35)]
    if values.size < 32:
        return 1.0
    p50 = float(np.percentile(values, 50))
    p90 = float(np.percentile(values, 90))
    # p90 tracks the strongest usable mark while p50 prevents one dark seal
    # from dominating the whole image. 110 matches the existing defaults.
    reference = max(p90, p50 * 1.45)
    return float(np.clip(reference / 110.0, 0.55, 1.75))


def structure_and_wash_masks(
    gray: np.ndarray,
    bone_delta: float,
    mid_delta: float,
    pale_delta: float,
    grad_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bone_mask, mid_mask, wash_mask) as bool arrays.

    Priority (high → low):
      1. Absolute / relative 浓墨 (true black) → always bone
      2. Mid ink with pen-edge → mid structure
      3. Soft mid-tone mass → wash (never steal pure black)
    """
    paper = estimate_paper_white(gray)
    ink = relative_ink(gray, paper)
    g = gray.astype(np.float32)

    tone_scale = adaptive_ink_scale(ink, pale_delta)
    # Do not lower the dark-ink gate for an entirely pale image: otherwise a
    # faint gray stroke gets promoted to 浓墨 instead of remaining 中/淡墨.
    bone_delta = max(float(bone_delta), float(bone_delta) * tone_scale)
    mid_delta = float(mid_delta) * tone_scale
    pale_delta = float(pale_delta) * tone_scale

    # Local contrast catches faint strokes on uneven paper.
    local = local_contrast_ink(gray, radius=max(12, min(gray.shape) // 40))
    ink_line = np.maximum(ink, local * 0.55)

    grad = sobel_magnitude(gray)
    grad = _box_blur(grad, 1)
    grad_soft = _box_blur(grad, 2)
    g_thr = float(grad_thresh)
    edge = grad >= g_thr
    soft_interior = grad_soft < max(g_thr * 1.15, 4.5)

    # --- Absolute black ink (must never become wash) ---
    # gray near 0, or very high relative ink amount.
    abs_black = (g <= 48.0) | (ink >= max(140.0, float(bone_delta) * 1.8))
    # --- Bone: dark enough relative to paper ---
    dark = ink >= float(bone_delta)
    bone = abs_black | dark
    # Soft *mid-tone* interiors can leave bone, but pure black stays.
    demote = soft_interior & ~edge & ~abs_black & (ink < float(bone_delta) * 1.35)
    bone = bone & ~demote

    # --- Mid (中墨细线) ---
    mid = (ink_line >= float(mid_delta)) & edge & ~bone
    mid |= (ink_line >= float(mid_delta) * 1.35) & (grad_soft >= g_thr * 0.65) & ~bone & ~abs_black

    structure = bone | mid

    # --- Wash: soft gray masses only (never abs_black) ---
    wash_seed = (ink >= float(pale_delta)) & ~structure & ~abs_black
    wash = wash_seed & (soft_interior | ((ink >= float(pale_delta) * 1.1) & (ink < float(bone_delta))))
    wash_soft = _box_blur(wash.astype(np.float32) * 255.0, 3) >= 70.0
    wash = wash & wash_soft
    # Soft mid-gray fills (clothing wash etc.), still not pure black.
    soft_gray = (
        (ink >= max(float(pale_delta), float(mid_delta) * 0.75))
        & (ink < float(bone_delta) * 1.2)
        & soft_interior
        & ~edge
        & ~abs_black
    )
    wash |= soft_gray & ~mid

    # Final priority: black > mid > wash
    bone = bone | abs_black
    mid = mid & ~bone
    wash = wash & ~bone & ~mid & ~abs_black

    return bone, mid, wash


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """BFS connected-component labelling. Returns (label_map, num_labels)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    num = 0
    ys, xs = np.nonzero(mask)
    visited = np.zeros_like(mask, dtype=bool)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        num += 1
        stack = [(sx, sy)]
        visited[sy, sx] = True
        labels[sy, sx] = num
        while stack:
            x, y = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        labels[ny, nx] = num
                        stack.append((nx, ny))
    return labels, num


def clean_mask(mask: np.ndarray, min_area: int, min_elon: float, keep_large: bool = True) -> np.ndarray:
    """Remove speckles; keep elongated strokes and optionally large blobs."""
    if not np.any(mask):
        return mask
    labels, num = _label_components(mask)
    keep = np.zeros_like(mask, dtype=bool)
    for lbl in range(1, num + 1):
        coords = np.argwhere(labels == lbl)
        area = len(coords)
        if area == 0:
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]
        h_span = int(ys.max() - ys.min()) + 1
        w_span = int(xs.max() - xs.min()) + 1
        elongation = max(h_span, w_span) / max(1, min(h_span, w_span))
        if area >= max(min_area * 4, 40) and keep_large:
            keep[labels == lbl] = True
        elif area >= min_area and elongation >= min_elon:
            keep[labels == lbl] = True
        elif area >= min_area * 3:
            # Compact but reasonably large ink dots / seals.
            keep[labels == lbl] = True
    return keep


def morphological_close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Closing = dilate then erode, restoring broken hair-line strokes."""
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MaxFilter(size))
    img = img.filter(ImageFilter.MinFilter(size))
    return np.asarray(img) > 127


def morphological_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.filter(ImageFilter.MinFilter(size))
    return np.asarray(img) > 127


def hollow_large_solids(
    bone: np.ndarray,
    mid: np.ndarray,
    wash: np.ndarray,
    gray: np.ndarray | None = None,
    min_solid_area: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outline-only for *mid-gray* fat blobs — never hollow pure black ink.

    True 浓墨 (near-black) stays solid black so the preview still shows it.
    Only soft mid-tone masses (clothing wash, soft fills) get ring+wash.
    """
    # Only consider mid structure for hollowing; bone (浓墨) is sacred.
    candidates = mid & ~bone
    if not np.any(candidates):
        return bone, mid, wash
    labels, num = _label_components(candidates)
    bone_out = bone.copy()
    mid_out = mid.copy()
    wash_out = wash.copy()
    g = None if gray is None else gray.astype(np.float32)
    for lbl in range(1, num + 1):
        comp = labels == lbl
        area = int(comp.sum())
        if area < min_solid_area:
            continue
        if g is not None:
            # Skip if this blob is actually dark ink (mean too black).
            if float(g[comp].mean()) < 90.0:
                continue
        eroded = morphological_erode(comp, 2)
        if int(eroded.sum()) < max(80, area // 5):
            continue  # already stroke-like
        ring = comp & ~morphological_erode(comp, 1)
        interior = comp & ~ring
        mid_out[comp] = False
        mid_out[ring] = True
        wash_out[interior] = True
    return bone_out, mid_out, wash_out


def suppress_border(arr: np.ndarray) -> np.ndarray:
    """Clear frame-hugging ink using the engine helper when available."""
    try:
        root = Path(__file__).resolve().parents[2]
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from whiteboard_skill.preprocess import suppress_canvas_border

        if arr.dtype == np.uint8:
            ink = arr < 250
            cleaned = suppress_canvas_border(ink)
            out = arr.copy()
            out[ink & ~cleaned] = 255
            return out
        cleaned = suppress_canvas_border(arr.astype(bool))
        return cleaned
    except Exception:
        return arr


def compose_output(
    bone: np.ndarray,
    mid: np.ndarray,
    wash: np.ndarray,
    wash_level: int,
    source_gray: np.ndarray | None = None,
) -> np.ndarray:
    """Compose three ink-density tiers for downstream tracing and preview."""
    out = np.full(bone.shape, 255, dtype=np.uint8)
    out[wash] = np.uint8(max(120, min(235, wash_level)))
    out[mid] = MID_LEVEL
    out[bone] = 0
    return out


def extract_inkwash(input_path: Path, output_path: Path) -> None:
    p = _read_params()
    gray, original_size = load_gray(input_path, p["max_size"])

    # Component thresholds are pixel-based, so scale them with the working
    # canvas. This keeps tiny details from disappearing on small images while
    # preventing paper dust from surviving on large scans.
    min_side = max(1, min(gray.shape))
    pixel_scale = min_side / 512.0
    area_scale = pixel_scale * pixel_scale
    min_area = max(2, int(round(float(p["min_area"]) * area_scale)))
    close_radius = max(0, int(round(float(p["close_r"]) * pixel_scale)))
    solid_area = max(120, int(round(min_side * min_side * 0.0016)))

    # Normalize non-white / uneven paper onto a white canvas before masks.
    # Pure white paper is effectively a no-op (paper≈255 → identity scale).
    gray = normalize_to_white_paper(gray)

    bone, mid, wash = structure_and_wash_masks(
        gray,
        bone_delta=p["bone"],
        mid_delta=p["mid"],
        pale_delta=p["pale"],
        grad_thresh=p["grad"],
    )

    # Preserve absolute black before despeckle (clean_mask can drop compact seals).
    # After normalize, true black ink still lands near 0 so <= 48 works.
    abs_black = gray <= 48
    bone = bone | abs_black

    bone = clean_mask(bone, min_area=max(2, min_area // 3), min_elon=min(p["min_elon"], 1.1), keep_large=True)
    # Force pure black back even if despeckle removed tiny dots.
    bone = bone | abs_black
    mid = clean_mask(mid, min_area=min_area, min_elon=p["min_elon"], keep_large=False)
    mid = mid & ~bone
    # Wash: keep large soft areas, drop dust; elongation less important.
    wash = clean_mask(wash, min_area=max(min_area * 3, 8), min_elon=1.0, keep_large=True)
    wash = wash & ~bone & ~abs_black

    # Only hollow mid-gray fat blobs — never 浓墨.
    bone, mid, wash = hollow_large_solids(
        bone, mid, wash, gray=gray, min_solid_area=solid_area
    )
    bone = bone | abs_black
    mid = mid & ~bone
    wash = wash & ~bone & ~abs_black

    # Reconnect broken structure hairlines only (not wash).
    structure = morphological_close(bone | mid, close_radius)
    bone = (structure & bone) | abs_black
    mid = structure & ~bone
    wash = wash & ~(bone | mid)

    final = compose_output(bone, mid, wash, p["wash_level"])
    final = suppress_border(final)
    # Last guard: any remaining near-black source pixel must stay black in output.
    final[abs_black] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = Image.fromarray(final, mode="L")
    if p["max_size"] > 0 and result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)
    result.convert("RGB").save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="水墨风线稿提取器 — 纯 PIL+NumPy，无需神经网络权重")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract_inkwash(args.input.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
