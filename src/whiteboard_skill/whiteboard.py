"""Stroke-level whiteboard renderer."""

from __future__ import annotations

import math
import bisect
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .preprocess import Stroke, _distance_transform, _freehand_scribble_strokes, load_on_canvas, postprocess_strokes, svg_to_strokes, to_strokes, trace_8connected, zhang_suen_skeleton
from .ui_state import DEFAULT_DRAWING_TOOL


BUILTIN_HANDS = ("asian", "black", "children", "white", "brush", "rooster-quill")
HAND_ANCHORS: dict[str, tuple[float, float]] = {
    "asian": (0.04, 0.28),
    "black": (0.05, 0.30),
    "children": (0.16, 0.15),
    "white": (0.04, 0.35),
    # Generated brush asset has the tip near the upper-left corner.
    "brush": (0.034, 0.185),
    # Generated rooster feather asset: nib sits near the lower-left tip.
    "rooster-quill": (0.12, 0.93),
}
SKETCH_INK_COLOR = (24, 22, 24)
SKETCH_INK_OPACITY = 1.0


@dataclass(frozen=True)
class HandCursor:
    """Prepared hand cursor image and its pen-tip anchor."""

    image: Image.Image
    anchor: tuple[float, float]


@dataclass
class TimelineStroke:
    """Stroke geometry mapped onto the render timeline."""

    stroke: Stroke
    cumulative: list[float]
    length: float
    start_unit: float
    end_unit: float
    pause_end_unit: float


@dataclass(frozen=True)
class ContourFillCache:
    """Precomputed field used by contour-aware color fill."""

    resistance: np.ndarray
    rows: np.ndarray
    wave: np.ndarray
    priority: np.ndarray


FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
)
YUXI_ASSET_PATH = Path(__file__).resolve().parents[2] / "assets" / "stamps" / "vintage-yuxi.png"
_YUXI_ASSET_CACHE: Image.Image | None = None
CURSOR_ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "cursors"
FONT_ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "fonts"
THEME_FONT_PATHS = {
    "mao": FONT_ASSET_ROOT / "fangzheng-dacao" / "MaoTi.ttf",
    "cursive": FONT_ASSET_ROOT / "liu-jian-mao-cao" / "LiuJianMaoCao-Regular.ttf",
}
THEME_AA_SCALE = 3
REAL_ERASER_ASSET_PATH = CURSOR_ASSET_ROOT / "real-eraser.png"
ANIMATED_CURSOR_DIRS = {
    "wukong-run": CURSOR_ASSET_ROOT / "wukong-run",
    "zhubajie-run": CURSOR_ASSET_ROOT / "zhubajie-action-v2",
    "tangsanzang-run": CURSOR_ASSET_ROOT / "tangsanzang-run",
    "guanyu-run": CURSOR_ASSET_ROOT / "guanyu-run",
    "zhugeliang-run": CURSOR_ASSET_ROOT / "zhugeliang-run",
}
ANIMATED_ENTRANCE_CURSOR_DIRS = {
    "zhubajie-run": CURSOR_ASSET_ROOT / "zhubajie-walk-entry-v2",
    "tangsanzang-run": CURSOR_ASSET_ROOT / "tangsanzang-walk-entry-v2",
    "zhugeliang-run": CURSOR_ASSET_ROOT / "zhugeliang-walk-entry-v2",
    "wukong-run": CURSOR_ASSET_ROOT / "wukong-cloud-entry",
    "guanyu-run": CURSOR_ASSET_ROOT / "guanyu-horse-entry",
}
_ANIMATED_CURSOR_FRAMES_CACHE: dict[tuple[str, str], tuple[Image.Image, ...]] = {}
_REAL_ERASER_SOURCE_CACHE: Image.Image | None = None
_REAL_ERASER_ROTATION_CACHE: dict[tuple[int, int], Image.Image] = {}


def ease_in_out_sine(t: float) -> float:
    """Return sine ease-in-out progress.

    Example:
        >>> round(ease_in_out_sine(0.5), 2)
        0.5
    """

    return -(math.cos(math.pi * max(0.0, min(1.0, t))) - 1) / 2


def _load_yuxi_asset() -> Image.Image | None:
    global _YUXI_ASSET_CACHE
    if _YUXI_ASSET_CACHE is None and YUXI_ASSET_PATH.exists():
        _YUXI_ASSET_CACHE = Image.open(YUXI_ASSET_PATH).convert("RGBA")
    return _YUXI_ASSET_CACHE.copy() if _YUXI_ASSET_CACHE is not None else None


def _load_animated_cursor_frames(
    style: str,
    stage: str = "creation",
) -> tuple[Image.Image, ...]:
    cache_key = (style, "entrance" if stage == "entrance" else "creation")
    if cache_key not in _ANIMATED_CURSOR_FRAMES_CACHE:
        frame_dir = (
            ANIMATED_ENTRANCE_CURSOR_DIRS.get(style)
            if stage == "entrance"
            else ANIMATED_CURSOR_DIRS.get(style)
        )
        frames = [] if frame_dir is None else [
            Image.open(path).convert("RGBA")
            for path in sorted(frame_dir.glob("frame-*.png"))
        ]
        _ANIMATED_CURSOR_FRAMES_CACHE[cache_key] = tuple(frames)
    return _ANIMATED_CURSOR_FRAMES_CACHE[cache_key]


def _animated_character_side(resolution: tuple[int, int], stage: str = "creation") -> int:
    """Return a readable character footprint for the current action stage."""

    min_side = min(resolution)
    if stage == "entrance":
        return max(132, min(380, int(round(min_side * 0.34))))
    if stage == "finish":
        return max(88, min(280, int(round(min_side * 0.23))))
    return max(108, min(300, int(round(min_side * 0.28))))


def _animated_entrance_frame_count(draw_frames: int, fps: int) -> int:
    """Reserve a readable entrance without consuming most of the drawing stage."""

    return min(
        max(12, int(round(max(1, fps) * 1.8))),
        max(0, int(round(draw_frames * 0.45))),
        max(0, draw_frames - 1),
    )


def _animated_locomotion_frame(distance: float, min_side: int, frame_count: int) -> int:
    """Select a leg pose from traveled distance so stationary actors stop running."""

    if frame_count <= 0:
        return 0
    pose_stride = max(8.0, float(min_side) * 0.035)
    return int(max(0.0, distance) / pose_stride) % frame_count


def _animated_sprite_position(
    canvas_size: tuple[int, int],
    sprite_size: tuple[int, int],
    left: float,
    top: float,
) -> tuple[int, int]:
    """Keep the complete character visible, including its head and tool."""

    max_left = max(0, canvas_size[0] - sprite_size[0])
    max_top = max(0, canvas_size[1] - sprite_size[1])
    return (
        max(0, min(max_left, int(round(left)))),
        max(0, min(max_top, int(round(top)))),
    )


def _paste_animated_runner(
    frame: Image.Image,
    x: float,
    y: float,
    angle: float,
    motion_distance: float,
    style: str,
    stage: str = "creation",
    size_override: int | None = None,
) -> Image.Image:
    """Place an upright animated character at the active stroke point."""

    frames = _load_animated_cursor_frames(style, stage=stage)
    if not frames:
        return frame
    pose_index = _animated_locomotion_frame(motion_distance, min(frame.size), len(frames))
    sprite = frames[pose_index].copy()
    if math.cos(angle) < 0:
        sprite = ImageOps.mirror(sprite)
    side = size_override or _animated_character_side(frame.size, stage=stage)
    if stage == "entrance":
        side = int(round(side * {
            "wukong-run": 1.25,
            "guanyu-run": 1.45,
        }.get(style, 1.05)))
    sprite.thumbnail((side, side), Image.Resampling.LANCZOS)
    bob_ratio = 0.038 if stage == "entrance" else 0.026
    bob = int(round(math.sin(pose_index * math.pi / 2.0) * max(1, side * bob_ratio)))
    left, top = _animated_sprite_position(
        frame.size,
        sprite.size,
        x - sprite.width * 0.5,
        y - sprite.height * 0.82 + bob,
    )
    composed = frame.convert("RGBA")
    composed.alpha_composite(sprite, (left, top))
    return composed.convert("RGB")


def _load_real_eraser_source() -> Image.Image:
    global _REAL_ERASER_SOURCE_CACHE
    if _REAL_ERASER_SOURCE_CACHE is None:
        _REAL_ERASER_SOURCE_CACHE = Image.open(REAL_ERASER_ASSET_PATH).convert("RGBA")
    return _REAL_ERASER_SOURCE_CACHE


def _paste_real_eraser(frame: Image.Image, x: float, y: float, angle: float) -> Image.Image:
    """Paste a physical eraser with its worn tip anchored to the active path."""

    length = max(72, min(180, int(round(min(frame.size) * 0.18))))
    direction_count = 32
    direction = int(round((angle % (2 * math.pi)) / (2 * math.pi) * direction_count)) % direction_count
    cache_key = (length, direction)
    sprite = _REAL_ERASER_ROTATION_CACHE.get(cache_key)
    if sprite is None:
        source = _load_real_eraser_source()
        height = max(1, int(round(source.height * length / source.width)))
        source = source.resize((length, height), Image.Resampling.LANCZOS)
        anchor_x = source.width * 0.96
        anchor_y = source.height * 0.52
        radius = int(math.ceil(max(
            math.hypot(px - anchor_x, py - anchor_y)
            for px, py in ((0, 0), (source.width, 0), (0, source.height), source.size)
        ))) + 3
        side = radius * 2 + 1
        center = radius
        anchored = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        anchored.alpha_composite(
            source,
            (int(round(center - anchor_x)), int(round(center - anchor_y))),
        )
        degrees = -direction * (360.0 / direction_count)
        sprite = anchored.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False)
        _REAL_ERASER_ROTATION_CACHE[cache_key] = sprite
    center = sprite.width // 2
    result = frame.convert("RGBA")
    result.alpha_composite(sprite, (int(round(x)) - center, int(round(y)) - center))
    return result.convert("RGB")


def segment_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return segment angle in radians."""

    return math.atan2(b[1] - a[1], b[0] - a[0])


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    return ffmpeg


def _resize_cover(image: Image.Image, resolution: tuple[int, int]) -> Image.Image:
    width, height = resolution
    src = image.convert("RGB")
    scale = max(width / src.width, height / src.height)
    resized = src.resize((max(1, int(round(src.width * scale))), max(1, int(round(src.height * scale)))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _resize_contain(image: Image.Image, resolution: tuple[int, int], background: Image.Image | None = None) -> Image.Image:
    width, height = resolution
    src = image.convert("RGB")
    src.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = background.copy() if background else Image.new("RGB", resolution, "white")
    canvas.paste(src, ((width - src.width) // 2, (height - src.height) // 2))
    return canvas


def _load_source_image_canvas(image_path: Path, resolution: tuple[int, int], fit: str = "blur-fill") -> Image.Image:
    """Load the real color source for the final fill stage."""

    source = Image.open(image_path).convert("RGB")
    if fit == "exact":
        return source.resize(resolution, Image.Resampling.LANCZOS)
    if fit == "cover":
        return _resize_cover(source, resolution)
    if fit == "contain":
        return _resize_contain(source, resolution)
    background = _resize_cover(source, resolution).filter(ImageFilter.GaussianBlur(radius=max(8, min(resolution) // 26)))
    background = Image.blend(background, Image.new("RGB", resolution, "white"), 0.12)
    return _resize_contain(source, resolution, background=background)


def _line_art_canvas(image_path: Path, resolution: tuple[int, int]) -> Image.Image:
    """Load source line art as a same-size white canvas."""

    return load_on_canvas(image_path, resolution)


def _paper_background(resolution: tuple[int, int]) -> Image.Image:
    """Return a nearly white paper surface with restrained fiber variation."""

    width, height = resolution
    yy, xx = np.mgrid[0:height, 0:width]
    grain = (
        0.8 * np.sin(xx / 17.0 + yy / 29.0)
        + 0.35 * np.sin(xx / 5.7 - yy / 11.0)
    )
    base = np.empty((height, width, 3), dtype=np.float32)
    base[:] = (250.0, 249.0, 247.0)
    base += grain[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


DEFAULT_LINE_ART_SNAP_THRESHOLD = 170


def _dilate_bool_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0 or mask.size == 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    span = radius * 2 + 1
    for dy in range(span):
        for dx in range(span):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def _line_art_ink_mask(line_art: Image.Image, size: tuple[int, int], threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD) -> Image.Image:
    """Solid ink body mask for snap/reveal — never a skeleton.

    Older code thinned thick ink to Zhang-Suen + halo so snap looked like a
    centerline bone in the video. Always keep the full ink mass instead.
    """

    source = line_art.convert("RGB")
    if source.size != size:
        source = source.resize(size, Image.Resampling.LANCZOS)
    gray = np.asarray(source.convert("L"), dtype=np.uint8)
    raw_mask = gray < threshold
    if not np.any(raw_mask):
        return Image.new("L", size, 0)
    return Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L")


def _line_art_binary_mask(line_art: Image.Image, threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD) -> np.ndarray:
    gray = np.asarray(line_art.convert("L"), dtype=np.uint8)
    return gray < threshold


def _line_art_tone_overlay(
    canvas: Image.Image,
    line_art: Image.Image,
    alpha: float = 1.0,
    structure_threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD,
    wash_threshold: int = 235,
    replace_background: bool = False,
) -> Image.Image:
    """Blend source ink tones back without reducing them to black/white."""

    base = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    source = line_art.convert("L")
    if source.size != canvas.size:
        source = source.resize(canvas.size, Image.Resampling.LANCZOS)
    gray = np.asarray(source, dtype=np.float32)
    strength = max(0.0, min(1.0, float(alpha)))
    structure = gray < float(structure_threshold)
    wash = (gray < float(wash_threshold)) & ~structure
    # During reveal, the source line art is authoritative for the whole local
    # region. Starting from white removes rough temporary strokes underneath.
    # Erase temporary strokes back to the same paper surface. Falling back to
    # pure white here creates visible rectangular patches around the ink.
    if replace_background:
        is_plain_white = bool(np.mean(base >= 254.0) > 0.95)
        result = (
            np.full_like(base, 255.0)
            if is_plain_white
            else np.asarray(_paper_background(canvas.size), dtype=np.float32)
        )
    else:
        result = base.copy()

    # A very soft halo makes the ink touch the paper instead of reading as a
    # perfectly cut-out foreground layer.
    ink_strength = np.clip((255.0 - gray) / 255.0, 0.0, 1.0)
    if np.any(structure):
        bleed = np.asarray(
            Image.fromarray((ink_strength * 255).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=0.8)),
            dtype=np.float32,
        ) / 255.0
        bleed_alpha = bleed * 0.045 * strength
        result = result * (1.0 - bleed_alpha[..., None]) + np.asarray((92, 86, 80), dtype=np.float32) * bleed_alpha[..., None]

    if np.any(structure):
        # The extracted raster is authoritative. Copy its actual gray values
        # instead of painting a new stroke over the same geometry; this keeps
        # ordinary line art identical to the preview before color begins.
        source_ink = np.repeat(gray[..., None], 3, axis=2)
        exact_alpha = structure.astype(np.float32) * strength
        result = result * (1.0 - exact_alpha[..., None]) + source_ink * exact_alpha[..., None]

    if np.any(wash):
        # A light gray wash remains visibly lighter than structure ink.
        wash_alpha = np.clip(
            (float(wash_threshold) - gray) / max(1.0, wash_threshold - structure_threshold),
            0.08,
            0.42,
        ) * strength
        wash_alpha *= wash
        source_wash = np.repeat(gray[..., None], 3, axis=2)
        result = result * (1.0 - wash_alpha[..., None]) + source_wash * wash_alpha[..., None]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def _estimate_line_art_width(line_art: Image.Image | None, threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD) -> int:
    """Estimate the visible line width of a raster line-art image."""

    if line_art is None:
        return 2
    mask = _line_art_binary_mask(line_art, threshold=threshold)
    if not np.any(mask):
        return 2
    skel = zhang_suen_skeleton(mask)
    skeleton_pixels = int(np.count_nonzero(skel))
    if skeleton_pixels <= 0:
        return 2
    estimated = float(np.count_nonzero(mask)) / skeleton_pixels
    return max(2, min(7, int(round(estimated))))


def _resolve_line_thickness(requested: int | None, estimated_line_width: int) -> int:
    """Resolve an explicit stroke width or adapt it to the source line art."""

    if requested is None or int(requested) <= 0:
        return max(1, min(7, int(round(estimated_line_width))))
    return max(1, int(requested))


def _render_aa_scale(resolution: tuple[int, int], has_exact_line_art: bool) -> int:
    """Use supersampling only when strokes are the final visual output."""

    if has_exact_line_art:
        return 1
    return 2 if max(resolution) <= 1920 else 1


def _combine_masks(a: Image.Image, b: Image.Image) -> Image.Image:
    arr = np.minimum(np.asarray(a.convert("L"), dtype=np.uint8), np.asarray(b.convert("L"), dtype=np.uint8))
    return Image.fromarray(arr, mode="L")


def _exact_line_art_layer(canvas: Image.Image, line_art: Image.Image) -> Image.Image:
    """Place the extracted raster unchanged on the renderer's paper surface."""

    base = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
    source = line_art.convert("RGB")
    if source.size != canvas.size:
        source = source.resize(canvas.size, Image.Resampling.LANCZOS)
    source_arr = np.asarray(source, dtype=np.uint8)
    gray = np.asarray(source.convert("L"), dtype=np.uint8)
    plain_white = bool(np.mean(base >= 254) > 0.95)
    background = (
        np.full_like(base, 255)
        if plain_white
        else np.asarray(_paper_background(canvas.size), dtype=np.uint8).copy()
    )
    # Include anti-aliased edge pixels as well as solid lines. Their RGB value
    # comes directly from the extraction preview, with no recoloring or width
    # synthesis by the video renderer.
    ink = gray < 250
    background[ink] = source_arr[ink]
    return Image.fromarray(background, mode="RGB")


def _complete_line_art_canvas(
    canvas: Image.Image,
    line_art: Image.Image | None,
    threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD,
    alpha: float = 1.0,
    ink_mask: Image.Image | None = None,
    preserve_tones: bool = False,
    exact_layer: Image.Image | None = None,
) -> Image.Image:
    """Composite missing black line-art pixels over a redrawn stroke canvas."""

    if line_art is None and ink_mask is None and exact_layer is None:
        return canvas.copy()
    base = canvas.convert("RGB")
    if preserve_tones and (line_art is not None or exact_layer is not None):
        exact = exact_layer if exact_layer is not None else _exact_line_art_layer(base, line_art)
        strength = max(0.0, min(1.0, float(alpha)))
        return exact if strength >= 1.0 else Image.blend(base, exact, strength)
    mask = ink_mask if ink_mask is not None else _line_art_ink_mask(line_art, base.size, threshold=threshold)
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)
    opacity = SKETCH_INK_OPACITY * max(0.0, min(1.0, alpha))
    mask = mask.point(lambda px: int(px * opacity))
    ink = Image.new("RGB", base.size, SKETCH_INK_COLOR)
    base.paste(ink, mask=mask)
    return base


def _reveal_line_art_canvas(
    canvas: Image.Image,
    line_art: Image.Image | None,
    reveal_mask: Image.Image | None,
    threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD,
    ink_mask: Image.Image | None = None,
    preserve_tones: bool = False,
    exact_layer: Image.Image | None = None,
) -> Image.Image:
    if reveal_mask is None or (line_art is None and ink_mask is None and exact_layer is None):
        return canvas.copy()
    base = canvas.convert("RGB")
    if preserve_tones and (line_art is not None or exact_layer is not None):
        exact = exact_layer if exact_layer is not None else _exact_line_art_layer(base, line_art)
        return Image.composite(exact, base, reveal_mask)
    mask = ink_mask if ink_mask is not None else _line_art_ink_mask(line_art, base.size, threshold=threshold)
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)
    reveal = _combine_masks(mask, reveal_mask).point(lambda px: int(px * SKETCH_INK_OPACITY))
    ink = Image.new("RGB", base.size, SKETCH_INK_COLOR)
    base.paste(ink, mask=reveal)
    return base


def _load_text_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=max(10, size))


def _load_seal_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Prefer a traditional serif face for carved seal lettering."""

    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/STSongti-SC-Bold.ttc",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return _load_text_font(size)


def _load_theme_text_font(style: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = THEME_FONT_PATHS.get(style)
    if path is not None and path.exists():
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return _load_text_font(size)


def _fit_text_font(
    text: str,
    resolution: tuple[int, int],
    font_style: str | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    width, height = resolution
    max_width = width * 0.82
    max_height = height * 0.12
    start = int(max(20, min(height * 0.085, width * 0.16)))
    probe = Image.new("L", resolution, 255)
    draw = ImageDraw.Draw(probe)
    for size in range(start, 13, -2):
        font = _load_theme_text_font(font_style, size) if font_style else _load_text_font(size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        if right - left <= max_width and bottom - top <= max_height:
            return font
    return _load_theme_text_font(font_style, 14) if font_style else _load_text_font(14)


def _fit_vertical_text_font(
    text: str,
    resolution: tuple[int, int],
    font_style: str,
    font_size: int | None = None,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    width, height = resolution
    probe = Image.new("L", resolution, 255)
    draw = ImageDraw.Draw(probe)
    if font_size is None:
        start = int(max(22, min(height * 0.11, width * 0.17)))
    else:
        normalized_size = max(24, min(220, int(font_size)))
        start = int(round(normalized_size * min(resolution) / 398.0))
    lines = text.splitlines() or [text]
    for size in range(start, 15, -2):
        font = _load_theme_text_font(font_style, size)
        columns = [[draw.textbbox((0, 0), char, font=font) for char in line] for line in lines]
        column_widths = [max((right - left for left, top, right, bottom in boxes), default=0) for boxes in columns]
        column_heights = [
            sum(bottom - top for left, top, right, bottom in boxes)
            + max(0, len(boxes) - 1) * max(2, int(round(size * 0.08)))
            for boxes in columns
        ]
        column_gap = max(4, int(round(size * 0.18)))
        total_width = sum(column_widths) + max(0, len(column_widths) - 1) * column_gap
        if max(column_widths, default=0) <= width * 0.30 and total_width <= width * 0.72 and max(column_heights, default=0) <= height * 0.62:
            return font
    return _load_theme_text_font(font_style, 16)


def _fit_horizontal_theme_text_font(
    text: str,
    resolution: tuple[int, int],
    font_style: str,
    font_size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    width, height = resolution
    probe = Image.new("L", resolution, 0)
    draw = ImageDraw.Draw(probe)
    normalized_size = max(24, min(220, int(font_size)))
    start = int(round(normalized_size * min(resolution) / 398.0))
    for size in range(start, 15, -2):
        font = _load_theme_text_font(font_style, size)
        spacing = max(4, int(round(size * 0.18)))
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
        max_height = height * (0.20 if "\n" not in text else 0.48)
        if right - left <= width * 0.82 and bottom - top <= max_height:
            return font
    return _load_theme_text_font(font_style, 16)


def _theme_preview_image(
    text: str | None,
    font_style: str,
    font_size: int = 72,
    size: tuple[int, int] = (320, 104),
) -> Image.Image:
    """Render the actual bundled theme font for the frontend preview."""

    cleaned_lines = [" ".join(line.split()) for line in str(text or "").splitlines()]
    cleaned = "\n".join(line for line in cleaned_lines if line) or "嫦娥"
    image = Image.new("RGB", size, (250, 250, 249))
    draw = ImageDraw.Draw(image)
    font = _load_theme_text_font(font_style, 20)
    preview_size = max(18, min(82, int(round(max(24, min(220, font_size)) * 0.72))))
    for candidate_size in range(preview_size, 17, -2):
        candidate = _load_theme_text_font(font_style, candidate_size)
        spacing = max(3, int(round(candidate_size * 0.16)))
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), cleaned, font=candidate, spacing=spacing, align="center"
        )
        if right - left <= size[0] * 0.88 and bottom - top <= size[1] * 0.72:
            font = candidate
            break
    spacing = max(3, int(round(getattr(font, "size", 20) * 0.16)))
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), cleaned, font=font, spacing=spacing, align="center"
    )
    x = (size[0] - (right - left)) / 2 - left
    y = (size[1] - (bottom - top)) / 2 - top
    draw.multiline_text((x, y), cleaned, font=font, fill=(20, 18, 18), spacing=spacing, align="center")
    return image


def _theme_text_mask(
    text: str,
    resolution: tuple[int, int],
    font_style: str,
    font_size: int,
    position: str = "right",
    scale: int = 1,
) -> Image.Image:
    """Render the exact vertical glyphs used by the theme-writing stage."""

    cleaned_lines = [" ".join(line.split()) for line in str(text or "").splitlines()]
    cleaned = "\n".join(line for line in cleaned_lines if line)
    scaled_resolution = (resolution[0] * scale, resolution[1] * scale)
    mask = Image.new("L", scaled_resolution, 0)
    if not cleaned:
        return mask
    draw = ImageDraw.Draw(mask)
    width, height = scaled_resolution
    margin = max(18 * scale, int(round(height * 0.055)))
    if position in {"left", "right"}:
        font = _fit_vertical_text_font(cleaned, scaled_resolution, font_style, font_size=font_size)
        gap = max(2 * scale, int(round(getattr(font, "size", 20) * 0.08)))
        column_gap = max(4 * scale, int(round(getattr(font, "size", 20) * 0.18)))
        columns = []
        for line in cleaned.splitlines():
            boxes = [draw.textbbox((0, 0), char, font=font) for char in line]
            column_width = max((right - left for left, top, right, bottom in boxes), default=0)
            columns.append((line, boxes, column_width))
        cursor_x = margin if position == "left" else width - margin
        for line, boxes, column_width in columns:
            column_x = cursor_x if position == "left" else cursor_x - column_width
            cursor_y = margin
            for char, (left, top, right, bottom) in zip(line, boxes):
                char_w, char_h = right - left, bottom - top
                x = column_x + (column_width - char_w) / 2 - left
                draw.text((x, cursor_y - top), char, font=font, fill=255)
                cursor_y += char_h + gap
            cursor_x += column_width + column_gap if position == "left" else -(column_width + column_gap)
    else:
        font = _fit_horizontal_theme_text_font(cleaned, scaled_resolution, font_style, font_size)
        spacing = max(4 * scale, int(round(getattr(font, "size", 20) * 0.18)))
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), cleaned, font=font, spacing=spacing, align="center"
        )
        text_w, text_h = right - left, bottom - top
        x = (width - text_w) / 2 - left
        if position == "top":
            y = margin - top
        elif position == "center":
            y = (height - text_h) / 2 - top
        else:
            y = height - margin - text_h - top
        draw.multiline_text((x, y), cleaned, font=font, fill=255, spacing=spacing, align="center")
    return mask


def _text_to_strokes(
    text: str,
    resolution: tuple[int, int],
    position: str = "bottom",
    font_style: str | None = None,
    font_size: int | None = None,
) -> list[Stroke]:
    """Convert a short caption into drawable centerline strokes."""

    cleaned_lines = [" ".join(line.split()) for line in str(text or "").splitlines()]
    cleaned = "\n".join(line for line in cleaned_lines if line)
    if not cleaned:
        return []
    width, height = resolution
    vertical = position in {"left", "right"} and bool(font_style)
    if vertical:
        font = _fit_vertical_text_font(cleaned, resolution, str(font_style), font_size=font_size)
    elif font_style:
        font = _fit_horizontal_theme_text_font(cleaned, resolution, str(font_style), font_size or 72)
    else:
        font = _fit_text_font(cleaned, resolution, font_style=font_style)
    image = Image.new("L", resolution, 255)
    draw = ImageDraw.Draw(image)
    margin = max(18, int(round(height * 0.055)))
    if font_style:
        exact_mask = _theme_text_mask(
            cleaned,
            resolution,
            str(font_style),
            font_size or 72,
            position=position,
        )
        image = Image.eval(exact_mask, lambda value: 255 - value)
    else:
        left, top, right, bottom = draw.textbbox((0, 0), cleaned, font=font)
        text_w, text_h = right - left, bottom - top
        x = (width - text_w) / 2.0 - left
        if position == "top":
            y = margin - top
        elif position == "center":
            y = (height - text_h) / 2.0 - top
        else:
            y = height - margin - text_h - top
        draw.text((x, y), cleaned, font=font, fill=0)

    mask = np.asarray(image, dtype=np.uint8) < 235
    if not np.any(mask):
        return []
    skel = zhang_suen_skeleton(mask)
    paths = trace_8connected(skel, min_points=4)
    loaded_size = float(getattr(font, "size", max(20, min(resolution) * 0.1)))
    brush_width = max(2.0, loaded_size * (0.09 if font_style == "mao" else 0.065))
    strokes = [
        Stroke(
            points=[(float(x), float(y)) for x, y in path],
            source=f"theme-{font_style}" if font_style else "text",
            width=brush_width if font_style else None,
        )
        for path in paths
    ]
    return postprocess_strokes(
        strokes,
        resolution,
        smooth=True,
        merge=False if font_style else True,
    )


def _theme_duration_seconds(text: str | None) -> float:
    """Return enough time for a short handwritten title before stamping."""

    cleaned = "" if text is None else "".join(str(text).split())
    if not cleaned:
        return 0.0
    return min(4.0, 1.2 + 0.45 * len(cleaned))


def _draw_cursor(frame: Image.Image, x: float, y: float, angle: float, scale: float = 0.85, style: str = "procedural") -> None:
    draw = ImageDraw.Draw(frame, "RGBA")
    ux, uy = math.cos(angle), math.sin(angle)
    nx, ny = -uy, ux

    def pt(back: float, side: float) -> tuple[float, float]:
        return (x - ux * back + nx * side, y - uy * back + ny * side)

    if style == "ip-signature":
        s = scale
        draw.line([pt(104 * s, 0), pt(10 * s, 0)], fill=(25, 39, 92, 255), width=max(3, int(6 * s)))
        draw.line([pt(96 * s, -3 * s), pt(16 * s, -3 * s)], fill=(75, 126, 255, 220), width=max(1, int(2 * s)))
        draw.polygon([(x, y), pt(-15 * s, -5 * s), pt(-15 * s, 5 * s)], fill=(16, 24, 55, 255))
        draw.ellipse((pt(66 * s, -6 * s)[0] - 4 * s, pt(66 * s, -6 * s)[1] - 4 * s, pt(66 * s, -6 * s)[0] + 4 * s, pt(66 * s, -6 * s)[1] + 4 * s), outline=(42, 211, 238, 240), width=max(1, int(2 * s)))
        return

    if style == "ip-stamp":
        s = scale
        draw.line([pt(92 * s, 0), pt(16 * s, 0)], fill=(122, 33, 49, 255), width=max(5, int(10 * s)))
        draw.polygon([(x, y), pt(-18 * s, -13 * s), pt(-18 * s, 13 * s)], fill=(155, 38, 54, 245))
        draw.ellipse((x - 11 * s, y - 11 * s, x + 11 * s, y + 11 * s), outline=(220, 65, 74, 240), width=max(2, int(3 * s)))
        return

    if style == "ip-spark":
        s = scale
        draw.line([pt(112 * s, 0), pt(12 * s, 0)], fill=(245, 173, 26, 255), width=max(5, int(12 * s)))
        draw.line([pt(103 * s, -4 * s), pt(18 * s, -4 * s)], fill=(255, 239, 99, 240), width=max(2, int(4 * s)))
        draw.polygon([(x, y), pt(-18 * s, -7 * s), pt(-18 * s, 7 * s)], fill=(255, 205, 31, 255))
        for radius in (15, 23):
            draw.line([(x - radius * s, y), (x + radius * s, y)], fill=(250, 204, 21, 190), width=max(1, int(2 * s)))
            draw.line([(x, y - radius * s), (x, y + radius * s)], fill=(250, 204, 21, 190), width=max(1, int(2 * s)))
        return

    if style == "ip-eraser":
        s = scale
        eraser_body = [pt(86 * s, -10 * s), pt(12 * s, -10 * s), pt(12 * s, 10 * s), pt(86 * s, 10 * s)]
        draw.polygon(eraser_body, fill=(225, 225, 228, 255))
        draw.line(eraser_body + [eraser_body[0]], fill=(105, 105, 112, 255), width=max(1, int(2 * s)), joint="curve")
        draw.line([pt(72 * s, -9 * s), pt(72 * s, 9 * s)], fill=(201, 58, 72, 230), width=max(2, int(4 * s)))
        draw.polygon([(x, y), pt(-16 * s, -7 * s), pt(-16 * s, 7 * s)], fill=(125, 125, 132, 255))
        return

    if style == "brush":
        s = scale
        # Tool-only cursor: lacquered handle, ferrule, and tapered bristles.
        draw.line([pt(112 * s, 0), pt(18 * s, 0)], fill=(86, 45, 25, 255), width=max(3, int(8 * s)))
        draw.line([pt(108 * s, -2 * s), pt(25 * s, -2 * s)], fill=(190, 105, 50, 220), width=max(1, int(2 * s)))
        draw.polygon(
            [pt(18 * s, -7 * s), pt(18 * s, 7 * s), pt(-3 * s, 4 * s), (x, y)],
            fill=(158, 112, 63, 255),
        )
        draw.polygon(
            [(x, y), pt(-20 * s, -4 * s), pt(-20 * s, 4 * s), (x, y)],
            fill=(38, 28, 22, 245),
        )
        return

    if style == "rooster-quill":
        s = scale * 1.18
        shaft_len = 92 * s
        draw.line([pt(shaft_len, 0), (x, y)], fill=(48, 32, 24, 255), width=max(3, int(4 * s)))
        draw.polygon([(x, y), pt(-13 * s, -4 * s), pt(-13 * s, 4 * s)], fill=(190, 170, 138, 255))
        feather_colors = [(35, 35, 42, 235), (106, 28, 30, 230), (182, 42, 35, 220)]
        for side, color in ((-1, feather_colors[0]), (1, feather_colors[1])):
            vane = [(x - ux * 16 * s, y - uy * 16 * s)]
            for i in range(9):
                t = 16 * s + 76 * s * i / 8
                width = 10 * s * (0.35 + 0.65 * math.sin(math.pi * i / 8))
                vane.append((x - ux * t + nx * side * width, y - uy * t + ny * side * width))
            vane.append((x - ux * shaft_len, y - uy * shaft_len))
            draw.polygon(vane, fill=color)
        draw.line([pt(16 * s, 0), pt(shaft_len, 0)], fill=(214, 73, 48, 210), width=max(2, int(2 * s)))
        for i in range(5):
            t = (28 + i * 15) * s
            draw.line([pt(t, -5 * s), pt(t + 14 * s, -18 * s)], fill=(226, 75, 46, 190), width=max(1, int(2 * s)))
            draw.line([pt(t, 5 * s), pt(t + 14 * s, 18 * s)], fill=(28, 28, 34, 185), width=max(1, int(2 * s)))
        return

    if style == "quill":
        s = scale
        # Shaft
        shaft_len = 72 * s
        draw.line([pt(shaft_len, 0), (x, y)], fill=(160, 30, 25, 255), width=max(2, int(3 * s)))

        # Nib
        nib = 10 * s
        draw.polygon([
            (x, y),
            pt(-nib * 0.25, -3.5 * s),
            pt(-nib * 0.15, 0),
            pt(-nib * 0.25, 3.5 * s),
        ], fill=(60, 48, 32, 255))
        draw.polygon([
            pt(-nib * 0.15, 0),
            pt(nib * 0.85, -1.2 * s),
            pt(nib, 0),
            pt(nib * 0.85, 1.2 * s),
        ], fill=(100, 85, 60, 255))

        # Feather vanes — left
        vane_start = shaft_len * 0.22
        vane_len = shaft_len * 0.78
        vane_w = 16 * s
        left_vane = [(x - ux * vane_start, y - uy * vane_start)]
        for i in range(8):
            t = vane_start + vane_len * i / 7
            w = vane_w * (0.3 + 0.7 * math.sin(math.pi * i / 7))
            left_vane.append((x - ux * t + nx * (-w * 1.1), y - uy * t + ny * (-w * 1.1)))
        left_vane.append((x - ux * shaft_len, y - uy * shaft_len))
        draw.polygon(left_vane, fill=(200, 35, 30, 210))

        # Feather vanes — right
        right_vane = [(x - ux * vane_start, y - uy * vane_start)]
        for i in range(8):
            t = vane_start + vane_len * i / 7
            w = vane_w * (0.3 + 0.7 * math.sin(math.pi * i / 7))
            right_vane.append((x - ux * t + nx * (w * 1.05), y - uy * t + ny * (w * 1.05)))
        right_vane.append((x - ux * shaft_len, y - uy * shaft_len))
        draw.polygon(right_vane, fill=(220, 45, 40, 200))

        # Rachis highlight (center line of feather)
        draw.line([pt(vane_start, 0), pt(shaft_len, 0)], fill=(180, 50, 45, 140), width=max(1, int(2.5 * s)))

        # Barb splits
        for i in range(4):
            t = vane_start + vane_len * (i + 0.5) / 4
            split = vane_w * (0.2 + 0.3 * math.sin(math.pi * i / 3.5))
            draw.line(
                [pt(t, -split * 0.3), pt(t, -split * 0.85)],
                fill=(140, 25, 20, 100), width=max(1, int(1.5 * s)),
            )
            draw.line(
                [pt(t, split * 0.3), pt(t, split * 0.8)],
                fill=(140, 25, 20, 100), width=max(1, int(1.5 * s)),
            )
        return

    # Default procedural stylus
    draw.line([pt(55 * scale, 0), (x, y)], fill=(20, 20, 20, 255), width=max(4, int(7 * scale)))
    draw.polygon([pt(7 * scale, -5 * scale), (x, y), pt(7 * scale, 5 * scale)], fill=(8, 8, 8, 255))
    draw.polygon([pt(50 * scale, 11 * scale), pt(96 * scale, 18 * scale), pt(104 * scale, 45 * scale), pt(46 * scale, 34 * scale)], fill=(239, 191, 147, 235))
    draw.polygon([pt(42 * scale, 20 * scale), pt(70 * scale, 24 * scale), pt(74 * scale, 52 * scale), pt(35 * scale, 44 * scale)], fill=(246, 204, 164, 245))


def available_hands() -> tuple[str, ...]:
    """Return valid hand cursor names."""

    return ("procedural", "none", "quill", *BUILTIN_HANDS)


def _hand_asset_path(hand: str) -> Path:
    filename = f"{hand}.png"
    package_path = Path(__file__).resolve().parent / "assets" / "hands" / filename
    if package_path.exists():
        return package_path
    skill_path = Path(__file__).resolve().parents[2] / "assets" / "hands" / filename
    if skill_path.exists():
        return skill_path
    raise ValueError(f"Unknown or missing hand asset: {hand}")


def _load_hand_cursor(hand: str | Path, resolution: tuple[int, int], hand_scale: float) -> HandCursor:
    name = str(hand)
    if name in BUILTIN_HANDS:
        image_path = _hand_asset_path(name)
        anchor_ratio = HAND_ANCHORS[name]
    else:
        image_path = Path(name).expanduser()
        if not image_path.exists():
            raise ValueError(f"Hand asset does not exist: {image_path}")
        anchor_ratio = (0.05, 0.30)

    image = Image.open(image_path).convert("RGBA")
    scale_factor = max(0.1, hand_scale) * max(0.35, min(resolution) / 640.0)
    if name in {"brush", "rooster-quill"}:
        # The photoreal asset is 1254px square; normalize it to the existing
        # hand cursor footprint before applying the output-size scale.
        scale_factor *= 0.12
    if abs(scale_factor - 1.0) > 0.01:
        image = image.resize((max(1, int(image.width * scale_factor)), max(1, int(image.height * scale_factor))), Image.Resampling.LANCZOS)
    anchor = (image.width * anchor_ratio[0], image.height * anchor_ratio[1])
    return HandCursor(image=image, anchor=anchor)


def _paste_hand_cursor(frame: Image.Image, x: float, y: float, cursor: HandCursor) -> Image.Image:
    dest = (int(round(x - cursor.anchor[0])), int(round(y - cursor.anchor[1])))
    rgba = frame.convert("RGBA")
    rgba.alpha_composite(cursor.image, dest)
    return rgba.convert("RGB")


def _stroke_cumulative(points: list[tuple[float, float]]) -> list[float]:
    cumulative = [0.0]
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        cumulative.append(total)
    return cumulative


def _spread_pause_indices(stroke_count: int, pause_count: int) -> set[int]:
    if stroke_count <= 1 or pause_count <= 0:
        return set()
    return {
        min(stroke_count - 2, max(0, round((idx + 1) * (stroke_count - 1) / (pause_count + 1))))
        for idx in range(pause_count)
    }


def _prepare_timeline(strokes: list[Stroke], draw_frames: int) -> list[TimelineStroke]:
    prepared: list[tuple[Stroke, list[float], float]] = []
    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        cumulative = _stroke_cumulative(stroke.points)
        geometric = cumulative[-1]
        if geometric <= 0:
            continue
        # Thick stamps / hatch blobs are short in arc length but visually large.
        # Budget draw-time by effective ink area so solid blocks don't flash by.
        fat = stroke.max_width(default=0.0)
        src = str(stroke.source)
        if fat >= 8.0:
            effective = max(geometric, fat * 1.8, geometric * max(1.0, fat / 4.0) * 0.15)
        elif src.startswith(("hatch", "ink-brush", "ink-direct", "ink-freehand", "blob")):
            effective = max(geometric, geometric * 0.35 + stroke.mean_width() * 0.8)
        else:
            effective = geometric
        prepared.append((stroke, cumulative, effective))
    total_length = sum(length for _, _, length in prepared)
    if total_length <= 0:
        return []

    strokes_per_frame = len(prepared) / max(1, draw_frames)
    pixels_per_frame = total_length / max(1, draw_frames)
    if strokes_per_frame >= 0.85 or pixels_per_frame >= 85:
        pause_ratio = 0.0
    elif strokes_per_frame >= 0.45 or pixels_per_frame >= 55:
        pause_ratio = 0.008
    else:
        pause_ratio = 0.03
    pause_count = min(max(0, int(round(draw_frames * pause_ratio))), max(0, len(prepared) - 1))
    pause_indices = _spread_pause_indices(len(prepared), pause_count)
    pause_unit = total_length / max(1, draw_frames - pause_count)
    cursor = 0.0
    timeline: list[TimelineStroke] = []
    for index, (stroke, cumulative, length) in enumerate(prepared):
        start = cursor
        end = start + length
        cursor = end + (pause_unit if index in pause_indices else 0.0)
        # Progress along the real geometry; `length` is only the timeline weight.
        geo_length = cumulative[-1]
        timeline.append(
            TimelineStroke(
                stroke=stroke,
                cumulative=cumulative,
                length=geo_length,
                start_unit=start,
                end_unit=end,
                pause_end_unit=cursor,
            )
        )
    return timeline


def _point_at_distance(points: list[tuple[float, float]], cumulative: list[float], distance: float) -> tuple[float, float]:
    if distance <= 0:
        return points[0]
    if distance >= cumulative[-1]:
        return points[-1]
    index = max(0, min(len(points) - 2, bisect.bisect_right(cumulative, distance) - 1))
    start, end = points[index], points[index + 1]
    seg_len = max(cumulative[index + 1] - cumulative[index], 1e-6)
    local = (distance - cumulative[index]) / seg_len
    return (start[0] + (end[0] - start[0]) * local, start[1] + (end[1] - start[1]) * local)


def _width_at_distance(widths: list[float] | None, cumulative: list[float], distance: float) -> float:
    if not widths or not cumulative:
        return 2.0
    if distance <= 0:
        return float(widths[0])
    if distance >= cumulative[-1]:
        return float(widths[-1])
    index = max(0, min(len(widths) - 2, bisect.bisect_right(cumulative, distance) - 1))
    seg_len = max(cumulative[index + 1] - cumulative[index], 1e-6)
    local = (distance - cumulative[index]) / seg_len
    return float(widths[index] + (widths[index + 1] - widths[index]) * local)


def _stroke_segment_between(
    points: list[tuple[float, float]],
    cumulative: list[float],
    start_distance: float,
    end_distance: float,
    widths: list[float] | None = None,
) -> tuple[list[tuple[float, float]], list[float] | None]:
    start_distance = max(0.0, min(cumulative[-1], start_distance))
    end_distance = max(start_distance, min(cumulative[-1], end_distance))
    if end_distance <= start_distance:
        return [], None
    use_widths = widths if widths and len(widths) == len(points) else None
    segment = [_point_at_distance(points, cumulative, start_distance)]
    segment_w: list[float] | None = (
        [_width_at_distance(widths, cumulative, start_distance)] if use_widths is not None else None
    )
    for index, distance in enumerate(cumulative[1:-1], start=1):
        if start_distance < distance < end_distance:
            segment.append(points[index])
            if segment_w is not None and use_widths is not None:
                segment_w.append(use_widths[index])
    segment.append(_point_at_distance(points, cumulative, end_distance))
    if segment_w is not None:
        segment_w.append(_width_at_distance(widths, cumulative, end_distance))
    deduped: list[tuple[float, float]] = []
    deduped_w: list[float] | None = [] if segment_w is not None else None
    for idx, point in enumerate(segment):
        if not deduped or math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) > 0.05:
            deduped.append(point)
            if deduped_w is not None and segment_w is not None:
                deduped_w.append(segment_w[idx])
    return deduped, deduped_w


def _line_width(base_width: float, progress: float, scale: int) -> int:
    pressure = 0.68 + 0.42 * math.sin(math.pi * max(0.0, min(1.0, progress)))
    return max(1, int(round(base_width * pressure * scale)))


def _stroke_base_width(stroke: Stroke, fallback: int) -> int:
    """Prefer measured ink width (墨迹厚度) when the extractor provided it."""
    measured = stroke.mean_width(default=0.0)
    if measured > 0:
        # Preserve measured broad ink masses; the extractor already measured
        # this width from the source raster.
        return max(1, min(64, int(round(measured))))
    return max(1, int(fallback))


def _draw_stroke_segment(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int,
    scale: int,
    widths: list[float] | None = None,
) -> None:
    if len(points) < 2:
        return
    scaled = [(x * scale, y * scale) for x, y in points] if scale != 1 else points
    # Variable-width path: short segments with local diameter.
    # Keep the measured raster width. It is the source of truth for water-ink
    # masses, so a small fixed cap would make the video thinner than preview.
    if widths and len(widths) == len(points) and (max(widths) - min(widths) > 0.75 or max(widths) >= 6.0):
        for (x0, y0), (x1, y1), w0, w1 in zip(scaled, scaled[1:], widths, widths[1:]):
            local = min(64.0, max(1.0, (w0 + w1) * 0.5))
            seg_w = max(1, int(round(local * scale)))
            draw.line([(x0, y0), (x1, y1)], fill=color, width=seg_w, joint="curve")
            radius = max(1, seg_w // 2)
            draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), fill=color)
            draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=color)
        return
    effective_width = max(1, min(width, 64 * max(1, scale)))
    draw.line(scaled, fill=color, width=effective_width, joint="curve")
    radius = max(1, effective_width // 2)
    for x, y in (scaled[0], scaled[-1]):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _draw_reveal_mask_segment(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    width: int,
    widths: list[float] | None = None,
) -> None:
    if len(points) < 2:
        return
    if widths and len(widths) == len(points) and (max(widths) - min(widths) > 0.75 or max(widths) >= 8.0):
        for (x0, y0), (x1, y1), w0, w1 in zip(points, points[1:], widths, widths[1:]):
            # Reveal a bit wider than the ink so line-art snap can catch antialiased edges.
            seg_w = max(2, int(round(((w0 + w1) * 0.5) * 1.15 + 2)))
            draw.line([(x0, y0), (x1, y1)], fill=255, width=seg_w, joint="curve")
            radius = max(1, seg_w // 2)
            draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), fill=255)
            draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=255)
        return
    draw.line(points, fill=255, width=width, joint="curve")
    radius = max(1, width // 2)
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)


def _present_canvas(canvas: Image.Image, resolution: tuple[int, int]) -> Image.Image:
    if canvas.size == resolution:
        return canvas.copy()
    return canvas.resize(resolution, Image.Resampling.LANCZOS)


def _open_ffmpeg_rawvideo_writer(out_path: Path, resolution: tuple[int, int], fps: int) -> tuple[subprocess.Popen[bytes], list[str]]:
    width, height = resolution
    preset, crf = _ffmpeg_encoding_options()
    command = [
        _ffmpeg(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        crf,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE), command


def _ffmpeg_encoding_options() -> tuple[str, str]:
    """Return fast, high-quality defaults with environment overrides."""

    preset = os.getenv("WHITEBOARD_FFMPEG_PRESET", "veryfast").strip() or "veryfast"
    crf = os.getenv("WHITEBOARD_FFMPEG_CRF", "16").strip() or "16"
    return preset, crf


def _write_ffmpeg_frame(process: subprocess.Popen[bytes], frame: Image.Image) -> None:
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is not available")
    payload = frame.tobytes() if frame.mode == "RGB" else frame.convert("RGB").tobytes()
    process.stdin.write(payload)


def _smooth_angle(history: list[float], angle: float, window: int = 5) -> float:
    history.append(angle)
    del history[:-window]
    return math.atan2(sum(math.sin(item) for item in history), sum(math.cos(item) for item in history))


def _top_down_block_fill(canvas: Image.Image, source: Image.Image, progress: float, blocks: int = 18) -> Image.Image:
    """Reveal source image in hard horizontal blocks from top to bottom."""

    width, height = canvas.size
    if progress <= 0:
        return canvas.copy()
    if progress >= 1:
        return source.copy()
    block_count = max(1, int(blocks))
    visible_blocks = max(1, min(block_count, math.ceil(progress * block_count)))
    reveal_y = min(height, math.ceil(height * visible_blocks / block_count))
    frame = canvas.copy()
    frame.paste(source.crop((0, 0, width, reveal_y)), (0, 0))
    return frame


def _top_down_brush_fill(
    canvas: Image.Image,
    source: Image.Image,
    progress: float,
    blocks: int = 18,
) -> tuple[Image.Image, tuple[float, float] | None, float]:
    """Reveal source in horizontal brush passes and return brush position."""

    width, height = canvas.size
    if progress <= 0:
        return canvas.copy(), None, 0.0
    if progress >= 1:
        return source.copy(), None, 0.0

    block_count = max(1, int(blocks))
    scaled = max(0.0, min(0.999999, progress)) * block_count
    active = min(block_count - 1, int(scaled))
    local = scaled - active
    y0 = int(round(height * active / block_count))
    y1 = int(round(height * (active + 1) / block_count))
    frame = canvas.copy()
    if y0 > 0:
        frame.paste(source.crop((0, 0, width, y0)), (0, 0))

    brush_pad = max(6, height // 180)
    if active % 2 == 0:
        x = int(round(width * ease_in_out_sine(local)))
        if x > 0:
            frame.paste(source.crop((0, max(0, y0 - brush_pad), x, min(height, y1 + brush_pad))), (0, max(0, y0 - brush_pad)))
        cursor = (float(x), float((y0 + y1) / 2))
        angle = 0.0
    else:
        x = int(round(width * (1.0 - ease_in_out_sine(local))))
        if x < width:
            frame.paste(source.crop((x, max(0, y0 - brush_pad), width, min(height, y1 + brush_pad))), (x, max(0, y0 - brush_pad)))
        cursor = (float(x), float((y0 + y1) / 2))
        angle = math.pi
    return frame, cursor, angle


def _prepare_contour_fill_cache(canvas: Image.Image) -> ContourFillCache:
    """Build an edge-to-interior field for organic region coloring."""

    width, height = canvas.size
    gray = canvas.convert("L")
    ink = gray.point(lambda px: 255 if px < 218 else 0)
    spread = max(3, min(17, min(width, height) // 64))
    if spread % 2 == 0:
        spread += 1
    resistance_img = ink.filter(ImageFilter.MaxFilter(spread)).filter(
        ImageFilter.GaussianBlur(radius=max(1.5, min(width, height) / 220.0))
    )
    resistance = np.asarray(resistance_img, dtype=np.float32) / 255.0

    # Pull resistance downward so color appears to catch on contours before
    # slowly passing them, instead of crossing as a flat horizontal scan line.
    drag = resistance.copy()
    decay = 0.86
    for row in range(1, height):
        drag[row] = np.maximum(drag[row], drag[row - 1] * decay)
    drag = np.clip(drag, 0.0, 1.0)

    # Pixels close to a drawn contour are painted first; the center of each
    # enclosed form follows later. This avoids a single global scan direction.
    ink_mask = np.asarray(gray, dtype=np.uint8) < 218
    try:
        import cv2

        distance = cv2.distanceTransform((~ink_mask).astype(np.uint8), cv2.DIST_L2, 3)
    except Exception:
        # Small NumPy fallback for environments without OpenCV.
        distance = np.full((height, width), np.inf, dtype=np.float32)
        frontier = ink_mask.copy()
        distance[frontier] = 0.0
        for step in range(1, max(2, min(160, max(width, height)))):
            padded = np.pad(frontier, 1, mode="constant", constant_values=False)
            expanded = (
                padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:]
                | padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:]
                | padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
            )
            new_pixels = expanded & ~frontier
            if not np.any(new_pixels):
                break
            distance[new_pixels] = float(step)
            frontier = expanded
        distance[~np.isfinite(distance)] = float(max(width, height))
    max_distance = max(1.0, float(np.percentile(distance[~ink_mask], 92)) if np.any(~ink_mask) else 1.0)
    priority = np.clip(distance / max_distance, 0.0, 1.0)
    # A gentle deterministic texture makes the boundary feel like a brush,
    # without producing large diagonal or horizontal bands.
    yy, xx = np.mgrid[0:height, 0:width]
    texture = (
        np.sin(xx / max(17.0, width / 13.0) + yy / max(23.0, height / 11.0))
        + 0.45 * np.sin(xx / max(7.0, width / 31.0) - yy / max(11.0, height / 25.0))
    ) * 0.035
    priority = np.clip(priority + texture, 0.0, 1.0).astype(np.float32)
    x = np.arange(width, dtype=np.float32)
    primary = np.sin(x / max(24.0, width / 20.0))
    secondary = np.sin(x / max(8.0, width / 72.0) + 1.7) * 0.35
    return ContourFillCache(
        resistance=drag,
        rows=np.arange(height, dtype=np.float32)[:, None],
        wave=(primary + secondary).astype(np.float32),
        priority=priority,
    )


def _contour_wipe_fill(
    canvas: Image.Image,
    source: Image.Image,
    progress: float,
    blocks: int = 18,
    cache: ContourFillCache | None = None,
) -> tuple[Image.Image, tuple[float, float] | None, float]:
    """Reveal color top-down with a contour-blocked moving boundary."""

    width, height = canvas.size
    if progress <= 0:
        return canvas.copy(), None, 0.0
    if progress >= 1:
        return source.copy(), None, 0.0
    if source.size != canvas.size:
        source = source.resize(canvas.size, Image.Resampling.LANCZOS)
    cache = cache or _prepare_contour_fill_cache(canvas)

    p = ease_in_out_sine(progress)
    # Reveal by local contour distance, with a soft threshold rather than a
    # top-to-bottom boundary.
    threshold = p * 1.08 - 0.08
    reveal = cache.priority <= threshold

    canvas_arr = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    source_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
    # Feather the moving boundary so color grows through the line art instead
    # of appearing as a hard rectangular wipe.
    feather_radius = max(2.0, min(12.0, min(width, height) / 100.0))
    reveal_img = Image.fromarray((reveal.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )
    alpha = np.asarray(reveal_img, dtype=np.float32) / 255.0
    out = canvas_arr * (1.0 - alpha[..., None]) + source_arr * alpha[..., None]
    frame = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")

    pass_count = max(1, int(blocks))
    phase = (progress * pass_count) % 1.0
    lane = int(progress * pass_count)
    eased = ease_in_out_sine(phase)
    cursor_x = int(round((eased if lane % 2 == 0 else 1.0 - eased) * (width - 1)))
    candidates = np.argwhere(reveal)
    if len(candidates):
        sample = candidates[int((progress * 997) % len(candidates))]
        cursor = (float(sample[1]), float(sample[0]))
    else:
        cursor = (float(width * 0.5), float(height * 0.5))
    angle = math.sin(progress * math.tau) * 0.35
    return frame, cursor, angle


def _color_fill_frame(
    canvas: Image.Image,
    source: Image.Image,
    progress: float,
    mode: str = "contour-wipe",
    blocks: int = 18,
    contour_cache: ContourFillCache | None = None,
) -> tuple[Image.Image, tuple[float, float] | None, float]:
    if mode in {"brush-path", "stroke-follow", "笔触跟随"}:
        # Color is revealed only by the active drawing stroke. Do not run a
        # second full-canvas fill pass after the linework is complete.
        return canvas.copy(), None, 0.0
    if mode == "fade":
        return Image.blend(canvas, source, progress), None, 0.0
    if mode == "top-down-blocks":
        return _top_down_block_fill(canvas, source, progress, blocks=blocks), None, 0.0
    if mode == "brush-scan":
        return _top_down_brush_fill(canvas, source, progress, blocks=blocks)
    if mode == "contour-wipe":
        return _contour_wipe_fill(canvas, source, progress, blocks=blocks, cache=contour_cache)
    return _top_down_brush_fill(canvas, source, progress, blocks=blocks)


def _reveal_source_along_strokes(
    canvas: Image.Image,
    source: Image.Image,
    reveal_mask: Image.Image,
    strength: float = 0.88,
) -> Image.Image:
    """Reveal source colors under the active hand-drawn stroke area."""

    if source.size != canvas.size:
        source = source.resize(canvas.size, Image.Resampling.LANCZOS)
    radius = max(1.0, min(7.0, min(canvas.size) / 180.0))
    soft = reveal_mask.filter(ImageFilter.GaussianBlur(radius=radius))
    alpha = soft.point(lambda px: int(max(0, min(255, px * max(0.0, min(1.0, strength))))))
    return Image.composite(source.convert("RGB"), canvas.convert("RGB"), alpha)


def _generate_doodle_fill_strokes(width: int, height: int, density: int = 5, rng_seed: int = 42) -> list:
    """Generate casual doodle/scribble fill paths covering the canvas.

    Returns list of (points, brush_width) tuples, ordered from broad coverage
    strokes to detail texture strokes.
    """
    rng = np.random.RandomState(rng_seed)
    strokes: list[tuple[list[tuple[float, float]], float]] = []
    min_dim = min(width, height)

    row_spacing = max(22, min_dim // max(density, 3))
    for y_base in range(row_spacing // 2, height, row_spacing):
        pts: list[tuple[float, float]] = []
        x = 0.0
        y = float(y_base)
        direction = 1.0
        while x < width:
            pts.append((x, y))
            step = rng.uniform(18, 50)
            x += step * direction
            y += rng.uniform(-row_spacing * 0.28, row_spacing * 0.28)
            y = max(1, min(height - 2, y))
            if rng.random() < 0.06:
                direction *= -1.0
                if x > 0:
                    x -= step * 0.3
        if len(pts) >= 2:
            pts = [(px + rng.uniform(-3, 3), py + rng.uniform(-3, 3)) for px, py in pts]
            strokes.append((pts, rng.uniform(10, 18)))

    diag_gap = max(28, min_dim // max(density - 1, 2))
    for base in range(-height, width + height, diag_gap):
        pts = []
        for t in range(0, max(width, height) * 3, int(rng.uniform(14, 38))):
            px = base + t * 0.55
            py = t * 0.55
            if 0 <= px < width and 0 <= py < height:
                pts.append((px + rng.uniform(-4, 4), py + rng.uniform(-4, 4)))
        if len(pts) >= 3:
            strokes.append((pts, rng.uniform(7, 12)))

    for base in range(-height, width * 2 + height, diag_gap):
        pts = []
        for t in range(0, max(width, height) * 3, int(rng.uniform(14, 38))):
            px = base - t * 0.55
            py = t * 0.55
            if 0 <= px < width and 0 <= py < height:
                pts.append((px + rng.uniform(-4, 4), py + rng.uniform(-4, 4)))
        if len(pts) >= 3:
            strokes.append((pts, rng.uniform(7, 12)))

    num_curls = max(2, width * height // 60000)
    for _ in range(num_curls):
        pts = []
        cx = rng.uniform(width * 0.15, width * 0.85)
        cy = rng.uniform(height * 0.15, height * 0.85)
        radius = rng.uniform(25, min_dim * 0.25)
        steps = int(rng.uniform(50, 130))
        for i in range(steps):
            angle_val = i * rng.uniform(0.08, 0.25)
            r = radius * (0.7 + 0.3 * math.sin(angle_val * 2.3))
            px = cx + r * math.cos(angle_val)
            py = cy + r * math.sin(angle_val * 0.7)
            if 0 <= px < width and 0 <= py < height:
                pts.append((px + rng.uniform(-3, 3), py + rng.uniform(-3, 3)))
        if len(pts) >= 3:
            strokes.append((pts, rng.uniform(5, 9)))

    num_scribbles = max(5, width * height // 30000)
    for _ in range(num_scribbles):
        pts = []
        sx = rng.uniform(0, width)
        sy = rng.uniform(0, height)
        steps = int(rng.uniform(4, 20))
        angle_val = rng.uniform(0, math.tau)
        for _ in range(steps):
            angle_val += rng.uniform(-0.5, 0.5)
            step = rng.uniform(8, 35)
            sx += step * math.cos(angle_val)
            sy += step * math.sin(angle_val)
            if 0 <= sx < width and 0 <= sy < height:
                pts.append((sx + rng.uniform(-2, 2), sy + rng.uniform(-2, 2)))
        if len(pts) >= 2:
            strokes.append((pts, rng.uniform(4, 8)))

    return strokes


def _generate_natural_repair_strokes(source: Image.Image, density: int = 5) -> list:
    """Create irregular repair strokes only over visible source content."""

    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(source.convert("L"), dtype=np.uint8)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    foreground = (gray < 246) | (chroma > 14)
    if not np.any(foreground):
        return []
    dist = _distance_transform(foreground)
    strokes = _freehand_scribble_strokes(
        foreground,
        dist=dist,
        min_area=max(24, int(min(source.size) * 0.012)),
        source="color-repair",
        color=(0, 0, 0),
        max_brush=max(8.0, min(source.size) / max(24.0, density * 3.5)),
    )
    return [(stroke.points, max(3.0, stroke.mean_width(default=8.0))) for stroke in strokes]


def _skin_tone_mask(source: Image.Image) -> np.ndarray:
    """Detect light through dark skin colors without treating white as skin."""

    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    ycbcr = np.asarray(source.convert("YCbCr"), dtype=np.uint8)
    y = ycbcr[..., 0].astype(np.int16)
    cb = ycbcr[..., 1].astype(np.int16)
    cr = ycbcr[..., 2].astype(np.int16)
    rgb_f = rgb.astype(np.float32)
    maximum = rgb_f.max(axis=2)
    minimum = rgb_f.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
    skin = (
        (y >= 45)
        & (cb >= 76)
        & (cb <= 135)
        & (cr >= 132)
        & (cr <= 182)
        & (saturation >= 0.07)
        & (saturation <= 0.58)
    )
    cleaned = Image.fromarray((skin.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MedianFilter(3))
    return np.asarray(cleaned, dtype=np.uint8) > 0


def _generate_skin_repair_strokes(source: Image.Image, density: int = 5) -> list:
    """Create soft, coherent coloring passes over detected faces and limbs."""

    skin = _skin_tone_mask(source)
    if not np.any(skin):
        return []
    height, width = skin.shape
    row_step = max(3, min(9, min(source.size) // max(70, density * 14)))
    brush_width = float(max(5, int(round(row_step * 1.8))))
    strokes: list[tuple[list[tuple[float, float]], float]] = []
    for row_index, y in enumerate(range(row_step // 2, height, row_step)):
        row = skin[y]
        transitions = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        for run_index, (x0, x1) in enumerate(zip(starts.tolist(), ends.tolist())):
            if x1 - x0 < 3:
                continue
            sample_step = max(2, row_step)
            xs = list(range(x0, x1, sample_step))
            if not xs or xs[-1] != x1 - 1:
                xs.append(x1 - 1)
            points = [(float(x), float(y + math.sin((x + y) * 0.11) * row_step * 0.18)) for x in xs]
            if (row_index + run_index) % 2:
                points.reverse()
            if len(points) >= 2:
                strokes.append((points, brush_width))
    return strokes


def _generate_contour_repair_strokes(line_art: Image.Image, size: tuple[int, int], density: int = 5) -> list:
    """Create loose color strokes only over extracted contour ink."""

    gray = line_art.convert("L")
    if gray.size != size:
        gray = gray.resize(size, Image.Resampling.LANCZOS)
    contour = np.asarray(gray, dtype=np.uint8) < 235
    if not np.any(contour):
        return []
    strokes = _freehand_scribble_strokes(
        contour,
        dist=_distance_transform(contour),
        min_area=max(8, int(min(size) * 0.004)),
        source="color-repair-contour",
        color=(0, 0, 0),
        max_brush=max(4.0, min(size) / max(50.0, density * 6.0)),
    )
    return [(stroke.points, max(2.0, stroke.mean_width(default=5.0))) for stroke in strokes]


def _generate_lineart_gap_strokes(
    source: Image.Image,
    line_art: Image.Image,
    size: tuple[int, int],
    density: int = 5,
) -> list:
    """Find source-image paint regions not represented by the line-art mask."""

    source_rgb = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    line_gray = line_art.convert("L")
    if line_gray.size != size:
        line_gray = line_gray.resize(size, Image.Resampling.LANCZOS)
    rgb = np.asarray(source_rgb, dtype=np.uint8)
    gray = np.asarray(source_rgb.convert("L"), dtype=np.uint8)
    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    source_content = (gray < 247) | (chroma > 12)
    line_ink = np.asarray(line_gray, dtype=np.uint8) < 235
    # Thick ink bodies (filled black hair/clothes) must still be colored, while
    # thin contour lines (distance ~1-2) must NOT be painted over. A distance
    # transform on the ink separates the two purely by ink thickness.
    try:
        solid_dist_thresh = float(os.getenv("WHITEBOARD_SOLID_DIST_THRESH", "3.5"))
    except ValueError:
        solid_dist_thresh = 3.5
    try:
        dark_thresh = int(os.getenv("WHITEBOARD_DARK_THRESH", "90"))
    except ValueError:
        dark_thresh = 90
    line_dist = _distance_transform(line_ink)
    solid_body = line_ink & (line_dist > solid_dist_thresh)
    dark_src = gray < dark_thresh
    # Thin contour lines (line_ink but not a thick body) stay as line art and are
    # NOT painted over. Everything else that has real content — mid-tone ink
    # washes, skin, soft gradients, and thick black bodies — becomes paintable so
    # it is revealed by progressive strokes rather than a single global fade.
    thin_line = line_ink & ~solid_body
    paintable = source_content & (~thin_line | (solid_body & dark_src))
    # Skin has dedicated soft passes. Excluding it here prevents the same face
    # from being painted twice before clothes and background colors are reached.
    paintable &= ~_skin_tone_mask(source_rgb)
    if not np.any(paintable):
        return []
    strokes = _freehand_scribble_strokes(
        paintable,
        dist=_distance_transform(paintable),
        min_area=max(12, int(min(size) * 0.006)),
        source="color-lineart-gap",
        color=(0, 0, 0),
        max_brush=max(4.0, min(size) / max(42.0, density * 5.5)),
    )
    return [(stroke.points, max(2.0, stroke.mean_width(default=5.0))) for stroke in strokes]


def _estimate_color_complexity(source: Image.Image) -> float:
    """Estimate actual color-paint workload rather than raw image detail."""

    sample = source.convert("RGB").resize((48, 48), Image.Resampling.BILINEAR)
    arr = np.asarray(sample, dtype=np.float32) / 255.0
    chroma_map = arr.max(axis=2) - arr.min(axis=2)
    gray = arr.mean(axis=2)
    paint_area = np.mean((gray < 0.96) | (chroma_map > 0.08))
    quantized = (arr * 5.0).astype(np.uint8)
    color_diversity = min(1.0, len(np.unique(quantized.reshape(-1, 3), axis=0)) / 90.0)
    horizontal = np.mean(np.any(quantized[:, 1:] != quantized[:, :-1], axis=2))
    vertical = np.mean(np.any(quantized[1:] != quantized[:-1], axis=2))
    region_boundaries = min(1.0, (horizontal + vertical) * 1.8)
    return max(0.0, min(1.0, paint_area * 0.35 + color_diversity * 0.25 + region_boundaries * 0.40))


def _color_finish_start(source: Image.Image) -> float:
    """Delay whole-image finishing until detailed colors have been hand-painted."""

    # Complex artwork gets a slightly later finish, but still leaves enough
    # frames for a visible repair transition instead of a last-frame jump.
    return 0.84 + _estimate_color_complexity(source) * 0.04


def _order_color_strokes(
    strokes: list[tuple[list[tuple[float, float]], float]],
    size: tuple[int, int],
) -> list[tuple[list[tuple[float, float]], float]]:
    """Finish one nearby color region before moving the tool elsewhere."""

    usable = [(list(points), width) for points, width in strokes if len(points) > 1]
    if len(usable) < 2:
        return usable
    tile = max(28.0, min(size) * 0.075)
    groups: dict[tuple[int, int], list[tuple[list[tuple[float, float]], float]]] = {}
    for points, width in usable:
        cx = sum(point[0] for point in points) / len(points)
        cy = sum(point[1] for point in points) / len(points)
        groups.setdefault((int(cx // tile), int(cy // tile)), []).append((points, width))

    remaining_tiles = set(groups)
    current: tuple[float, float] | None = None
    ordered: list[tuple[list[tuple[float, float]], float]] = []
    while remaining_tiles:
        if current is None:
            tile_key = min(remaining_tiles, key=lambda key: (key[1], key[0]))
        else:
            tile_key = min(
                remaining_tiles,
                key=lambda key: (key[0] * tile + tile / 2 - current[0]) ** 2
                + (key[1] * tile + tile / 2 - current[1]) ** 2,
            )
        remaining_tiles.remove(tile_key)
        local = groups[tile_key]
        while local:
            if current is None:
                index = min(range(len(local)), key=lambda idx: (local[idx][0][0][1], local[idx][0][0][0]))
            else:
                index = min(
                    range(len(local)),
                    key=lambda idx: min(
                        (local[idx][0][0][0] - current[0]) ** 2 + (local[idx][0][0][1] - current[1]) ** 2,
                        (local[idx][0][-1][0] - current[0]) ** 2 + (local[idx][0][-1][1] - current[1]) ** 2,
                    ),
                )
            points, width = local.pop(index)
            if current is not None:
                start_distance = (points[0][0] - current[0]) ** 2 + (points[0][1] - current[1]) ** 2
                end_distance = (points[-1][0] - current[0]) ** 2 + (points[-1][1] - current[1]) ** 2
                if end_distance < start_distance:
                    points.reverse()
            ordered.append((points, width))
            current = points[-1]
    return ordered


def _doodle_fill_frame(
    canvas: Image.Image,
    source: Image.Image,
    progress: float,
    doodle_strokes: list,
    mask: Image.Image,
    mask_draw: ImageDraw.ImageDraw,
    last_idx: int,
    global_gradient: bool = False,
    skin_mask: np.ndarray | None = None,
    finish_start: float | None = None,
) -> tuple[Image.Image, tuple[float, float] | None, float, int]:
    """Reveal color source through progressive doodle scribble strokes."""
    total = len(doodle_strokes)
    if total == 0 or progress <= 0:
        return canvas.copy(), None, 0.0, last_idx
    if progress >= 1:
        return source.copy(), None, 0.0, total

    target = int(progress * total)
    cursor = None
    angle = 0.0

    while last_idx < target and last_idx < total:
        pts, bw = doodle_strokes[last_idx]
        width_int = max(1, int(bw))
        for i in range(len(pts) - 1):
            local_w = max(1, int(width_int * (0.85 + 0.3 * math.sin(i * 0.7))))
            mask_draw.line([pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]], fill=255, width=local_w)
        if len(pts) >= 2:
            cursor = (pts[-1][0], pts[-1][1])
            angle = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
        last_idx += 1

    frac = (progress * total) - last_idx
    if frac > 0 and last_idx < total:
        pts, bw = doodle_strokes[last_idx]
        n = max(1, int(len(pts) * frac))
        sub = pts[:n]
        width_int = max(1, int(bw))
        if len(sub) >= 2:
            for i in range(len(sub) - 1):
                local_w = max(1, int(width_int * (0.85 + 0.3 * math.sin(i * 0.7))))
                mask_draw.line([sub[i][0], sub[i][1], sub[i + 1][0], sub[i + 1][1]], fill=255, width=local_w)
            cursor = (sub[-1][0], sub[-1][1])
            angle = math.atan2(sub[-1][1] - sub[-2][1], sub[-1][0] - sub[-2][0])

    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=1.35))
    mask_arr = np.asarray(soft_mask, dtype=np.float32) / 255.0
    if skin_mask is not None and np.any(skin_mask):
        # Skin paint is naturally blended with a soft brush. Expand only the
        # part of the reveal that intersects skin, avoiding white holes and
        # the abrupt whole-face jump that used to happen near completion.
        expanded = mask.filter(ImageFilter.GaussianBlur(radius=1.8))
        expanded_arr = np.minimum(1.0, np.asarray(expanded, dtype=np.float32) / 255.0 * 1.4)
        mask_arr = np.maximum(mask_arr, expanded_arr * skin_mask)
    mask_3d = np.stack([mask_arr] * 3, axis=-1)
    canvas_arr = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    source_arr = np.asarray(source.convert("RGB"), dtype=np.float32)
    out = canvas_arr * (1.0 - mask_3d) + source_arr * mask_3d
    # Brush marks lead, then a continuous finishing blend closes every small
    # uncovered gap. This reaches the original smoothly instead of jumping
    # from a capped 92-98% frame straight to 100% at the stamp transition.
    blend_start = finish_start if finish_start is not None else (0.90 if global_gradient else 0.94)
    blend_start = max(0.0, min(0.99, float(blend_start)))
    finish_blend = ease_in_out_sine(max(0.0, min(1.0, (progress - blend_start) / (1.0 - blend_start))))
    if finish_blend > 0:
        out = out * (1.0 - finish_blend) + source_arr * finish_blend
    frame = Image.fromarray(out.astype(np.uint8), mode="RGB")
    return frame, cursor, angle, last_idx


def _allocate_render_frames(
    total_frames: int,
    fps: int,
    color_seconds: float,
) -> tuple[int, int, int]:
    """Allocate drawing, coloring, and stamp frames without stealing color time."""

    total_frames = max(1, int(total_frames))
    stamp_frames = min(
        max(1, int(round(max(1, fps) * 1.2))),
        max(1, total_frames // 8),
    )
    if color_seconds <= 0:
        return max(1, total_frames - stamp_frames), 0, stamp_frames
    min_draw_frames = max(1, int(round(total_frames * 0.40)))
    requested_color_frames = max(1, int(round(color_seconds * max(1, fps))))
    available_color_frames = max(0, total_frames - stamp_frames - min_draw_frames)
    color_frames = min(requested_color_frames, available_color_frames)
    draw_frames = max(1, total_frames - stamp_frames - color_frames)
    return draw_frames, color_frames, stamp_frames


def _generate_quill_fill_strokes(width: int, height: int, density: int = 5, rng_seed: int = 42) -> list:
    """Generate quill-pen-style fill strokes: elegant cross-hatching.

    Returns list of (points, brush_width) tuples. Strokes are long, parallel
    hatch lines at multiple angles — thinner and more orderly than doodle-fill.
    """
    rng = np.random.RandomState(rng_seed)
    strokes: list[tuple[list[tuple[float, float]], float]] = []
    min_dim = min(width, height)
    spacing = max(16, min_dim // max(density, 3))

    angles_deg = [15, 75, 45, -30, -60, 0]

    for angle_deg in angles_deg:
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        bw = rng.uniform(3.5, 7.0)

        diag = math.hypot(width, height)
        step_count = int(diag / spacing) + 2
        for i in range(step_count):
            offset = spacing * i
            pts: list[tuple[float, float]] = []

            proj = -diag + offset
            while proj < diag * 2:
                px = proj * cos_a
                py = proj * sin_a
                if 0 <= px < width and 0 <= py < height:
                    pts.append((px + rng.uniform(-2, 2), py + rng.uniform(-2, 2)))
                else:
                    if pts:
                        if len(pts) >= 2:
                            strokes.append((pts, bw))
                        pts = []
                proj += rng.uniform(25, 55)
            if len(pts) >= 2:
                strokes.append((pts, bw))

    for angle_deg in [90, -15]:
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        bw = rng.uniform(2.5, 5.5)
        diag = math.hypot(width, height)
        step_count = int(diag / spacing) + 2
        for i in range(step_count):
            offset = spacing * i
            pts = []
            proj = -diag + offset
            while proj < diag * 2:
                px = proj * cos_a + rng.uniform(-1.5, 1.5)
                py = proj * sin_a + rng.uniform(-1.5, 1.5)
                if 0 <= px < width and 0 <= py < height:
                    pts.append((px, py))
                else:
                    if pts:
                        if len(pts) >= 2:
                            strokes.append((pts, bw))
                        pts = []
                proj += rng.uniform(30, 60)
            if len(pts) >= 2:
                strokes.append((pts, bw))

    return strokes


def _parse_seal_coords(position: str) -> tuple[float, float] | None:
    """Parse a free seal position given as "x,y" percentages (0-100).

    Returns (px, py) when the value is a valid coordinate pair, otherwise None
    so callers fall back to the named 9-grid presets (left-center, etc.).
    """
    if not isinstance(position, str) or "," not in position:
        return None
    parts = position.split(",")
    if len(parts) != 2:
        return None
    try:
        px = float(parts[0].strip())
        py = float(parts[1].strip())
    except ValueError:
        return None
    if not (0.0 <= px <= 100.0 and 0.0 <= py <= 100.0):
        return None
    return px, py


def _apply_red_seal(
    frame: Image.Image,
    progress: float,
    text: str = "老林涂鸦",
    style: str = "white-text",
    position: str = "left-center",
) -> Image.Image:
    """Animate a physical seal and reveal the selected custom imprint."""

    progress = max(0.0, min(1.0, float(progress)))
    width, height = frame.size
    min_side = min(width, height)
    seal_size = max(56, int(min_side * 0.14))
    rng = np.random.RandomState(23)
    stamp = Image.new("RGBA", (seal_size, seal_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stamp, "RGBA")
    style = style if style in {"vintage", "inkwash", "circle", "ellipse", "white-text", "borderless"} else "white-text"
    red = (154, 4, 20, 238) if style == "vintage" else (170, 6, 24, 226)
    edge = max(2, seal_size // 22)
    inset = max(4, seal_size // 14)
    gap = max(1, seal_size // 28)
    mid = seal_size // 2

    if style == "inkwash":
        # Overlapping wet impressions create an organic edge and translucent
        # deposits instead of a mechanically perfect square.
        pad = max(3, seal_size // 18)
        draw.rounded_rectangle(
            (inset, inset, seal_size - inset, seal_size - inset),
            radius=max(5, seal_size // 8),
            fill=(red[0], red[1], red[2], 216),
            outline=(126, 2, 17, 238),
            width=max(2, edge),
        )
        for _ in range(24):
            px = int(rng.uniform(pad, seal_size - pad))
            py = int(rng.uniform(pad, seal_size - pad))
            rx = int(rng.uniform(seal_size * 0.13, seal_size * 0.31))
            ry = int(rng.uniform(seal_size * 0.10, seal_size * 0.27))
            alpha = int(rng.uniform(145, 225))
            draw.ellipse((px - rx, py - ry, px + rx, py + ry), fill=(red[0], red[1], red[2], alpha))
        for _ in range(18):
            px = int(rng.uniform(inset, seal_size - inset))
            py = int(rng.uniform(inset, seal_size - inset))
            rx = int(rng.uniform(1, max(2, seal_size * 0.040)))
            ry = int(rng.uniform(1, max(2, seal_size * 0.022)))
            draw.ellipse((px - rx, py - ry, px + rx, py + ry), fill=(0, 0, 0, 0))
        for _ in range(14):
            px = int(rng.uniform(0, seal_size))
            py = int(rng.uniform(0, seal_size))
            radius = int(rng.uniform(1, max(2, seal_size * 0.026)))
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(red[0], red[1], red[2], int(rng.uniform(45, 150))))
    elif style in {"circle", "ellipse", "white-text"}:
        # Solid vermilion body with reversed (white) glyphs — 阴文/白文印。
        if style == "circle":
            draw.ellipse((inset, inset, seal_size - inset, seal_size - inset), fill=(red[0], red[1], red[2], 224))
        elif style == "ellipse":
            ex = max(2, seal_size // 7)
            draw.ellipse((ex, inset, seal_size - ex, seal_size - inset), fill=(red[0], red[1], red[2], 224))
        else:  # white-text
            draw.rounded_rectangle((inset, inset, seal_size - inset, seal_size - inset), radius=max(4, seal_size // 12), fill=(red[0], red[1], red[2], 228))
    elif style == "borderless":
        pass  # 无边框无底，仅朱红字，随形自然。
    else:
        # Four irregular ink blocks leave a deliberately visible white cross.
        for row in range(2):
            for col in range(2):
                x0 = inset if col == 0 else mid + gap
                y0 = inset if row == 0 else mid + gap
                x1 = mid - gap if col == 0 else seal_size - inset
                y1 = mid - gap if row == 0 else seal_size - inset
                points = [
                    (x0 + rng.randint(-2, 3), y0 + rng.randint(-2, 3)),
                    (x1 + rng.randint(-2, 3), y0 + rng.randint(-2, 3)),
                    (x1 + rng.randint(-2, 3), y1 + rng.randint(-2, 3)),
                    (x0 + rng.randint(-2, 3), y1 + rng.randint(-2, 3)),
                ]
                draw.polygon(points, fill=red)

    # Broken outer frame, with uneven pressure and dry edges.
    frame_points = []
    for side in range(4):
        for i in range(7):
            t = i / 6
            if side == 0:
                point = (inset + (seal_size - 2 * inset) * t, inset)
            elif side == 1:
                point = (seal_size - inset, inset + (seal_size - 2 * inset) * t)
            elif side == 2:
                point = (seal_size - inset - (seal_size - 2 * inset) * t, seal_size - inset)
            else:
                point = (inset, seal_size - inset - (seal_size - 2 * inset) * t)
            frame_points.append((int(point[0] + rng.uniform(-2, 2)), int(point[1] + rng.uniform(-2, 2))))
    if style == "vintage":
        for start in range(0, len(frame_points), 9):
            segment = frame_points[start:start + 7]
            if len(segment) >= 2:
                draw.line(segment, fill=red, width=edge, joint="curve")

    # Small transparent voids and darker deposits make the print feel pressed
    # into paper rather than rendered as a clean vector rectangle.
    for _ in range(12):
        side = rng.randint(0, 4)
        if side in {0, 2}:
            px = int(rng.uniform(inset, seal_size - inset))
            py = inset if side == 0 else seal_size - inset
        else:
            px = inset if side == 1 else seal_size - inset
            py = int(rng.uniform(inset, seal_size - inset))
        radius = int(rng.uniform(1, max(2, seal_size * 0.022)))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(0, 0, 0, 0))
    for _ in range(8):
        px = int(rng.uniform(0, seal_size))
        py = int(rng.uniform(0, seal_size))
        radius = int(rng.uniform(1, max(2, seal_size * 0.025)))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(159, 22, 28, int(rng.uniform(80, 180))))

    # Fit custom content into the imprint instead of silently truncating it.
    cleaned_text = "".join(str(text or "老林涂鸦").split()) or "老林涂鸦"
    chars = list(cleaned_text[:12])
    columns = max(1, int(math.ceil(math.sqrt(len(chars)))))
    rows = max(1, int(math.ceil(len(chars) / columns)))
    content_size = seal_size - 2 * inset
    cell_w = max(8, content_size // columns)
    cell_h = max(8, content_size // rows)
    font = _load_seal_font(max(10, int(min(cell_w, cell_h) * 0.70)))
    glyph_color = red if style == "borderless" else (255, 247, 235, 255)
    for index, char in enumerate(chars):
        row, col = divmod(index, columns)
        left_box, top_box, right_box, bottom_box = draw.textbbox((0, 0), char, font=font)
        cell_x = inset + col * cell_w
        cell_y = inset + row * cell_h
        x = cell_x + (cell_w - (right_box - left_box)) / 2 - left_box + rng.uniform(-1, 1)
        y = cell_y + (cell_h - (bottom_box - top_box)) / 2 - top_box + rng.uniform(-1, 1)
        draw.text((x, y), char, font=font, fill=glyph_color)

    stamp = stamp.rotate(-2.0 + 2.0 * ease_in_out_sine(progress), resample=Image.Resampling.BICUBIC, expand=True)
    margin = max(18, int(min_side * 0.075))
    coords = _parse_seal_coords(position)
    if coords is not None:
        # Free placement: (px, py) are 0-100 percentages of the frame and mark
        # the seal center, so any position is reachable, not just the 9 presets.
        px, py = coords
        x = int(round(width * px / 100.0 - stamp.width / 2))
        y = int(round(height * py / 100.0 - stamp.height / 2))
    else:
        horizontal, _, vertical = position.partition("-")
        if horizontal == "right":
            x = width - margin - stamp.width
        elif horizontal == "center":
            x = (width - stamp.width) // 2
        else:
            x = margin
        if vertical == "top":
            y = margin
        elif vertical == "bottom":
            y = height - margin - stamp.height
        else:
            y = (height - stamp.height) // 2
    x = max(0, min(width - stamp.width, x))
    y = max(0, min(height - stamp.height, y))
    result = frame.convert("RGBA")
    # Keep the print fully hidden while the physical stamp covers the paper.
    # It is revealed only after the stamp has lifted clear of the canvas.
    imprint_alpha = max(0.0, min(1.0, (progress - 0.79) / 0.13))
    if imprint_alpha > 0:
        stamp.putalpha(stamp.getchannel("A").point(lambda px: int(px * imprint_alpha)))
        result.alpha_composite(stamp, (x, y))

    # A solid stamp head falls, compresses into the red shape, then lifts.
    if progress < 0.78:
        tool_w = tool_h = max(82, int(seal_size * 1.42))
        yuxi_asset = _load_yuxi_asset()
        if yuxi_asset is not None:
            tool = yuxi_asset.resize((tool_w, tool_h), Image.Resampling.LANCZOS)
        else:
            # Minimal fallback used only when the generated project asset is
            # unavailable; normal renders use the detailed dragon-jade PNG.
            tool = Image.new("RGBA", (tool_w, tool_h), (0, 0, 0, 0))
            tool_draw = ImageDraw.Draw(tool, "RGBA")
            rim = max(5, tool_w // 10)
            tool_draw.rounded_rectangle(
                (rim, rim, tool_w - rim, tool_h - rim),
                radius=max(3, tool_w // 18),
                fill=(126, 3, 18, 255),
                outline=(196, 125, 54, 255),
                width=max(2, tool_w // 28),
            )
            tool_draw.ellipse(
                (tool_w * 0.30, tool_h * 0.30, tool_w * 0.70, tool_h * 0.70),
                fill=(91, 3, 15, 255),
                outline=(226, 72, 79, 255),
                width=max(2, tool_w // 30),
            )
        descend = max(0.0, min(1.0, (progress - 0.04) / 0.40))
        lift = max(0.0, min(1.0, (progress - 0.52) / 0.24))
        press = math.sin(max(0.0, min(1.0, (progress - 0.40) / 0.16)) * math.pi)

        # Move along camera depth, not along the canvas Y axis. The stamp stays
        # over the imprint while perspective scale makes it approach the paper
        # and then return toward the viewer.
        approach = ease_in_out_sine(max(0.0, min(1.0, (progress - 0.02) / 0.38)))
        exit_progress = ease_in_out_sine(lift)
        center_x = x + stamp.width / 2
        center_y = y + stamp.height / 2
        if progress < 0.52:
            depth_scale = 2.15 - 1.15 * approach
            rotation = -7.0 * (1.0 - approach)
            entry_visibility = max(0.0, min(1.0, (progress - 0.015) / 0.22))
            tool_opacity = ease_in_out_sine(entry_visibility)
        else:
            depth_scale = 1.0 + 1.35 * exit_progress
            rotation = 5.0 * exit_progress
            tool_opacity = max(0.0, 1.0 - exit_progress)
        depth_scale *= 1.0 - 0.045 * press
        scaled_tool = tool.resize(
            (max(1, int(round(tool.width * depth_scale))), max(1, int(round(tool.height * depth_scale)))),
            Image.Resampling.LANCZOS,
        )
        moving_tool = scaled_tool.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        if tool_opacity < 1.0:
            moving_tool.putalpha(moving_tool.getchannel("A").point(lambda px: int(px * tool_opacity)))
        tool_x = int(center_x - moving_tool.width / 2)
        tool_y = int(center_y - moving_tool.height / 2)
        # Contact shadow widens as the stamp approaches the canvas.
        if descend > 0.65 and lift < 0.35:
            shadow = Image.new("RGBA", result.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow, "RGBA")
            shadow_w = int(seal_size * (0.48 + 0.08 * min(1.0, press)))
            shadow_h = int(seal_size * (0.40 + 0.06 * min(1.0, press)))
            shadow_cx = x + stamp.width // 2
            shadow_cy = y + stamp.height // 2
            shadow_draw.ellipse(
                (shadow_cx - shadow_w, shadow_cy - shadow_h, shadow_cx + shadow_w, shadow_cy + shadow_h),
                fill=(48, 8, 12, int(42 * (1.0 - lift))),
            )
            result = Image.alpha_composite(result, shadow.filter(ImageFilter.GaussianBlur(radius=4.0)))
        result.alpha_composite(moving_tool, (tool_x, tool_y))
    return result.convert("RGB")


def render_scene(
    image_path: Path,
    strokes: list[Stroke],
    duration: float,
    out_path: Path,
    fps: int = 60,
    resolution: tuple[int, int] = (1920, 1080),
    tail_color_sec: float = 2.0,
    line_thickness: int | None = 0,
    show_cursor: bool = True,
    source_image: Image.Image | None = None,
    hand_style: str | Path = DEFAULT_DRAWING_TOOL,
    hand_scale: float = 1.0,
    color_hand_style: str | Path = "brush",
    color_hand_scale: float = 1.0,
    color_fill_mode: str = "natural-repair",
    color_fill_blocks: int = 18,
    theme_text: str | None = None,
    theme_font_style: str = "mao",
    theme_font_size: int = 72,
    theme_position: str = "right",
    seal_style: str = "white-text",
    seal_text: str = "老林涂鸦",
    seal_position: str = "left-center",
    complete_line_art: Image.Image | None = None,
    line_art_snap: bool = True,
    line_art_snap_threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD,
    draw_mode: str = "direct-ink",
    preserve_line_art_tones: bool = False,
    color_during_drawing: bool = True,
) -> None:
    """Render strokes into a hand-drawn MP4 scene.

    Example:
        >>> render_scene(Path("missing.png"), [], 1.0, Path("out.mp4"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
    """

    if not strokes:
        raise ValueError("No strokes to render")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = source_image.convert("RGB").resize(resolution) if source_image else load_on_canvas(image_path, resolution)
    aa_scale = _render_aa_scale(resolution, has_exact_line_art=complete_line_art is not None)
    estimated_line_width = _estimate_line_art_width(complete_line_art, threshold=line_art_snap_threshold)
    # When strokes carry measured ink widths, raise the adaptive baseline so
    # fallback strokes (no width) still look like brush marks, not hairlines.
    stroke_widths = [float(s.mean_width(default=0.0)) for s in strokes if s.mean_width(default=0.0) > 0]
    if stroke_widths:
        median_ink = float(sorted(stroke_widths)[len(stroke_widths) // 2])
        estimated_line_width = max(estimated_line_width, int(round(min(median_ink, 24.0))))
    effective_line_thickness = _resolve_line_thickness(line_thickness, estimated_line_width)
    reveal_width = max(effective_line_thickness * 3 + 2, int(round(estimated_line_width * 2.2)), 8)
    snap_ink_mask = (
        _line_art_ink_mask(complete_line_art, resolution, threshold=line_art_snap_threshold)
        if complete_line_art is not None and line_art_snap
        else None
    )
    ink_wash_render = os.getenv("WHITEBOARD_INKWASH_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    canvas_size = (resolution[0] * aa_scale, resolution[1] * aa_scale)
    canvas = _paper_background(canvas_size) if ink_wash_render else Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(canvas)
    exact_line_art_layer = (
        _exact_line_art_layer(_present_canvas(canvas, resolution), complete_line_art)
        if preserve_line_art_tones and complete_line_art is not None
        else None
    )
    reveal_mask = Image.new("L", resolution, 0) if complete_line_art is not None and line_art_snap else None
    reveal_draw = ImageDraw.Draw(reveal_mask) if reveal_mask is not None else None
    ink_spread_mask = Image.new("L", resolution, 0) if complete_line_art is not None and line_art_snap else None
    ink_spread_draw = ImageDraw.Draw(ink_spread_mask) if ink_spread_mask is not None else None

    total_frames = max(1, int(round(duration * fps)))
    draw_frames, color_frames, stamp_frames = _allocate_render_frames(
        total_frames,
        fps,
        tail_color_sec,
    )
    theme_strokes = _text_to_strokes(
        theme_text or "",
        resolution,
        position=theme_position,
        font_style=theme_font_style,
        font_size=theme_font_size,
    )
    requested_theme_frames = int(round(_theme_duration_seconds(theme_text) * max(1, fps)))
    theme_frames = min(requested_theme_frames, max(0, draw_frames - 1)) if theme_strokes else 0
    draw_frames -= theme_frames
    animated_entry = show_cursor and str(hand_style) in ANIMATED_CURSOR_DIRS
    entrance_frames = _animated_entrance_frame_count(draw_frames, fps) if animated_entry else 0
    stroke_draw_frames = max(1, draw_frames - entrance_frames)
    completion_frames = min(max(6, int(round(fps * 0.45))), max(1, stroke_draw_frames // 5))
    timeline = _prepare_timeline(strokes, stroke_draw_frames)
    if not timeline:
        raise ValueError("No drawable stroke segments")
    total_units = max(item.pause_end_unit for item in timeline)
    theme_timeline = _prepare_timeline(theme_strokes, max(1, theme_frames)) if theme_frames else []
    theme_total_units = max((item.pause_end_unit for item in theme_timeline), default=0.0)
    theme_progress = [0.0 for _ in theme_timeline]
    theme_canvas: Image.Image | None = None
    theme_finished_canvas: Image.Image | None = None
    theme_target_mask: Image.Image | None = None
    theme_reveal_mask: Image.Image | None = None
    theme_draw: ImageDraw.ImageDraw | None = None
    theme_cursor: tuple[float, float] | None = None
    theme_angle = 0.0
    stroke_progress = [0.0 for _ in timeline]
    first_stroke_point = timeline[0].stroke.points[0]
    cursor = first_stroke_point if animated_entry else (resolution[0] / 2, resolution[1] / 2)
    angle = 0.0
    angle_history: list[float] = []
    runner_distance = 0.0
    runner_last_cursor: tuple[float, float] | None = None
    color_runner_distance = 0.0
    color_runner_last_cursor: tuple[float, float] | None = None
    hand_cursor = None
    color_hand_cursor = None
    contour_fill_cache: ContourFillCache | None = None
    color_base_canvas: Image.Image | None = None
    doodle_strokes: list | None = None
    doodle_mask: Image.Image | None = None
    doodle_mask_draw: ImageDraw.ImageDraw | None = None
    doodle_last_idx = 0
    skin_color_mask = None
    color_finish_start = 0.94
    if source_image is not None and color_frames > 0:
        raw_skin_mask = _skin_tone_mask(source)
        skin_color_mask = np.asarray(
            Image.fromarray((raw_skin_mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.GaussianBlur(radius=1.2)),
            dtype=np.float32,
        ) / 255.0
        color_finish_start = _color_finish_start(source)
    tool_cursor_styles = {"procedural", "none", "quill", "brush", "rooster-quill", "real-eraser", *ANIMATED_CURSOR_DIRS, "ip-signature", "ip-stamp", "ip-spark", "ip-eraser"}
    if show_cursor and str(hand_style) not in tool_cursor_styles:
        hand_cursor = _load_hand_cursor(hand_style, resolution, hand_scale)
    if show_cursor and str(color_hand_style) not in tool_cursor_styles:
        color_hand_cursor = _load_hand_cursor(color_hand_style, resolution, color_hand_scale)
    if color_frames > 0 and color_fill_mode in ("doodle-fill", "quill-fill", "natural-repair", "lineart-gap-fill"):
        if color_fill_mode == "lineart-gap-fill" and complete_line_art is not None and source_image is not None:
            skin_strokes = _generate_skin_repair_strokes(source, density=max(3, color_fill_blocks))
            gap_strokes = _generate_lineart_gap_strokes(
                source, complete_line_art, resolution, density=max(3, color_fill_blocks)
            )
            doodle_strokes = _order_color_strokes([*skin_strokes, *gap_strokes], resolution)
        elif color_fill_mode == "lineart-gap-fill":
            doodle_strokes = _generate_doodle_fill_strokes(
                resolution[0], resolution[1], density=max(3, color_fill_blocks)
            )
        elif color_fill_mode in ("doodle-fill", "natural-repair"):
            if color_fill_mode in {"doodle-fill", "natural-repair"} and complete_line_art is not None:
                gap_strokes = _generate_lineart_gap_strokes(
                    source, complete_line_art, resolution, density=max(3, color_fill_blocks)
                )
                skin_strokes = _generate_skin_repair_strokes(source, density=max(3, color_fill_blocks))
                doodle_strokes = _order_color_strokes([*skin_strokes, *gap_strokes], resolution)
            elif color_fill_mode == "natural-repair" and source_image is not None:
                doodle_strokes = _generate_natural_repair_strokes(source, density=max(3, color_fill_blocks))
            else:
                doodle_strokes = _generate_doodle_fill_strokes(
                    resolution[0], resolution[1], density=max(3, color_fill_blocks)
                )
        else:
            doodle_strokes = _generate_quill_fill_strokes(
                resolution[0], resolution[1], density=max(3, color_fill_blocks)
            )
        doodle_mask = Image.new("L", resolution, 0)
        doodle_mask_draw = ImageDraw.Draw(doodle_mask)
    process, ffmpeg_command = _open_ffmpeg_rawvideo_writer(out_path, resolution, fps)
    try:
        for frame_index in range(total_frames):
            if frame_index < draw_frames:
                local_draw_frame = frame_index - entrance_frames
                target = total_units * max(
                    0.0,
                    min(1.0, (local_draw_frame + 1) / stroke_draw_frames),
                )
                for stroke_index, item in enumerate(timeline):
                    if target <= item.start_unit:
                        break
                    if target >= item.end_unit:
                        desired = item.length
                    else:
                        local = (target - item.start_unit) / max(1e-6, item.end_unit - item.start_unit)
                        desired = item.length * ease_in_out_sine(local)
                    previous = stroke_progress[stroke_index]
                    if desired > previous:
                        pts, seg_widths = _stroke_segment_between(
                            item.stroke.points,
                            item.cumulative,
                            previous,
                            desired,
                            widths=item.stroke.widths,
                        )
                        if len(pts) >= 2:
                            progress_ratio = desired / max(item.length, 1e-6)
                            stroke_base = _stroke_base_width(item.stroke, effective_line_thickness)
                            # Explicit CLI override still wins for thin strokes without measured widths.
                            if item.stroke.widths is None and item.stroke.width is None and line_thickness is not None and int(line_thickness) > 0:
                                stroke_base = float(effective_line_thickness)
                            width = _line_width(stroke_base, progress_ratio, aa_scale)
                            # The traced skeleton is only a timeline/cursor
                            # guide. When the extracted raster is available,
                            # reveal that exact ink body instead of painting a
                            # synthetic centerline that looks like a bone.
                            raw_fill_stroke = str(item.stroke.source).startswith("ink-freehand-fill")
                            if not (line_art_snap and complete_line_art is not None) or raw_fill_stroke:
                                _draw_stroke_segment(draw, pts, item.stroke.color, width, aa_scale, widths=seg_widths)
                            if raw_fill_stroke and ink_spread_draw is not None:
                                # A wider, soft low-opacity pass simulates wet
                                # ink spreading from a pressed brush.
                                spread_width = max(2, int(round(width * 1.45 + 2)))
                                ink_spread_draw.line(pts, fill=150, width=spread_width, joint="curve")
                            if reveal_draw is not None:
                                # Reveal should cover the painted ink body, not a thin skeleton halo.
                                local_reveal = max(reveal_width, int(round(stroke_base * 2.4 + 2)))
                                _draw_reveal_mask_segment(reveal_draw, pts, local_reveal, widths=seg_widths)
                            cursor = pts[-1]
                            angle = _smooth_angle(angle_history, segment_angle(pts[-2], pts[-1]))
                        stroke_progress[stroke_index] = desired
                frame = _present_canvas(canvas, resolution)
                if ink_spread_mask is not None:
                    soft_spread = ink_spread_mask.filter(ImageFilter.GaussianBlur(radius=2.0))
                    spread_alpha = np.asarray(soft_spread, dtype=np.float32) / 255.0 * 0.22
                    frame_arr = np.asarray(frame.convert("RGB"), dtype=np.float32)
                    spread_color = np.asarray((45, 40, 42), dtype=np.float32)
                    frame_arr = frame_arr * (1.0 - spread_alpha[..., None]) + spread_color * spread_alpha[..., None]
                    frame = Image.fromarray(np.clip(frame_arr, 0, 255).astype(np.uint8), mode="RGB")
                if source_image is not None and color_during_drawing and color_fill_mode not in {"natural-repair", "doodle-fill", "lineart-gap-fill"} and reveal_mask is not None:
                    frame = _reveal_source_along_strokes(frame, source, reveal_mask)
                frame = _reveal_line_art_canvas(
                    frame, complete_line_art, reveal_mask,
                    threshold=line_art_snap_threshold,
                    ink_mask=snap_ink_mask,
                    preserve_tones=preserve_line_art_tones,
                    exact_layer=exact_line_art_layer,
                )
                if line_art_snap:
                    completion_start = max(0, draw_frames - completion_frames)
                    if frame_index >= completion_start:
                        completion = (frame_index - completion_start + 1) / max(1, completion_frames)
                        frame = _complete_line_art_canvas(
                            frame, complete_line_art,
                            threshold=line_art_snap_threshold,
                            alpha=completion,
                            ink_mask=snap_ink_mask,
                            preserve_tones=preserve_line_art_tones,
                            exact_layer=exact_line_art_layer,
                        )
                if show_cursor:
                    if hand_cursor is not None:
                        frame = _paste_hand_cursor(frame, cursor[0], cursor[1], hand_cursor)
                    elif hand_style in ANIMATED_CURSOR_DIRS:
                        if frame_index < entrance_frames:
                            raw_entry_progress = (frame_index + 1) / max(1, entrance_frames)
                            entry_progress = ease_in_out_sine(raw_entry_progress)
                            entrance_size = _animated_character_side(resolution, stage="entrance")
                            creation_size = _animated_character_side(resolution, stage="creation")
                            enters_from_left = first_stroke_point[0] <= resolution[0] / 2
                            entry_start_x = (
                                -entrance_size * 0.18
                                if enters_from_left
                                else resolution[0] + entrance_size * 0.18
                            )
                            entry_cursor = (
                                entry_start_x + (first_stroke_point[0] - entry_start_x) * entry_progress,
                                first_stroke_point[1],
                            )
                            angle = 0.0 if enters_from_left else math.pi
                            shrink_progress = ease_in_out_sine(max(
                                0.0,
                                min(1.0, (raw_entry_progress - 0.72) / 0.28),
                            ))
                            size_override = int(round(
                                entrance_size + (creation_size - entrance_size) * shrink_progress
                            ))
                            cursor = entry_cursor
                        else:
                            size_override = None
                        if runner_last_cursor is not None:
                            runner_distance += math.hypot(
                                cursor[0] - runner_last_cursor[0],
                                cursor[1] - runner_last_cursor[1],
                            )
                        if frame_index < entrance_frames:
                            pose_stride = max(8.0, min(resolution) * 0.035)
                            runner_distance = max(
                                runner_distance,
                                (frame_index + 1) * pose_stride * 0.42,
                            )
                        runner_last_cursor = cursor
                        frame = _paste_animated_runner(
                            frame,
                            cursor[0],
                            cursor[1],
                            angle,
                            runner_distance,
                            str(hand_style),
                            stage="entrance" if frame_index < entrance_frames else "creation",
                            size_override=size_override,
                        )
                    elif hand_style == "real-eraser":
                        frame = _paste_real_eraser(frame, cursor[0], cursor[1], angle)
                    elif hand_style != "none":
                        _draw_cursor(frame, cursor[0], cursor[1], angle, style=str(hand_style))
            elif frame_index < draw_frames + color_frames:
                progress = (frame_index - draw_frames + 1) / max(1, color_frames)
                if color_base_canvas is None:
                    color_base_canvas = _present_canvas(canvas, resolution)
                    if source_image is not None and color_during_drawing and color_fill_mode not in {"natural-repair", "doodle-fill", "lineart-gap-fill"} and reveal_mask is not None:
                        color_base_canvas = _reveal_source_along_strokes(color_base_canvas, source, reveal_mask)
                    color_base_canvas = _reveal_line_art_canvas(
                        color_base_canvas,
                        complete_line_art,
                        reveal_mask,
                        threshold=line_art_snap_threshold,
                        ink_mask=snap_ink_mask,
                        preserve_tones=preserve_line_art_tones,
                        exact_layer=exact_line_art_layer,
                    )
                    if line_art_snap:
                        color_base_canvas = _complete_line_art_canvas(
                            color_base_canvas,
                            complete_line_art,
                            threshold=line_art_snap_threshold,
                            ink_mask=snap_ink_mask,
                            preserve_tones=preserve_line_art_tones,
                            exact_layer=exact_line_art_layer,
                        )
                color_canvas = color_base_canvas
                if color_fill_mode == "contour-wipe" and contour_fill_cache is None:
                    contour_fill_cache = _prepare_contour_fill_cache(color_canvas)
                if color_fill_mode in ("doodle-fill", "quill-fill", "natural-repair", "lineart-gap-fill") and doodle_strokes is not None:
                    frame, fill_cursor, fill_angle, doodle_last_idx = _doodle_fill_frame(
                        color_canvas,
                        source,
                        progress,
                        doodle_strokes,
                        doodle_mask,
                        doodle_mask_draw,
                        doodle_last_idx,
                        global_gradient=(color_fill_mode in {"natural-repair", "doodle-fill", "lineart-gap-fill"}),
                        skin_mask=skin_color_mask,
                        finish_start=color_finish_start,
                    )
                else:
                    frame, fill_cursor, fill_angle = _color_fill_frame(
                        color_canvas,
                        source,
                        progress,
                        mode=color_fill_mode,
                        blocks=color_fill_blocks,
                        contour_cache=contour_fill_cache,
                    )
                if show_cursor and fill_cursor is not None:
                    if color_hand_cursor is not None:
                        frame = _paste_hand_cursor(frame, fill_cursor[0], fill_cursor[1], color_hand_cursor)
                    elif color_hand_style in ANIMATED_CURSOR_DIRS:
                        if color_runner_last_cursor is not None:
                            color_runner_distance += math.hypot(
                                fill_cursor[0] - color_runner_last_cursor[0],
                                fill_cursor[1] - color_runner_last_cursor[1],
                            )
                        color_runner_last_cursor = fill_cursor
                        frame = _paste_animated_runner(
                            frame,
                            fill_cursor[0],
                            fill_cursor[1],
                            fill_angle,
                            color_runner_distance,
                            str(color_hand_style),
                        )
                    elif color_hand_style == "real-eraser":
                        frame = _paste_real_eraser(frame, fill_cursor[0], fill_cursor[1], fill_angle)
                    elif color_hand_style != "none":
                        _draw_cursor(frame, fill_cursor[0], fill_cursor[1], fill_angle, style=str(color_hand_style))
            elif frame_index < draw_frames + color_frames + theme_frames:
                if theme_canvas is None:
                    if source_image is not None:
                        theme_base = source.copy()
                    else:
                        theme_base = _complete_line_art_canvas(
                            _present_canvas(canvas, resolution),
                            complete_line_art,
                            threshold=line_art_snap_threshold,
                            ink_mask=snap_ink_mask,
                            preserve_tones=preserve_line_art_tones,
                            exact_layer=exact_line_art_layer,
                        )
                    theme_canvas = theme_base.resize(
                        (
                            resolution[0] * THEME_AA_SCALE,
                            resolution[1] * THEME_AA_SCALE,
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    theme_target_mask = _theme_text_mask(
                        theme_text or "",
                        resolution,
                        theme_font_style,
                        theme_font_size,
                        position=theme_position,
                        scale=THEME_AA_SCALE,
                    )
                    theme_reveal_mask = Image.new("L", theme_canvas.size, 0)
                    theme_draw = ImageDraw.Draw(theme_reveal_mask)
                    theme_finished_canvas = theme_canvas.copy()
                    theme_finished_canvas.paste(
                        (18, 16, 16),
                        (0, 0, theme_finished_canvas.width, theme_finished_canvas.height),
                        theme_target_mask,
                    )
                local_theme_frame = frame_index - draw_frames - color_frames
                target = theme_total_units * (local_theme_frame + 1) / max(1, theme_frames)
                for stroke_index, item in enumerate(theme_timeline):
                    if target <= item.start_unit:
                        break
                    desired = item.length if target >= item.end_unit else item.length * ease_in_out_sine(
                        (target - item.start_unit) / max(1e-6, item.end_unit - item.start_unit)
                    )
                    previous = theme_progress[stroke_index]
                    if desired <= previous:
                        continue
                    pts, seg_widths = _stroke_segment_between(
                        item.stroke.points,
                        item.cumulative,
                        previous,
                        desired,
                        widths=item.stroke.widths,
                    )
                    if len(pts) >= 2 and theme_draw is not None:
                        progress_ratio = desired / max(item.length, 1e-6)
                        stroke_base = _stroke_base_width(item.stroke, effective_line_thickness)
                        width = _line_width(stroke_base, progress_ratio, THEME_AA_SCALE)
                        _draw_stroke_segment(
                            theme_draw,
                            pts,
                            255,
                            max(width * 2, 8 * THEME_AA_SCALE),
                            THEME_AA_SCALE,
                            widths=seg_widths,
                        )
                        theme_cursor = pts[-1]
                        theme_angle = segment_angle(pts[-2], pts[-1])
                    theme_progress[stroke_index] = desired
                theme_frame_canvas = theme_canvas.copy()
                if local_theme_frame + 1 >= theme_frames and theme_finished_canvas is not None:
                    theme_frame_canvas = theme_finished_canvas.copy()
                elif theme_target_mask is not None and theme_reveal_mask is not None:
                    visible_ink = Image.new("L", theme_target_mask.size, 0)
                    visible_ink = Image.composite(theme_target_mask, visible_ink, theme_reveal_mask)
                    theme_frame_canvas.paste(
                        (18, 16, 16),
                        (0, 0, theme_frame_canvas.width, theme_frame_canvas.height),
                        visible_ink,
                    )
                frame = _present_canvas(theme_frame_canvas, resolution)
                if show_cursor and theme_cursor is not None:
                    if hand_cursor is not None:
                        frame = _paste_hand_cursor(frame, theme_cursor[0], theme_cursor[1], hand_cursor)
                    elif hand_style in ANIMATED_CURSOR_DIRS:
                        if runner_last_cursor is not None:
                            runner_distance += math.hypot(
                                theme_cursor[0] - runner_last_cursor[0],
                                theme_cursor[1] - runner_last_cursor[1],
                            )
                        runner_last_cursor = theme_cursor
                        frame = _paste_animated_runner(
                            frame,
                            theme_cursor[0],
                            theme_cursor[1],
                            theme_angle,
                            runner_distance,
                            str(hand_style),
                            stage="creation",
                        )
                    elif hand_style == "real-eraser":
                        frame = _paste_real_eraser(frame, theme_cursor[0], theme_cursor[1], theme_angle)
                    elif hand_style != "none":
                        _draw_cursor(frame, theme_cursor[0], theme_cursor[1], theme_angle, style=str(hand_style))
            else:
                stamp_progress = (
                    frame_index - draw_frames - color_frames - theme_frames + 1
                ) / max(1, stamp_frames)
                if theme_finished_canvas is not None:
                    frame = _present_canvas(theme_finished_canvas, resolution)
                elif theme_canvas is not None:
                    frame = _present_canvas(theme_canvas, resolution)
                elif source_image is not None:
                    # Color mode ends on the original image. Re-applying the
                    # extracted line art here would make the final frame look
                    # like the line-art preview instead of the source image.
                    frame = source.copy()
                else:
                    frame = _complete_line_art_canvas(
                        _present_canvas(canvas, resolution),
                        complete_line_art,
                        threshold=line_art_snap_threshold,
                        ink_mask=snap_ink_mask,
                        preserve_tones=preserve_line_art_tones,
                        exact_layer=exact_line_art_layer,
                    )
                frame = _apply_red_seal(
                    frame,
                    stamp_progress,
                    text=seal_text,
                    style=seal_style,
                    position=seal_position,
                )
            _write_ffmpeg_frame(process, frame)
    except Exception:
        if process.stdin is not None:
            process.stdin.close()
        process.wait()
        raise
    if process.stdin is not None:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, ffmpeg_command)


def render_image(
    image_path: Path,
    out_path: Path,
    duration: float = 8.0,
    fps: int = 60,
    resolution: tuple[int, int] = (1920, 1080),
    tail_color_sec: float = 2.0,
    source_image_path: Path | None = None,
    source_fit: str = "blur-fill",
    # Use the contour-aware repair flow by default for both standard and
    # ink-wash line art so their color reveal behaves consistently.
    color_fill_mode: str = "natural-repair",
    color_fill_blocks: int = 18,
    hand_style: str | Path = DEFAULT_DRAWING_TOOL,
    hand_scale: float = 1.0,
    color_hand_style: str | Path = "brush",
    color_hand_scale: float = 1.0,
    draw_text: str | None = None,
    draw_text_position: str = "bottom",
    theme_text: str | None = None,
    theme_font_style: str = "mao",
    theme_font_size: int = 72,
    theme_position: str = "right",
    seal_style: str = "white-text",
    seal_text: str = "老林涂鸦",
    seal_position: str = "left-center",
    line_art_snap: bool = True,
    line_art_snap_threshold: int = DEFAULT_LINE_ART_SNAP_THRESHOLD,
    line_thickness: int | None = 0,
    stroke_detail: str = "max",
    disable_hatching: bool = False,
    draw_mode: str = "direct-ink",
    ink_darkness: int = 90,
    ink_brush: float = 5.5,
) -> None:
    """Extract strokes from one image and render a scene MP4.

    draw_mode:
      - direct-ink: direct ink ordering while preserving extracted geometry (default)
      - structure-then-ink: outline/skeleton first, then ink fill

    ink_darkness: 0–100 (0=white, 100=pure black, default 90)
    ink_brush: max freehand stroke width in pixels (default 5.5)
    """

    source_image = None
    complete_line_art = None
    if image_path.suffix.lower() == ".svg":
        strokes, source_image = svg_to_strokes(image_path, resolution)
        complete_line_art = source_image
    else:
        strokes = to_strokes(
            image_path,
            resolution,
            stroke_detail=stroke_detail,
            disable_hatching=disable_hatching,
            draw_mode=draw_mode,
            ink_darkness=ink_darkness,
            ink_brush=ink_brush,
        )
        complete_line_art = _line_art_canvas(image_path, resolution)
    if draw_text:
        strokes.extend(_text_to_strokes(draw_text, resolution, draw_text_position))
    if source_image_path:
        source_image = _load_source_image_canvas(source_image_path, resolution, source_fit)

    # Duration policy:
    # - Default: honor the user-requested duration exactly (UI slider / --duration).
    # - Optional stretch: set WHITEBOARD_AUTO_EXTEND_DURATION=1 to grow the video when
    #   there is too much ink for a comfortable drawing speed (~50px/frame).
    # - Always clamp tail color so it cannot swallow the whole clip.
    import math
    import os

    requested = max(0.5, float(duration))
    tail = max(0.0, float(tail_color_sec))
    # Leave at least ~40% of the clip for drawing when color fill is enabled.
    if tail > 0:
        tail = min(tail, max(0.0, requested * 0.45))

    auto_extend = os.getenv("WHITEBOARD_AUTO_EXTEND_DURATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    theme_duration = _theme_duration_seconds(theme_text)
    effective_duration = requested + theme_duration
    if auto_extend and strokes:
        total_arc = 0.0
        for s in strokes:
            pts = s.points
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                dx = pts[i][0] - pts[i - 1][0]
                dy = pts[i][1] - pts[i - 1][1]
                total_arc += math.hypot(dx, dy)
        target_px_per_frame = float(os.getenv("WHITEBOARD_TARGET_PX_PER_FRAME", "80") or 80)
        draw_frames_needed = total_arc / max(1.0, target_px_per_frame)
        min_draw_sec = draw_frames_needed / max(1, fps)
        min_duration = min_draw_sec + tail + theme_duration + 0.25
        effective_duration = max(requested, min_duration)

    render_scene(
        image_path,
        strokes,
        effective_duration,
        out_path,
        fps=fps,
        resolution=resolution,
        tail_color_sec=tail,
        source_image=source_image,
        show_cursor=(hand_style != "none" or color_hand_style != "none"),
        hand_style=hand_style,
        hand_scale=hand_scale,
        color_hand_style=color_hand_style,
        color_hand_scale=color_hand_scale,
        line_thickness=line_thickness,
        color_fill_mode=color_fill_mode,
        color_fill_blocks=color_fill_blocks,
        theme_text=theme_text,
        theme_font_style=theme_font_style,
        theme_font_size=theme_font_size,
        theme_position=theme_position,
        seal_style=seal_style,
        seal_text=seal_text,
        seal_position=seal_position,
        complete_line_art=complete_line_art,
        line_art_snap=line_art_snap,
        line_art_snap_threshold=line_art_snap_threshold,
        draw_mode=draw_mode,
        preserve_line_art_tones=True,
        # Ordinary and passthrough line art must be completed exactly before
        # any source color is allowed onto the canvas. Ink-wash keeps its
        # intentional wet color/ink reveal during drawing.
        color_during_drawing=os.getenv("WHITEBOARD_INKWASH_MODE", "").strip().lower() in {"1", "true", "yes", "on"},
    )
