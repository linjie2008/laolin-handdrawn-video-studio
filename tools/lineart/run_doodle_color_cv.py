#!/usr/bin/env python3
"""Color-aware doodle extractor built on the ink-wash structure model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from run_inkwash_cv import (
    clean_mask,
    compose_output,
    morphological_close,
    normalize_to_white_paper,
    structure_and_wash_masks,
    suppress_border,
)


def _load_rgb(path: Path, max_size: int) -> tuple[Image.Image, tuple[int, int]]:
    raw = Image.open(path)
    if raw.mode in {"RGBA", "LA"} or "transparency" in raw.info:
        rgba = raw.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert("RGB")
    else:
        image = raw.convert("RGB")
    original_size = image.size
    if max_size > 0:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image, original_size


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Keep dark marks and near-white pastel colors without selecting paper."""

    arr = rgb.astype(np.float32)
    distance_from_white = np.sqrt(np.sum((255.0 - arr) ** 2, axis=2))
    chroma = arr.max(axis=2) - arr.min(axis=2)
    luminance = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    foreground = (distance_from_white >= 13.0) | (luminance < 244.0) | ((chroma >= 8.0) & (distance_from_white >= 8.0))
    # Remove isolated compression noise while keeping narrow colored pen marks.
    softened = Image.fromarray((foreground.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MedianFilter(3))
    return np.asarray(softened, dtype=np.uint8) > 0


def _quantized_rgb(image: Image.Image, colors: int = 64) -> np.ndarray:
    # Smooth paper grain and watercolor texture before region segmentation;
    # the original pixels remain untouched in the color reference output.
    region_image = image.filter(ImageFilter.GaussianBlur(radius=1.15))
    quantized = region_image.quantize(colors=max(8, min(128, colors)), method=Image.Quantize.MEDIANCUT)
    return np.asarray(quantized.convert("RGB"), dtype=np.uint8)


def _color_boundaries(quantized: np.ndarray, foreground: np.ndarray, threshold: float = 36.0) -> np.ndarray:
    q = quantized.astype(np.float32)
    edge = np.zeros(foreground.shape, dtype=bool)
    horizontal = np.sqrt(np.sum((q[:, 1:] - q[:, :-1]) ** 2, axis=2)) >= threshold
    horizontal &= foreground[:, 1:] | foreground[:, :-1]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    vertical = np.sqrt(np.sum((q[1:] - q[:-1]) ** 2, axis=2)) >= threshold
    vertical &= foreground[1:] | foreground[:-1]
    edge[1:] |= vertical
    edge[:-1] |= vertical
    # The foreground silhouette catches pale regions beside pure white paper.
    fg_img = Image.fromarray((foreground.astype(np.uint8) * 255), mode="L")
    expanded = np.asarray(fg_img.filter(ImageFilter.MaxFilter(3)), dtype=np.uint8) > 0
    contracted = np.asarray(fg_img.filter(ImageFilter.MinFilter(3)), dtype=np.uint8) > 0
    return edge | (expanded ^ contracted)


def _palette_payload(quantized: np.ndarray, foreground: np.ndarray, limit: int = 24) -> list[dict[str, object]]:
    pixels = quantized[foreground]
    if pixels.size == 0:
        return []
    colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = max(1, int(counts.sum()))
    payload = []
    for index in order[:limit]:
        rgb = [int(value) for value in colors[index]]
        payload.append(
            {
                "rgb": rgb,
                "hex": "#" + "".join(f"{value:02X}" for value in rgb),
                "coverage": round(float(counts[index]) / total, 5),
            }
        )
    return payload


def extract_doodle_colors(
    input_path: Path,
    output_path: Path,
    colors_output: Path | None = None,
    palette_output: Path | None = None,
) -> None:
    max_size = int(os.getenv("DOODLE_COLOR_SIZE", "0") or 0)
    image, original_size = _load_rgb(input_path, max_size)
    rgb = np.asarray(image, dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    normalized_gray = normalize_to_white_paper(gray)
    foreground = _foreground_mask(rgb)
    quantized = _quantized_rgb(image, colors=int(os.getenv("DOODLE_PALETTE_COLORS", "64") or 64))
    boundaries = _color_boundaries(quantized, foreground)

    bone, mid, wash = structure_and_wash_masks(
        normalized_gray,
        bone_delta=float(os.getenv("DOODLE_BONE_DELTA", "62") or 62),
        mid_delta=float(os.getenv("DOODLE_MID_DELTA", "24") or 24),
        pale_delta=float(os.getenv("DOODLE_PALE_DELTA", "10") or 10),
        grad_thresh=float(os.getenv("DOODLE_GRAD_THRESH", "5") or 5),
    )
    min_side = max(1, min(foreground.shape))
    min_area = max(2, int(round((min_side / 512.0) ** 2 * 4)))
    rgb_i16 = rgb.astype(np.int16)
    chroma = rgb_i16.max(axis=2) - rgb_i16.min(axis=2)
    neutral_ink = (gray <= 112) & ((chroma <= 58) | (gray <= 52))
    # Colored fills become region outlines; only genuinely dark/neutral marks
    # remain solid structure ink.
    bone = clean_mask(bone & neutral_ink, min_area=max(2, min_area // 2), min_elon=1.0, keep_large=True)
    dark_region_edges = boundaries & (gray <= 178)
    mid = clean_mask((mid & neutral_ink) | dark_region_edges, min_area=min_area, min_elon=1.0, keep_large=True) & ~bone
    structure = morphological_close(bone | mid, 1)
    bone = structure & bone
    mid = structure & ~bone
    lineart = compose_output(bone, mid, np.zeros_like(wash), wash_level=205)
    lineart = suppress_border(lineart)
    line_image = Image.fromarray(lineart, mode="L")

    color_reference = np.full_like(rgb, 255)
    color_reference[foreground] = rgb[foreground]
    color_image = Image.fromarray(color_reference, mode="RGB")

    if max_size > 0 and line_image.size != original_size:
        line_image = line_image.resize(original_size, Image.Resampling.LANCZOS)
        color_image = color_image.resize(original_size, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_image.convert("RGB").save(output_path)
    if colors_output is not None:
        colors_output.parent.mkdir(parents=True, exist_ok=True)
        color_image.save(colors_output)
    if palette_output is not None:
        palette_output.parent.mkdir(parents=True, exist_ok=True)
        palette_output.write_text(
            json.dumps(_palette_payload(quantized, foreground), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="涂鸦作品颜色与结构提取器")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colors-output", type=Path)
    parser.add_argument("--palette-output", type=Path)
    args = parser.parse_args()
    colors_output = args.colors_output or (Path(os.environ["DOODLE_COLOR_OUTPUT"]) if os.getenv("DOODLE_COLOR_OUTPUT") else None)
    palette_output = args.palette_output or (Path(os.environ["DOODLE_PALETTE_OUTPUT"]) if os.getenv("DOODLE_PALETTE_OUTPUT") else None)
    extract_doodle_colors(args.input.resolve(), args.output.resolve(), colors_output, palette_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
