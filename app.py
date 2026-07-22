import os
import glob
import hashlib
import json
import subprocess
import shutil
import tempfile
import threading
import time
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFilter

from whiteboard_skill.ui_state import DEFAULT_DRAWING_TOOL, matching_color_tool
from whiteboard_skill.whiteboard import _theme_preview_image

# Temporary path for storing the extracted lineart preview
LINEART_PREVIEW_PATH = os.path.abspath("out/web_output-lineart.png")
LINEART_PREVIEW_SVG_PATH = os.path.abspath("out/web_output-lineart.svg")
COLOR_LINEART_PREVIEW_PATH = os.path.abspath("out/web_output-color-lineart.png")
DOODLE_PALETTE_PATH = os.path.abspath("out/web_output-doodle-palette.json")
_EXTRACTION_LOCK = threading.Lock()
_UPLOAD_BG_LOCK = threading.Lock()
_PROCESSED_UPLOADS = {}


def _cleanup_out_dir(keep=None):
    """Trim generated outputs so out/ doesn't grow without bound.

    Only touches timestamp-named render outputs (out/web_output-*.mp4) and the
    white-background upload cache (out/uploads/*.png). Preview files such as
    web_output-lineart.png / -color-lineart.png / -doodle-palette.json are not
    matched (different extensions), and hand-named demos (doodle-*.mp4, etc.)
    are left untouched.
    """
    if keep is None:
        try:
            keep = int(os.environ.get("WHITEBOARD_KEEP_OUTPUTS", "20"))
        except ValueError:
            keep = 20
    keep = max(0, keep)
    patterns = [
        os.path.join("out", "web_output-*.mp4"),
        os.path.join("out", "uploads", "*.png"),
    ]
    for pattern in patterns:
        files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
        files.sort(key=os.path.getmtime, reverse=True)
        for stale in files[keep:]:
            try:
                os.remove(stale)
            except OSError:
                pass


def render_theme_font_preview(text, font_style, font_size):
    return np.asarray(_theme_preview_image(text, font_style or "mao", int(font_size or 72)))


def _raw_uploaded_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", None)


def _uploaded_path(file_obj):
    path = _raw_uploaded_path(file_obj)
    return _PROCESSED_UPLOADS.get(path, path)


def _prepare_white_background(file_obj):
    """Replace a non-white edge-connected background with pure white."""

    raw_path = _raw_uploaded_path(file_obj)
    if not raw_path or not os.path.exists(raw_path) or raw_path.lower().endswith(".svg"):
        return raw_path
    with _UPLOAD_BG_LOCK:
        cached = _PROCESSED_UPLOADS.get(raw_path)
        if cached and os.path.exists(cached):
            return cached

        source = Image.open(raw_path)
        if source.mode in {"RGBA", "LA"} or "transparency" in source.info:
            rgba = source.convert("RGBA")
            source = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        else:
            source = source.convert("RGB")

        sample = source.copy()
        sample.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        arr = np.asarray(sample, dtype=np.uint8)
        border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]), axis=0)
        border_gray = border.mean(axis=1)
        border_chroma = border.max(axis=1) - border.min(axis=1)
        already_white = float(np.mean((border_gray >= 246) & (border_chroma <= 10))) >= 0.88
        if already_white:
            _PROCESSED_UPLOADS[raw_path] = raw_path
            return raw_path

        median = np.median(border.astype(np.float32), axis=0)
        marker = (1, 254, 253)
        flood = sample.copy()
        width, height = sample.size
        seeds = []
        for step in range(13):
            x = int(round((width - 1) * step / 12))
            y = int(round((height - 1) * step / 12))
            seeds.extend(((x, 0), (x, height - 1), (0, y), (width - 1, y)))
        for seed in seeds:
            pixel = np.asarray(flood.getpixel(seed), dtype=np.float32)
            if tuple(int(v) for v in pixel) == marker:
                continue
            if float(np.linalg.norm(pixel - median)) <= 48.0:
                ImageDraw.floodfill(flood, seed, marker, thresh=34)

        flood_arr = np.asarray(flood, dtype=np.uint8)
        background = np.all(flood_arr == np.asarray(marker, dtype=np.uint8), axis=2)
        if not np.any(background):
            _PROCESSED_UPLOADS[raw_path] = raw_path
            return raw_path
        mask = Image.fromarray(background.astype(np.uint8) * 255, mode="L")
        if mask.size != source.size:
            mask = mask.resize(source.size, Image.Resampling.BILINEAR)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(0.6, min(source.size) / 1800.0)))
        prepared = Image.composite(Image.new("RGB", source.size, "white"), source, mask)

        os.makedirs("out/uploads", exist_ok=True)
        identity = f"{raw_path}:{os.path.getmtime(raw_path)}:{os.path.getsize(raw_path)}".encode("utf-8")
        output_path = os.path.abspath(f"out/uploads/white-bg-{hashlib.sha1(identity).hexdigest()[:16]}.png")
        prepared.save(output_path, format="PNG")
        _PROCESSED_UPLOADS[raw_path] = output_path
        _PROCESSED_UPLOADS[output_path] = output_path
        return output_path


def _suggest_color_duration(file_obj, duration):
    """Return the visible color phase duration for the current source image."""

    try:
        path = _uploaded_path(file_obj)
        total = max(5.0, float(duration or 15))
        if not path or not os.path.exists(path):
            return round(min(4.0, total * 0.30), 1)
        sample = Image.open(path).convert("RGB").resize((48, 48), Image.Resampling.BILINEAR)
        arr = np.asarray(sample, dtype=np.float32) / 255.0
        chroma_map = arr.max(axis=2) - arr.min(axis=2)
        gray = arr.mean(axis=2)
        paint_area = np.mean((gray < 0.96) | (chroma_map > 0.08))
        quantized = (arr * 5.0).astype(np.uint8)
        diversity = min(1.0, len(np.unique(quantized.reshape(-1, 3), axis=0)) / 90.0)
        horizontal = np.mean(np.any(quantized[:, 1:] != quantized[:, :-1], axis=2))
        vertical = np.mean(np.any(quantized[1:] != quantized[:-1], axis=2))
        boundaries = min(1.0, (horizontal + vertical) * 1.8)
        complexity = max(0.0, min(1.0, paint_area * 0.35 + diversity * 0.25 + boundaries * 0.40))
        base = min(4.0, total * 0.30)
        suggested = base * (0.70 + 1.30 * complexity)
        return round(min(suggested, total * 0.45), 1)
    except (OSError, ValueError, TypeError):
        return round(min(4.0, max(5.0, float(duration or 15)) * 0.30), 1)


def _progress_markup(value: float, label: str = "") -> str:
    """Render a compact progress bar whose color changes every 20%."""

    percent = max(0, min(100, int(round(value * 100))))
    colors = ("#ef4444", "#f97316", "#eab308", "#2563eb", "#16a34a")
    color = colors[min(4, percent // 20)]
    return (
        f"<div class='render-progress' aria-label='{percent}% {label}'>"
        f"<div class='render-progress-track'><div class='render-progress-fill' "
        f"style='width:{percent}%;background:{color}'></div></div>"
        f"<div class='render-progress-caption'><span>{label}</span><strong>{percent}%</strong></div></div>"
    )


def _lineart_mode(mode: str) -> bool:
    return mode == "lineart"


def _has_current_extraction(input_path):
    """Accept only a lineart file produced after the current upload."""

    if not input_path or not os.path.exists(LINEART_PREVIEW_PATH):
        return False
    try:
        return os.path.getmtime(LINEART_PREVIEW_PATH) >= os.path.getmtime(input_path)
    except OSError:
        return False


def _write_color_lineart_preview(source_path, lineart_path):
    """Keep source colors only where the extracted line-art has ink."""

    source = Image.open(source_path).convert("RGB")
    lineart = Image.open(lineart_path).convert("L")
    source = source.resize(lineart.size, Image.Resampling.LANCZOS)
    source_arr = np.asarray(source, dtype="uint8")
    gray = np.asarray(lineart, dtype="uint8")
    mask = gray < 235
    # Slightly widen antialiased strokes so the color is visible in preview.
    mask_img = Image.fromarray((mask.astype("uint8") * 255), mode="L").filter(ImageFilter.GaussianBlur(radius=1.2))
    alpha = np.asarray(mask_img, dtype="float32") / 255.0
    white = np.full_like(source_arr, 255)
    result = source_arr.astype("float32") * alpha[..., None] + white.astype("float32") * (1.0 - alpha[..., None])
    Image.fromarray(result.astype("uint8"), mode="RGB").save(COLOR_LINEART_PREVIEW_PATH)
    return COLOR_LINEART_PREVIEW_PATH


def _upscale_lineart_for_render(lineart_path, target_size):
    """Restore the extractor result to the upload working size with AA edges."""

    lineart = Image.open(lineart_path).convert("L")
    if lineart.size != target_size:
        lineart = lineart.resize(target_size, Image.Resampling.LANCZOS)
    lineart.save(lineart_path)

def extract_lineart_preview(
    file_obj,
    lineart_provider,
    bypass_ai,
    stroke_detail,
    inkwash_bone_delta,
    inkwash_mid_delta,
    inkwash_pale_delta,
    inkwash_grad_thresh,
    inkwash_min_area,
    inkwash_min_elon,
    inkwash_solid_thresh,
    inkwash_draw_thresh,
):
    if file_obj is None:
        return None, "请先上传一张图片！", False
        
    input_path = _uploaded_path(file_obj)
    if not input_path:
        return None, "上传文件无效。", False
    is_svg = input_path.lower().endswith(".svg")
    
    os.makedirs("out", exist_ok=True)
    
    if is_svg:
        # For SVG, we don't need AI extraction, but we copy it and show it in the preview box
        shutil.copy(input_path, LINEART_PREVIEW_SVG_PATH)
        return LINEART_PREVIEW_SVG_PATH, "SVG 线稿已就绪，可以生成视频。", True

    if bypass_ai:
        # "Already line art" is a true passthrough. Do not normalize,
        # threshold, resize, or otherwise reinterpret the uploaded pixels.
        source_lineart = Image.open(input_path)
        if source_lineart.mode in {"RGBA", "LA"} or "transparency" in source_lineart.info:
            rgba = source_lineart.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            source_lineart = Image.alpha_composite(white, rgba).convert("RGB")
        else:
            source_lineart = source_lineart.convert("RGB")
        with _EXTRACTION_LOCK:
            temp_preview = tempfile.NamedTemporaryFile(
                prefix="whiteboard-lineart-passthrough-",
                suffix=".png",
                dir=os.path.dirname(LINEART_PREVIEW_PATH),
                delete=False,
            )
            temp_preview_path = temp_preview.name
            temp_preview.close()
            try:
                source_lineart.save(temp_preview_path, format="PNG")
                os.replace(temp_preview_path, LINEART_PREVIEW_PATH)
            finally:
                if os.path.exists(temp_preview_path):
                    os.remove(temp_preview_path)
        _write_color_lineart_preview(input_path, LINEART_PREVIEW_PATH)
        return LINEART_PREVIEW_PATH, "原线稿已无损载入，可以直接生成视频。", True

    cli_path = os.path.join(os.path.dirname(__file__), ".venv-lineart/bin/whiteboard")

    # Use a unique temp file because upload-triggered extraction and a manual
    # retry can arrive close together.
    image = Image.open(input_path)
    # Keep enough edge detail for smooth ink contours; the old 1024 cap made
    # curves and fine brush edges visibly stair-stepped after rendering.
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    temp_handle = tempfile.NamedTemporaryFile(prefix="whiteboard-input-", suffix=".png", delete=False)
    temp_input = temp_handle.name
    temp_handle.close()
    image.save(temp_input)
    
    # Using AI lineart model
    cmd = [
        cli_path,
        "extract-lineart",
        temp_input,
        "-o", LINEART_PREVIEW_PATH,
        "--provider", lineart_provider
    ]
        
    env = os.environ.copy()
    if lineart_provider in ("ink-wash", "modern-ink"):
        env["WHITEBOARD_INKWASH_MODE"] = "1"
        env["INKWASH_BONE_DELTA"] = str(int(inkwash_bone_delta))
        env["INKWASH_DARK_THRESHOLD"] = str(int(inkwash_bone_delta))
        env["INKWASH_MID_DELTA"] = str(int(inkwash_mid_delta))
        env["INKWASH_MID_THRESHOLD"] = str(int(inkwash_mid_delta))
        env["INKWASH_PALE_DELTA"] = str(int(inkwash_pale_delta))
        env["INKWASH_PALE_THRESHOLD"] = str(int(inkwash_pale_delta))
        env["INKWASH_GRAD_THRESH"] = str(int(inkwash_grad_thresh))
        env["INKWASH_MIN_AREA"] = str(int(inkwash_min_area))
        env["INKWASH_MIN_ELON"] = str(float(inkwash_min_elon))
        env["INKWASH_SOLID_THRESH"] = str(int(inkwash_solid_thresh))
        env["INKWASH_DRAW_THRESH"] = str(int(inkwash_draw_thresh))
    elif lineart_provider == "doodle-color":
        env["DOODLE_COLOR_OUTPUT"] = COLOR_LINEART_PREVIEW_PATH
        env["DOODLE_PALETTE_OUTPUT"] = DOODLE_PALETTE_PATH

    try:
        with _EXTRACTION_LOCK:
            if lineart_provider == "doodle-color":
                for stale_path in (COLOR_LINEART_PREVIEW_PATH, DOODLE_PALETTE_PATH):
                    if os.path.exists(stale_path):
                        os.remove(stale_path)
            result = None
            last_error = None
            for attempt in range(2):
                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                        env=env,
                    )
                    break
                except subprocess.CalledProcessError as exc:
                    last_error = exc
                    if attempt == 0:
                        time.sleep(0.35)
            if result is None and last_error is not None:
                raise last_error
        if os.path.exists(LINEART_PREVIEW_PATH):
            _upscale_lineart_for_render(LINEART_PREVIEW_PATH, image.size)
            if lineart_provider == "doodle-color" and os.path.exists(COLOR_LINEART_PREVIEW_PATH):
                color_preview = Image.open(COLOR_LINEART_PREVIEW_PATH).convert("RGB")
                if color_preview.size != image.size:
                    color_preview = color_preview.resize(image.size, Image.Resampling.LANCZOS)
                    color_preview.save(COLOR_LINEART_PREVIEW_PATH)
                color_count = 0
                if os.path.exists(DOODLE_PALETTE_PATH):
                    try:
                        with open(DOODLE_PALETTE_PATH, encoding="utf-8") as palette_file:
                            color_count = len(json.load(palette_file))
                    except (OSError, ValueError, TypeError):
                        color_count = 0
                count_text = f"，识别到 {color_count} 组主要颜色" if color_count else ""
                return COLOR_LINEART_PREVIEW_PATH, f"涂鸦结构与全部颜色区域识别成功{count_text}，可以生成视频。", True
            _write_color_lineart_preview(input_path, LINEART_PREVIEW_PATH)
            return LINEART_PREVIEW_PATH, "黑白线稿提取成功，颜色参考已同步，可以生成视频。", True
        else:
            return None, f"错误：线稿文件未生成。\n日志：\n{result.stderr}", False
    except subprocess.CalledProcessError as e:
        return None, f"线稿提取失败。\n错误详情：\n{e.stderr}", False
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)


def render_video_from_lineart(
    file_obj,
    extraction_ready,
    duration,
    fps,
    hand,
    theme_text,
    theme_font_style,
    theme_font_size,
    theme_position,
    seal_style,
    seal_text,
    seal_position,
    seal_x,
    seal_y,
    color_tool,
    color_fill,
    stroke_detail,
    line_thickness,
    lineart_provider,
    tail_color,
    color_enabled,
    resolution,
    bypass_ai,
    disable_hatching,
    draw_mode,
    ink_darkness,
    ink_brush,
    inkwash_bone_delta,
    inkwash_mid_delta,
    inkwash_pale_delta,
    inkwash_grad_thresh,
    inkwash_min_area,
    inkwash_min_elon,
    inkwash_solid_thresh,
    inkwash_draw_thresh,
):
    if file_obj is None:
        yield None, _progress_markup(0, "请先上传文件"), "请先上传文件！"
        return
        
    input_path = _uploaded_path(file_obj)
    if not input_path:
        yield None, _progress_markup(0, "上传文件无效"), "上传文件无效。"
        return
    if not extraction_ready and not _has_current_extraction(input_path):
        yield None, _progress_markup(0, "等待线稿"), "请先点击“提取线稿”，确认线稿预览成功后再生成视频。"
        return
    yield gr.skip(), _progress_markup(0.10, "已确认线稿，准备渲染"), "正在准备视频渲染参数。"
    is_svg = input_path.lower().endswith(".svg")
    
    use_original_size = (resolution == "original")
    width, height = 1920, 1080
    if not use_original_size:
        try:
            w_str, h_str = resolution.split("x")
            width = int(w_str)
            height = int(h_str)
        except ValueError:
            pass
        
    os.makedirs("out", exist_ok=True)
    # A unique path prevents the browser/video component from replaying a
    # cached result when consecutive renders use different line-art logic.
    output_video = os.path.abspath(f"out/web_output-{time.time_ns()}.mp4")
        
    cli_path = os.path.join(os.path.dirname(__file__), ".venv-lineart/bin/whiteboard")
    effective_tail_color = tail_color if color_enabled else 0.0
    # Standard line art uses the original-vs-line-art gap mask. Ink-wash keeps
    # its own contour repair and wet-spread behavior.
    effective_color_fill = (
        "lineart-gap-fill"
        if not is_svg and lineart_provider not in ("ink-wash", "modern-ink")
        else color_fill
    )
    
    if is_svg:
        # SVG doesn't use the extracted lineart file, it uses the original SVG file directly
        temp_input = "temp_input.svg"
        shutil.copy(input_path, temp_input)
        
        cmd = [
            cli_path,
            "render-image",
            temp_input,
            "-o", output_video,
            "--duration", str(int(duration)),
            "--fps", str(int(fps)),
            "--stroke-detail", stroke_detail,
            "--hand", hand,
            "--color-hand", COLOR_TOOL_CONFIG.get(color_tool, ("brush", "natural-repair"))[0],
            "--line-thickness", str(int(line_thickness)),
            "--tail-color", str(float(effective_tail_color)),
            "--color-fill", effective_color_fill,
            "--draw-mode", str(draw_mode or "direct-ink"),
            "--ink-darkness", str(int(ink_darkness)),
            "--ink-brush", str(float(ink_brush)),
        ]
        if use_original_size:
            cmd.extend(["--width", "1920", "--height", "1080"])
        else:
            cmd.extend(["--width", str(width), "--height", str(height)])
        if disable_hatching:
            cmd.append("--no-hatching")
    else:
        # For raster images, ensure we have the preview from Step 1
        if not os.path.exists(LINEART_PREVIEW_PATH):
            yield None, _progress_markup(0, "未找到线稿"), "错误：未找到已提取的线稿。请先点击左侧的『步骤 1：提取并预览线稿』并确认效果！"
            return
            
        temp_input = "temp_input_lineart.png"
        shutil.copy(LINEART_PREVIEW_PATH, temp_input)
        
        cmd = [
            cli_path,
            "render-image",
            temp_input,
            "-o", output_video,
            "--duration", str(int(duration)),
            "--fps", str(int(fps)),
            "--stroke-detail", stroke_detail,
            "--hand", hand,
            "--color-hand", COLOR_TOOL_CONFIG.get(color_tool, ("brush", "natural-repair"))[0],
            "--line-thickness", str(int(line_thickness)),
            "--tail-color", str(float(effective_tail_color)),
            "--color-fill", effective_color_fill,
            "--draw-mode", str(draw_mode or "direct-ink"),
            "--ink-darkness", str(int(ink_darkness)),
            "--ink-brush", str(float(ink_brush)),
        ]
        if use_original_size:
            cmd.append("--size-from-image")
        else:
            cmd.extend(["--width", str(width), "--height", str(height)])
            
        if color_enabled:
            cmd.extend(["--source-image", input_path])
        if disable_hatching:
            cmd.append("--no-hatching")

    cleaned_theme = "\n".join(
        cleaned_line
        for line in str(theme_text or "").splitlines()
        if (cleaned_line := " ".join(line.split()))
    )
    if cleaned_theme:
        cmd.extend([
            "--theme", cleaned_theme,
            "--theme-font", str(theme_font_style or "mao"),
            "--theme-font-size", str(int(theme_font_size or 72)),
            "--theme-position", str(theme_position or "right"),
        ])
    seal_pos = str(seal_position or "left-center")
    if seal_pos == "custom":
        seal_pos = f"{int(seal_x)},{int(seal_y)}"
    cmd.extend([
        "--seal-style", str(seal_style or "white-text"),
        "--seal-text", "".join(str(seal_text or "老林涂鸦").split()) or "老林涂鸦",
        "--seal-position", seal_pos,
    ])
        
    env = os.environ.copy()
    if lineart_provider in ("ink-wash", "modern-ink"):
        env["WHITEBOARD_INKWASH_MODE"] = "1"
        env["INKWASH_BONE_DELTA"] = str(int(inkwash_bone_delta))
        env["INKWASH_DARK_THRESHOLD"] = str(int(inkwash_bone_delta))
        env["INKWASH_MID_DELTA"] = str(int(inkwash_mid_delta))
        env["INKWASH_MID_THRESHOLD"] = str(int(inkwash_mid_delta))
        env["INKWASH_PALE_DELTA"] = str(int(inkwash_pale_delta))
        env["INKWASH_PALE_THRESHOLD"] = str(int(inkwash_pale_delta))
        env["INKWASH_GRAD_THRESH"] = str(int(inkwash_grad_thresh))
        env["INKWASH_MIN_AREA"] = str(int(inkwash_min_area))
        env["INKWASH_MIN_ELON"] = str(float(inkwash_min_elon))
        env["INKWASH_SOLID_THRESH"] = str(int(inkwash_solid_thresh))
        env["INKWASH_DRAW_THRESH"] = str(int(inkwash_draw_thresh))

    try:
        yield gr.skip(), _progress_markup(0.22, "开始绘制线稿"), "正在绘制线稿。"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        started = time.monotonic()
        theme_seconds = 0.0 if not cleaned_theme else min(4.0, 1.2 + 0.45 * len(cleaned_theme.replace(" ", "")))
        estimate = max(
            8.0,
            (float(duration or 15) + theme_seconds) * (1.25 if color_enabled else 0.90),
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            render_progress = min(0.88, 0.22 + 0.64 * (elapsed / estimate))
            # Only the progress bar changes during polling. Re-sending None
            # to Video clears and remounts its player; repeatedly updating the
            # status textbox also causes a visible flash in Gradio.
            yield gr.skip(), _progress_markup(render_progress, "正在绘制线稿与渐变上色"), gr.skip()
            time.sleep(0.35)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
        yield gr.skip(), _progress_markup(0.96, "正在封装视频"), "正在封装视频。"
        if os.path.exists(output_video):
            yield output_video, _progress_markup(1.0, "视频生成完成"), "步骤 2 完成：视频渲染成功！"
            return
        else:
            yield None, _progress_markup(0.96, "视频文件未生成"), "错误：视频文件未生成。"
            return
    except subprocess.CalledProcessError as e:
        yield None, _progress_markup(0, "渲染失败"), f"视频渲染失败。\n错误详情：\n{e.stderr}"
        return
    finally:
        if not is_svg and os.path.exists(temp_input):
            os.remove(temp_input)

INKWASH_PRESETS = {
    "balanced": (70, 32, 22, 6, 6, 1.25, 100, 90),
    "clean": (82, 38, 30, 8, 10, 1.45, 105, 90),
    "detail": (58, 24, 16, 4, 3, 1.10, 92, 105),
}

# Human coloring tools mapped to the renderer's corresponding stroke behavior.
COLOR_TOOL_CONFIG = {
    "soft-goat": ("brush", "natural-repair"),
    "hard-combination": ("brush", "doodle-fill"),
    "wide-brush": ("brush", "brush-scan"),
    "detail-wolf": ("brush", "brush-path"),
    "wet-ink": ("brush", "fade"),
    "quill": ("quill", "quill-fill"),
    "rooster-quill": ("rooster-quill", "natural-repair"),
    "wukong-run": ("wukong-run", "natural-repair"),
    "zhubajie-run": ("zhubajie-run", "natural-repair"),
    "tangsanzang-run": ("tangsanzang-run", "natural-repair"),
    "guanyu-run": ("guanyu-run", "natural-repair"),
    "zhugeliang-run": ("zhugeliang-run", "natural-repair"),
    "ip-signature": ("ip-signature", "brush-path"),
    "ip-stamp": ("ip-stamp", "top-down-blocks"),
    "ip-spark": ("ip-spark", "fade"),
    "eraser": ("real-eraser", "natural-repair"),
    "none": ("none", "natural-repair"),
}


def apply_inkwash_preset(name):
    values = INKWASH_PRESETS.get(name, INKWASH_PRESETS["balanced"])
    return [gr.update(value=value) for value in values]


theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["-apple-system", "BlinkMacSystemFont", "SF Pro Display", "Helvetica Neue", "sans-serif"],
)

APP_CSS = """
:root { --paper: #ffffff; --ink: #111827; --muted: #64748b; --line: #e5e7eb; --blue: #2563eb; --body-background-fill: #ffffff; --block-background-fill: #ffffff; }
* { box-sizing: border-box; }
html, body, #root, .gradio-container { background: #ffffff !important; color: var(--ink) !important; }
html, body, #root { overflow: hidden !important; scrollbar-width: none !important; }
html::-webkit-scrollbar, body::-webkit-scrollbar, #root::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
.app-shell { max-width: 1660px; margin: 0 auto; padding: 0 16px; }
.app-header { border-bottom: 1px solid var(--line); padding: 6px 0 9px; margin-bottom: 9px; background: #ffffff !important; }
.app-header h1, .app-header h2, .app-header h3 { color: var(--ink) !important; margin: 0; }
.brand-cn { color: #111827; font-size: 21px; font-weight: 700; letter-spacing: .02em; line-height: 1.1; }
.brand-en { color: #2563eb; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 9px; font-weight: 700; letter-spacing: .2em; margin-top: 3px; }
.panel { border: 1px solid rgba(210,210,215,.9); border-radius: 14px; background: rgba(255,255,255,.82); box-shadow: 0 6px 22px rgba(0,0,0,.05); padding: 11px; }
.preview-panel { min-height: 0; }
.preview-panel .gr-image { background: #f0f0f2; border-radius: 12px; }
.preview-panel .upload-zone { flex: 0 0 auto; }
.preview-panel .upload-zone .upload-container,
.preview-panel .upload-zone .image-container,
.preview-panel .upload-zone .wrap { min-height: 94px !important; }
.preview-panel .upload-zone .upload-text { line-height: 1.35 !important; white-space: normal !important; overflow: hidden !important; }
.preview-panel .upload-zone .upload-text p { margin: 2px 0 !important; }
.preview-panel .preview-pair { flex: 0 0 auto; gap: 8px; }
.preview-panel .video-output { flex: 1 1 auto; min-height: 0; }
.preview-panel .video-output video { max-height: 100%; object-fit: contain; }
.preview-panel .status-row { flex: 0 0 auto; }
.primary-action { min-height: 38px; font-weight: 650; background: var(--blue) !important; border-color: var(--blue) !important; }
.panel h3 { font-size: 15px; font-weight: 650; margin: 0 0 3px; }
.panel label span { color: var(--muted); font-size: 12px; }
.status textarea { font-family: ui-monospace, SFMono-Regular, monospace; overflow: hidden !important; resize: none !important; white-space: nowrap; }
.panel .gr-accordion { border-color: var(--line); border-radius: 10px; }
.panel input, .panel textarea, .panel button, .panel select { border-radius: 8px !important; }
.workspace, .preview-panel, .settings-panel { gap: 7px !important; }
.preview-panel .gr-markdown, .settings-panel .gr-markdown { margin: 0 !important; }
.preview-panel .status-row { margin-top: 0 !important; }
.status textarea { min-height: 30px !important; height: 30px !important; }
.render-progress { padding: 3px 1px 0; }
.render-progress-track { height: 6px; overflow: hidden; border-radius: 99px; background: #e5e7eb; }
.render-progress-fill { height: 100%; border-radius: 99px; transition: width .25s ease, background-color .25s ease; }
.render-progress-caption { display: flex; justify-content: space-between; color: var(--muted); font-size: 10px; line-height: 16px; }
.render-progress-caption strong { color: var(--ink); font-weight: 650; }
.settings-panel .gr-accordion { margin: 0 !important; }
.settings-panel > *, .settings-panel > .form > * { flex: 0 0 auto !important; }
.settings-panel {
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  overflow-x: hidden !important;
  scrollbar-width: none !important;
}
.settings-panel::-webkit-scrollbar { display: none !important; width: 0 !important; height: 0 !important; }
.settings-panel .form, .settings-panel .row { gap: 7px !important; }
.processing-mode { overflow: visible !important; }
.processing-mode > .wrap { display: flex !important; flex-wrap: wrap !important; overflow: visible !important; }
.processing-mode label { flex: 0 0 auto !important; }
@media (min-width: 901px) {
  html, body, #root, .gradio-container { height: 100%; overflow: hidden !important; scrollbar-width: none !important; }
  .app-shell { height: calc(100vh - 16px); overflow: hidden; }
  .workspace { height: calc(100vh - 91px); overflow: hidden; gap: 12px !important; }
  .preview-panel { height: 100%; overflow: hidden !important; }
  .settings-panel {
    height: calc(100vh - 91px) !important;
    max-height: calc(100vh - 91px) !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
  }
}
@media (max-width: 900px) { .preview-panel { min-height: auto; } }
"""

with gr.Blocks(title="老林手绘视频工坊") as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML(
            "<div class='app-header'><div class='brand-cn'>老林手绘视频工坊</div>"
            "<div class='brand-en'>LAOLIN HAND-DRAWN VIDEO STUDIO</div></div>"
        )

        with gr.Row(equal_height=False, elem_classes="workspace"):
            with gr.Column(scale=7, elem_classes="panel preview-panel"):
                input_file = gr.Image(
                    type="filepath",
                    # Upload keeps both drag-and-drop and the native file
                    # picker responsive. Clipboard listeners can delay the
                    # picker while the browser negotiates permissions.
                    sources=["upload"],
                    label="上传或粘贴图片",
                    height=112,
                    elem_classes="upload-zone",
                )
                with gr.Row():
                    extract_btn = gr.Button("提取线稿", variant="secondary")
                    # Keep the command clickable so the backend can explain a
                    # missing extraction instead of hiding the failure in a
                    # stale async UI state.
                    render_btn = gr.Button("生成视频", variant="primary", interactive=True, elem_classes="primary-action")
                with gr.Row(elem_classes="preview-pair"):
                    input_image_preview = gr.Image(label="原图", interactive=False, height=82)
                    lineart_preview = gr.Image(label="线稿 / 颜色识别", interactive=False, height=82)
                output_video_player = gr.Video(label="视频结果", height=245, elem_classes="video-output")
                with gr.Row(elem_classes="status-row"):
                    status_box = gr.Textbox(label="状态", interactive=False, lines=1, max_lines=1, elem_classes="status")
                progress_bar = gr.HTML(value=_progress_markup(0, "等待生成"), elem_classes="render-progress-host")
                extraction_ready = gr.State(False)
                color_tool_overridden = gr.State(False)

            with gr.Column(scale=5, elem_classes=["panel", "settings-panel"]):
                with gr.Accordion("线稿提取", open=True):
                    with gr.Row():
                        style_mode = gr.Radio(
                            choices=[("普通线稿", "standard"), ("水墨线稿", "ink-wash"), ("涂鸦作品", "doodle"), ("已经是线稿", "lineart")],
                            value="standard", label="处理类型", elem_classes="processing-mode"
                        )
                        lineart_provider = gr.Dropdown(
                            choices=[("动漫线稿", "anime2sketch"), ("自动选择", "auto"), ("物体轮廓", "informative"), ("水墨专用", "ink-wash"), ("涂鸦全色识别", "doodle-color")],
                            value="anime2sketch", label="线稿模型", visible=False
                        )
                    bypass_ai = gr.Checkbox(value=False, label="输入已经是线稿", visible=False)
                    inkwash_preset = gr.Dropdown(
                        choices=[("均衡", "balanced"), ("干净", "clean"), ("保留细节", "detail")],
                        value="balanced", label="水墨预设", visible=False
                    )

                with gr.Accordion("视频输出", open=True):
                    with gr.Row():
                        duration = gr.Number(value=15, minimum=5, maximum=120, precision=0, label="视频时长（秒）")
                        resolution = gr.Dropdown(
                            choices=[("原图尺寸", "original"), ("横屏 1920×1080", "1920x1080"), ("竖屏 1080×1920", "1080x1920"), ("横屏 1280×720", "1280x720")],
                            value="original", label="画布尺寸"
                        )
                        fps = gr.Slider(15, 60, value=20, step=5, label="帧率")
                    with gr.Row():
                        hand = gr.Dropdown(
                            choices=[
                                ("鹅毛笔", "quill"),
                                ("大公鸡羽毛笔", "rooster-quill"),
                                ("毛笔", "brush"),
                                ("孙悟空 · 筋斗云入场/奔跑绘制", "wukong-run"),
                                ("猪八戒 · 步行入场/挥耙绘制", "zhubajie-run"),
                                ("唐僧 · 步行入场/行走绘制", "tangsanzang-run"),
                                ("关羽 · 骑马入场/持刀绘制", "guanyu-run"),
                                ("诸葛亮 · 步行入场/羽扇绘制", "zhugeliang-run"),
                                ("IP 签名笔", "ip-signature"),
                                ("IP 印章笔", "ip-stamp"),
                                ("IP 星光笔", "ip-spark"),
                                ("不显示工具", "none"),
                            ],
                            value=DEFAULT_DRAWING_TOOL, label="绘制工具"
                        )
                    with gr.Row():
                        theme_text = gr.Textbox(
                            label="盖章前书写主题",
                            placeholder="例如：嫦娥（按 Enter 换行）",
                            lines=2,
                            max_lines=4,
                            scale=3,
                        )
                        theme_font_style = gr.Dropdown(
                            choices=[("毛体", "mao"), ("草书", "cursive")],
                            value="mao",
                            label="主题字体",
                            scale=1,
                        )
                        theme_position = gr.Dropdown(
                            choices=[
                                ("右侧竖排", "right"),
                                ("左侧竖排", "left"),
                                ("顶部横排", "top"),
                                ("底部横排", "bottom"),
                                ("画面居中", "center"),
                            ],
                            value="left",
                            label="主题位置",
                            scale=1,
                        )
                        theme_font_size = gr.Slider(
                            36,
                            160,
                            value=50,
                            step=2,
                            label="主题字体大小",
                            scale=2,
                        )
                    theme_font_preview = gr.Image(
                        value=render_theme_font_preview("嫦娥", "mao", 50),
                        label="字体预览",
                        interactive=False,
                        height=104,
                    )
                    with gr.Row():
                        seal_style = gr.Dropdown(
                            choices=[
                                ("复古朱印 · 玉玺", "vintage"),
                                ("水墨印章 · 飞白散墨", "inkwash"),
                                ("圆形印 · 朱底白文", "circle"),
                                ("椭圆印 · 朱底白文", "ellipse"),
                                ("白文印 · 阴文满朱", "white-text"),
                                ("无框随形印 · 朱字", "borderless"),
                            ],
                            value="white-text",
                            label="印章风格",
                            scale=1,
                        )
                        seal_text = gr.Textbox(
                            value="老林涂鸦",
                            label="印章内容",
                            max_lines=1,
                            scale=2,
                        )
                        seal_position = gr.Dropdown(
                            choices=[
                                ("左上", "left-top"),
                                ("左中", "left-center"),
                                ("左下", "left-bottom"),
                                ("中上", "center-top"),
                                ("正中", "center-center"),
                                ("中下", "center-bottom"),
                                ("右上", "right-top"),
                                ("右中", "right-center"),
                                ("右下", "right-bottom"),
                                ("自定义坐标…", "custom"),
                            ],
                            value="custom",
                            label="印章位置",
                            scale=1,
                        )
                    with gr.Row():
                        seal_x = gr.Slider(0, 100, value=8, step=1, label="印章 X%（选“自定义坐标”后生效）", visible=True)
                        seal_y = gr.Slider(0, 100, value=32, step=1, label="印章 Y%（选“自定义坐标”后生效）", visible=True)

                    def _toggle_seal_coords(pos):
                        vis = (pos == "custom")
                        return gr.update(visible=vis), gr.update(visible=vis)

                    seal_position.change(_toggle_seal_coords, inputs=[seal_position], outputs=[seal_x, seal_y])
                    with gr.Row():
                        color_enabled = gr.Checkbox(value=True, label="生成后添加彩色上色", scale=1)
                        color_tool = gr.Dropdown(
                            choices=[
                                ("羊毫笔 · 水墨晕染/大面积柔和上色", "soft-goat"),
                                ("兼毫笔 · 涂鸦/弹性铺色", "hard-combination"),
                                ("排笔 · 背景/大色块快速铺色", "wide-brush"),
                                ("小狼毫 · 简笔画/边缘细节补色", "detail-wolf"),
                                ("湿墨笔 · 水墨渐变/自然渗开", "wet-ink"),
                                ("鹅毛笔 · 细线/纹理排色", "quill"),
                                ("大公鸡羽毛笔 · 水墨/自然散墨", "rooster-quill"),
                                ("孙悟空 · 奔跑上色", "wukong-run"),
                                ("猪八戒 · 奔跑上色", "zhubajie-run"),
                                ("唐僧 · 行走上色", "tangsanzang-run"),
                                ("关羽 · 持刀上色", "guanyu-run"),
                                ("诸葛亮 · 羽扇上色", "zhugeliang-run"),
                                ("IP 签名笔 · 固定品牌线条/落款", "ip-signature"),
                                ("IP 印章笔 · 结论/重点盖章", "ip-stamp"),
                                ("IP 星光笔 · 高光/转场强调", "ip-spark"),
                                ("橡皮 · 水墨留白/擦出颜色", "eraser"),
                                ("不显示上色工具", "none"),
                            ],
                            value=DEFAULT_DRAWING_TOOL, label="上色工具", scale=2
                        )
                        color_fill = gr.Dropdown(
                            choices=[("自然涂鸦上色（推荐）", "doodle-fill"), ("自然修复上色", "natural-repair"), ("笔触跟随上色", "brush-path"), ("轮廓渐染", "contour-wipe"), ("横向排刷", "brush-scan"), ("分块填充", "top-down-blocks"), ("柔和渐显", "fade"), ("鹅毛笔排线", "quill-fill")],
                            value="natural-repair", label="上色方式", interactive=True, scale=2
                        )
                        tail_color = gr.Number(value=4, minimum=0, maximum=10, precision=1, label="上色时长（秒，自动建议）", interactive=True, scale=1)

                with gr.Accordion("水墨高级参数", open=False, visible=False) as inkwash_settings:
                    with gr.Row():
                        inkwash_bone_delta = gr.Slider(15, 140, value=70, step=5, label="浓墨")
                        inkwash_mid_delta = gr.Slider(8, 80, value=32, step=2, label="中墨")
                    with gr.Row():
                        inkwash_pale_delta = gr.Slider(4, 60, value=22, step=1, label="晕染洁净度")
                        inkwash_grad_thresh = gr.Slider(1, 20, value=6, step=1, label="边缘敏感度")
                    with gr.Row():
                        inkwash_min_area = gr.Slider(1, 50, value=6, step=1, label="杂点过滤")
                        inkwash_min_elon = gr.Slider(1, 3, value=1.25, step=.05, label="飞白保留")
                    with gr.Row():
                        inkwash_solid_thresh = gr.Slider(10, 250, value=100, step=5, label="墨块填充")
                        inkwash_draw_thresh = gr.Slider(10, 250, value=90, step=5, label="结构线范围")

                with gr.Accordion("高级渲染", open=False, visible=False):
                    with gr.Row():
                        stroke_detail = gr.Dropdown([("极致细节", "max"), ("丰富细节", "rich"), ("平衡", "balanced")], value="max", label="笔触细节")
                    with gr.Row():
                        draw_mode = gr.Dropdown([("直接铺墨", "direct-ink"), ("结构后铺墨", "structure-then-ink")], value="direct-ink", label="运笔模式")
                        line_thickness = gr.Slider(0, 10, value=0, step=1, label="线条粗细（0 为自动）")
                    with gr.Row():
                        ink_darkness = gr.Slider(30, 100, value=90, step=5, label="墨色深度")
                        ink_brush = gr.Slider(2, 10, value=5.5, step=.5, label="笔触宽度")
                    disable_hatching = gr.Checkbox(value=False, label="禁用铺墨填充")
            
    # Simple UI feedback: disable tail_color slider if color_enabled is false
    def toggle_color_options(enabled):
        return (
            gr.update(interactive=enabled),
            gr.update(interactive=enabled),
            gr.update(interactive=enabled),
        )
        
    color_enabled.change(
        fn=toggle_color_options,
        inputs=[color_enabled],
        outputs=[tail_color, color_fill, color_tool]
    )

    def apply_manual_color_tool(tool):
        fill_mode = COLOR_TOOL_CONFIG.get(tool, ("brush", "natural-repair"))[1]
        return gr.update(value=fill_mode), True

    color_tool.input(
        fn=apply_manual_color_tool,
        inputs=[color_tool],
        outputs=[color_fill, color_tool_overridden],
        queue=False,
    )

    def sync_color_tool(draw_tool, current_color_tool, manually_overridden):
        value = matching_color_tool(draw_tool, current_color_tool, manually_overridden)
        return gr.update(value=value)

    hand.change(
        fn=sync_color_tool,
        inputs=[hand, color_tool, color_tool_overridden],
        outputs=[color_tool],
        queue=False,
    )
    
    # Disable lineart provider selection if bypass_ai is checked
    def toggle_ai_provider(bypass):
        return gr.update(interactive=not bypass)
        
    bypass_ai.change(
        fn=toggle_ai_provider,
        inputs=[bypass_ai],
        outputs=[lineart_provider]
    )

    def switch_style(mode):
        if mode == "ink-wash":
            return (
                gr.update(value="ink-wash", interactive=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value="natural-repair"),
            )
        if mode == "doodle":
            return (
                gr.update(value="doodle-color", interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="natural-repair"),
            )
        return (
            gr.update(
                value="anime2sketch",
                interactive=True,
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="natural-repair"),
        )

    style_mode.change(
        fn=switch_style,
        inputs=[style_mode],
        outputs=[lineart_provider, inkwash_settings, inkwash_preset, color_fill],
    )

    def invalidate_extraction(*_):
        return False

    def update_render_button(ready):
        # Backend validation remains authoritative; do not let async state
        # updates hide the command from the user.
        return gr.update(interactive=True)

    extraction_ready.change(
        fn=update_render_button,
        inputs=[extraction_ready],
        outputs=[render_btn],
    )

    # Show/hide inkwash settings dynamically based on provider selection
    def toggle_inkwash_settings(provider):
        return gr.update(visible=(provider == "ink-wash"))
        
    # The visible style selector is the source of truth. This hidden provider
    # event only updates presentation; it must not invalidate a fresh extract,
    # otherwise switching to water-ink can disable the render button again.
    lineart_provider.change(
        fn=lambda provider: (
            gr.update(visible=(provider == "ink-wash")),
            gr.update(visible=(provider == "ink-wash")),
        ),
        inputs=[lineart_provider],
        outputs=[inkwash_settings, inkwash_preset]
    )

    inkwash_preset.change(
        fn=lambda name: (*apply_inkwash_preset(name), False),
        inputs=[inkwash_preset],
        outputs=[
            inkwash_bone_delta,
            inkwash_mid_delta,
            inkwash_pale_delta,
            inkwash_grad_thresh,
            inkwash_min_area,
            inkwash_min_elon,
            inkwash_solid_thresh,
            inkwash_draw_thresh,
            extraction_ready,
        ],
    )
    
    # Automatically update the original image preview when user uploads a file
    def show_original_image(file_obj, total_duration, draw_tool):
        prepared_path = _prepare_white_background(file_obj)
        matched_tool = matching_color_tool(draw_tool, DEFAULT_DRAWING_TOOL, manually_overridden=False)
        return (
            prepared_path,
            _suggest_color_duration(prepared_path, total_duration),
            gr.update(value=matched_tool),
            False,
        )

    def reset_for_new_upload(file_obj):
        path = _prepare_white_background(file_obj)
        if not path:
            return None, None, "请上传图片。", False
        return path, None, "图片已上传，请点击“提取线稿”。", False

    def auto_extract_uploaded(file_obj, mode, bypass):
        file_obj = _prepare_white_background(file_obj)
        provider = "ink-wash" if mode == "ink-wash" else ("doodle-color" if mode == "doodle" else "anime2sketch")
        bypass = bool(bypass or _lineart_mode(mode))
        bone, mid, pale, grad, area, elon, solid, draw = INKWASH_PRESETS["balanced"]
        return extract_lineart_preview(
            file_obj,
            provider,
            bypass,
            "max",
            bone,
            mid,
            pale,
            grad,
            area,
            elon,
            solid,
            draw,
        )

    def extract_from_ui(file_obj, mode, bypass, detail, bone, mid, pale, grad, area, elon, solid, draw):
        file_obj = _prepare_white_background(file_obj)
        provider = "ink-wash" if mode == "ink-wash" else ("doodle-color" if mode == "doodle" else "anime2sketch")
        bypass = bool(bypass or _lineart_mode(mode))
        return extract_lineart_preview(
            file_obj, provider, bypass, detail,
            bone, mid, pale, grad, area, elon, solid, draw,
        )

    style_mode.change(
        fn=auto_extract_uploaded,
        inputs=[input_file, style_mode, bypass_ai],
        outputs=[lineart_preview, status_box, extraction_ready],
    )

    input_file.change(
        fn=show_original_image,
        inputs=[input_file, duration, hand],
        outputs=[input_image_preview, tail_color, color_tool, color_tool_overridden],
        queue=False,
    )

    duration.change(
        fn=_suggest_color_duration,
        inputs=[input_file, duration],
        outputs=[tail_color],
        queue=False,
    )

    # Upload replaces the current image immediately, then starts extraction
    # automatically so the preview is ready without an extra click.
    input_file.upload(
        fn=auto_extract_uploaded,
        inputs=[input_file, style_mode, bypass_ai],
        outputs=[lineart_preview, status_box, extraction_ready],
        queue=False,
        show_progress="minimal",
    )

    bypass_ai.change(
        fn=invalidate_extraction,
        inputs=[bypass_ai],
        outputs=[extraction_ready],
    )

    theme_text.input(
        fn=render_theme_font_preview,
        inputs=[theme_text, theme_font_style, theme_font_size],
        outputs=[theme_font_preview],
        queue=False,
        show_progress="hidden",
    )
    theme_text.change(
        fn=render_theme_font_preview,
        inputs=[theme_text, theme_font_style, theme_font_size],
        outputs=[theme_font_preview],
        queue=False,
        show_progress="hidden",
    )
    theme_font_style.change(
        fn=render_theme_font_preview,
        inputs=[theme_text, theme_font_style, theme_font_size],
        outputs=[theme_font_preview],
        queue=False,
        show_progress="hidden",
    )
    theme_font_size.change(
        fn=render_theme_font_preview,
        inputs=[theme_text, theme_font_style, theme_font_size],
        outputs=[theme_font_preview],
        queue=False,
        show_progress="hidden",
    )

    # Event Handlers
    extract_btn.click(
        fn=extract_from_ui,
        inputs=[
            input_file,
            style_mode,
            bypass_ai,
            stroke_detail,
            inkwash_bone_delta,
            inkwash_mid_delta,
            inkwash_pale_delta,
            inkwash_grad_thresh,
            inkwash_min_area,
            inkwash_min_elon,
            inkwash_solid_thresh,
            inkwash_draw_thresh
        ],
        outputs=[lineart_preview, status_box, extraction_ready]
    )
            
    render_btn.click(
        fn=render_video_from_lineart,
        inputs=[
            input_file,
            extraction_ready,
            duration,
            fps,
            hand,
            theme_text,
            theme_font_style,
            theme_font_size,
            theme_position,
            seal_style,
            seal_text,
            seal_position,
            seal_x,
            seal_y,
            color_tool,
            color_fill,
            stroke_detail,
            line_thickness,
            lineart_provider,
            tail_color,
            color_enabled,
            resolution,
            bypass_ai,
            disable_hatching,
            draw_mode,
            ink_darkness,
            ink_brush,
            inkwash_bone_delta,
            inkwash_mid_delta,
            inkwash_pale_delta,
            inkwash_grad_thresh,
            inkwash_min_area,
            inkwash_min_elon,
            inkwash_solid_thresh,
            inkwash_draw_thresh
        ],
        outputs=[output_video_player, progress_bar, status_box],
        show_progress="hidden",
        scroll_to_output=False,
    )

if __name__ == "__main__":
    _cleanup_out_dir()
    # Default to loopback for safety; set WHITEBOARD_HOST=0.0.0.0 for LAN access.
    host = os.environ.get("WHITEBOARD_HOST", "127.0.0.1")
    demo.launch(server_name=host, server_port=7860, theme=theme, css=APP_CSS)
