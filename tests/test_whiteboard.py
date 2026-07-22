from whiteboard_skill.preprocess import Stroke
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from whiteboard_skill.whiteboard import (
    ANIMATED_ENTRANCE_CURSOR_DIRS,
    THEME_AA_SCALE,
    SKETCH_INK_COLOR,
    _apply_red_seal,
    _animated_character_side,
    _animated_locomotion_frame,
    _animated_entrance_frame_count,
    _animated_sprite_position,
    _color_fill_frame,
    _color_finish_start,
    _complete_line_art_canvas,
    _estimate_line_art_width,
    _ffmpeg_encoding_options,
    _line_art_ink_mask,
    _load_animated_cursor_frames,
    _load_source_image_canvas,
    _prepare_timeline,
    _allocate_render_frames,
    _paste_real_eraser,
    _resolve_line_thickness,
    _render_aa_scale,
    _reveal_line_art_canvas,
    _stroke_segment_between,
    _skin_tone_mask,
    _order_color_strokes,
    _text_to_strokes,
    _theme_duration_seconds,
    _theme_preview_image,
    _theme_text_mask,
    _top_down_block_fill,
)


def test_raster_line_art_render_avoids_redundant_supersampling():
    assert _render_aa_scale((1280, 720), has_exact_line_art=True) == 1
    assert _render_aa_scale((1280, 720), has_exact_line_art=False) == 2


def test_default_video_encoding_is_tuned_for_interactive_speed():
    assert _ffmpeg_encoding_options() == ("veryfast", "16")


def test_complex_colors_delay_the_final_whole_image_blend():
    simple = Image.new("RGB", (80, 80), "white")
    complex_image = Image.new("RGB", (80, 80), "white")
    pixels = np.asarray(complex_image).copy()
    for y in range(80):
        for x in range(80):
            pixels[y, x] = ((x * 17) % 256, (y * 23) % 256, ((x + y) * 13) % 256)
    complex_image = Image.fromarray(pixels, mode="RGB")

    assert _color_finish_start(complex_image) > _color_finish_start(simple)
    assert _color_finish_start(complex_image) >= 0.87
    assert _color_finish_start(complex_image) <= 0.89


def test_color_strokes_finish_the_nearby_region_before_moving_away():
    strokes = [
        ([(180.0, 12.0), (195.0, 12.0)], 8.0),
        ([(21.0, 11.0), (30.0, 30.0)], 8.0),
        ([(5.0, 10.0), (20.0, 10.0)], 8.0),
    ]

    ordered = _order_color_strokes(strokes, (200, 200))

    assert ordered[0][0][0] == (5.0, 10.0)
    assert ordered[1][0][0] == (21.0, 11.0)


def test_color_duration_excludes_the_stamp_action():
    draw_frames, color_frames, stamp_frames = _allocate_render_frames(
        total_frames=300,
        fps=20,
        color_seconds=4.0,
    )

    assert color_frames == 80
    assert draw_frames + color_frames + stamp_frames == 300
    assert draw_frames >= 120


def test_stamp_enters_without_a_hard_cut_from_the_finished_image():
    finished = Image.new("RGB", (320, 180), (180, 150, 100))

    first_stamp_frame = _apply_red_seal(finished, progress=1 / 15)
    difference = np.abs(
        np.asarray(first_stamp_frame, dtype=np.float32)
        - np.asarray(finished, dtype=np.float32)
    ).mean()

    assert difference < 4.0


def test_seal_style_and_custom_text_change_the_final_imprint():
    finished = Image.new("RGB", (398, 548), "white")

    vintage = _apply_red_seal(finished, progress=1.0, text="老林涂鸦", style="vintage")
    inkwash = _apply_red_seal(finished, progress=1.0, text="老林涂鸦", style="inkwash")
    custom = _apply_red_seal(finished, progress=1.0, text="林哥印记", style="inkwash")

    assert ImageChops.difference(vintage, inkwash).getbbox() is not None
    assert ImageChops.difference(inkwash, custom).getbbox() is not None


def test_seal_position_moves_both_axes_on_the_canvas():
    finished = Image.new("RGB", (600, 400), "white")

    def center(position: str) -> tuple[float, float]:
        stamped = _apply_red_seal(finished, progress=1.0, position=position)
        box = ImageChops.difference(finished, stamped).getbbox()
        assert box is not None
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    left_top = center("left-top")
    middle = center("center-center")
    right_bottom = center("right-bottom")

    assert left_top[0] < middle[0] < right_bottom[0]
    assert left_top[1] < middle[1] < right_bottom[1]
    assert abs(middle[0] - finished.width / 2) < 4
    assert abs(middle[1] - finished.height / 2) < 4


def test_character_is_larger_on_entry_and_during_creation():
    creation = _animated_character_side((398, 548), stage="creation")
    entrance = _animated_character_side((398, 548), stage="entrance")

    assert creation >= 108
    assert entrance >= 132
    assert entrance > creation


def test_character_entry_is_slow_enough_to_show_a_run_cycle():
    assert _animated_entrance_frame_count(draw_frames=196, fps=20) == 36
    assert _animated_entrance_frame_count(draw_frames=75, fps=20) >= 33


def test_character_leg_cycle_is_driven_by_distance_not_video_time():
    assert _animated_locomotion_frame(0, 400, frame_count=4) == 0
    assert _animated_locomotion_frame(0, 400, frame_count=4) == 0
    assert _animated_locomotion_frame(15, 400, frame_count=4) == 1
    assert _animated_locomotion_frame(30, 400, frame_count=4) == 2


def test_pigsy_uses_a_full_run_and_rake_swing_cycle():
    frames = _load_animated_cursor_frames("zhubajie-run")

    assert len(frames) >= 8


def test_characters_use_dedicated_entrance_actions():
    expected_sizes = {
        "zhubajie-run": 8,
        "tangsanzang-run": 8,
        "zhugeliang-run": 8,
        "wukong-run": 8,
        "guanyu-run": 8,
    }

    for style, frame_count in expected_sizes.items():
        entrance = _load_animated_cursor_frames(style, stage="entrance")
        creation = _load_animated_cursor_frames(style, stage="creation")
        assert len(entrance) == frame_count
        assert entrance[0].tobytes() != creation[0].tobytes()

    for style in ("zhubajie-run", "tangsanzang-run", "zhugeliang-run"):
        assert ANIMATED_ENTRANCE_CURSOR_DIRS[style].name.endswith("walk-entry-v2")


def test_animated_character_is_kept_fully_inside_the_video_frame():
    assert _animated_sprite_position((398, 548), (150, 150), -25, 12) == (0, 12)
    assert _animated_sprite_position((398, 548), (150, 150), 370, 530) == (248, 398)


def test_real_eraser_trails_its_contact_point_in_both_directions():
    canvas = Image.new("RGB", (240, 140), "white")

    moving_right = _paste_real_eraser(canvas, 120, 70, 0.0)
    moving_left = _paste_real_eraser(canvas, 120, 70, 3.141592653589793)

    assert ImageChops.difference(canvas, moving_right).crop((45, 35, 115, 105)).getbbox() is not None
    assert ImageChops.difference(canvas, moving_left).crop((125, 35, 195, 105)).getbbox() is not None


def test_skin_tone_mask_keeps_skin_shades_without_selecting_white_or_primary_colors():
    swatches = Image.new("RGB", (60, 10), "white")
    values = [
        (238, 194, 160),
        (158, 105, 76),
        (255, 255, 255),
        (230, 25, 30),
        (25, 95, 220),
        (250, 220, 20),
    ]
    for x, value in enumerate(values):
        for px in range(x * 10, (x + 1) * 10):
            for py in range(10):
                swatches.putpixel((px, py), value)

    mask = _skin_tone_mask(swatches)

    selected = [bool(mask[5, x * 10 + 5]) for x in range(6)]
    assert selected == [True, True, False, False, False, False]


def test_resolve_line_thickness_adapts_unless_overridden():
    assert _resolve_line_thickness(0, 4) == 4
    assert _resolve_line_thickness(None, 3) == 3
    assert _resolve_line_thickness(1, 5) == 1


def test_prepare_timeline_keeps_stroke_intervals_ordered():
    strokes = [
        Stroke(points=[(0, 0), (20, 0)]),
        Stroke(points=[(30, 0), (40, 0)]),
    ]

    timeline = _prepare_timeline(strokes, draw_frames=60)
    assert len(timeline) == 2
    assert timeline[0].start_unit == 0
    assert timeline[0].end_unit <= timeline[0].pause_end_unit
    assert timeline[0].pause_end_unit <= timeline[1].start_unit


def test_prepare_timeline_drops_pauses_for_dense_short_duration():
    strokes = [Stroke(points=[(0, float(i)), (20, float(i))]) for i in range(80)]

    timeline = _prepare_timeline(strokes, draw_frames=30)

    assert timeline
    assert all(item.pause_end_unit == item.end_unit for item in timeline)


def test_stroke_segment_between_interpolates_partial_path():
    stroke = Stroke(points=[(0, 0), (10, 0), (10, 10)])
    timeline = _prepare_timeline([stroke], draw_frames=30)[0]

    segment, seg_widths = _stroke_segment_between(stroke.points, timeline.cumulative, 5, 15)
    assert segment[0] == (5, 0)
    assert segment[-1] == (10, 5)
    assert seg_widths is None


def test_load_source_image_canvas_blur_fill_covers_portrait(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 100), (200, 40, 40)).save(source)

    canvas = _load_source_image_canvas(source, (90, 160), "blur-fill")
    assert canvas.size == (90, 160)
    assert canvas.getpixel((2, 2)) != (255, 255, 255)


def test_load_source_image_canvas_exact_resizes_to_canvas(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (30, 60), (20, 80, 200)).save(source)

    canvas = _load_source_image_canvas(source, (90, 160), "exact")
    assert canvas.size == (90, 160)


def test_top_down_block_fill_reveals_hard_rows():
    canvas = Image.new("RGB", (10, 10), "white")
    source = Image.new("RGB", (10, 10), "black")

    frame = _top_down_block_fill(canvas, source, progress=0.5, blocks=2)
    assert frame.getpixel((5, 1)) == (0, 0, 0)
    assert frame.getpixel((5, 8)) == (255, 255, 255)


def test_brush_scan_fill_returns_moving_cursor():
    canvas = Image.new("RGB", (10, 10), "white")
    source = Image.new("RGB", (10, 10), "black")

    frame, cursor, angle = _color_fill_frame(canvas, source, progress=0.3, mode="brush-scan", blocks=4)
    assert cursor is not None
    assert frame.getpixel((1, 1)) == (0, 0, 0)
    assert angle in (0.0, 3.141592653589793)


def test_contour_fill_grows_from_contours_instead_of_global_scan():
    canvas = Image.new("RGB", (40, 40), "white")
    for y in range(17, 20):
        for x in range(6, 34):
            canvas.putpixel((x, y), (18, 18, 18))
    source = Image.new("RGB", (40, 40), (210, 30, 30))

    frame, cursor, _angle = _color_fill_frame(canvas, source, progress=0.55, mode="contour-wipe", blocks=4)

    assert cursor is not None
    # The contour and nearby interior are colored first, while a distant area
    # remains unfinished. This guards against a global top-to-bottom wipe.
    assert frame.getpixel((20, 18)) == (210, 30, 30)
    assert frame.getpixel((20, 35)) != (210, 30, 30)


def test_complete_line_art_canvas_restores_missing_ink():
    canvas = Image.new("RGB", (20, 20), "white")
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((10, 10), (0, 0, 0))

    completed = _complete_line_art_canvas(canvas, line_art)

    pixel = completed.getpixel((10, 10))
    assert all(SKETCH_INK_COLOR[idx] <= pixel[idx] < 255 for idx in range(3))


def test_complete_line_art_canvas_can_blend_missing_ink():
    canvas = Image.new("RGB", (20, 20), "white")
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((10, 10), (0, 0, 0))

    completed = _complete_line_art_canvas(canvas, line_art, alpha=0.5)

    assert completed.getpixel((10, 10))[0] > 18
    assert completed.getpixel((10, 10))[0] < 255


def test_complete_line_art_canvas_ignores_light_gray_noise_by_default():
    canvas = Image.new("RGB", (20, 20), "white")
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((10, 10), (238, 238, 238))

    completed = _complete_line_art_canvas(canvas, line_art)

    assert completed.getpixel((10, 10)) == (255, 255, 255)


def test_reveal_line_art_canvas_uses_reveal_mask():
    canvas = Image.new("RGB", (20, 20), "white")
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((5, 5), (0, 0, 0))
    line_art.putpixel((15, 15), (0, 0, 0))
    reveal = Image.new("L", (20, 20), 0)
    reveal.putpixel((5, 5), 255)

    frame = _reveal_line_art_canvas(canvas, line_art, reveal)

    pixel = frame.getpixel((5, 5))
    assert all(SKETCH_INK_COLOR[idx] <= pixel[idx] < 255 for idx in range(3))
    assert frame.getpixel((15, 15)) == (255, 255, 255)


def test_tone_aware_reveal_clears_temporary_ink_outside_source_line():
    canvas = Image.new("RGB", (20, 20), "white")
    canvas.putpixel((5, 5), (20, 20, 20))
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((10, 10), (0, 0, 0))
    reveal = Image.new("L", (20, 20), 0)
    reveal.putpixel((5, 5), 255)

    frame = _reveal_line_art_canvas(canvas, line_art, reveal, preserve_tones=True)

    assert frame.getpixel((5, 5)) == (255, 255, 255)


def test_tone_aware_reveal_uses_exact_extracted_pixels_during_drawing():
    canvas = Image.new("RGB", (20, 20), "white")
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((5, 5), (73, 73, 73))
    line_art.putpixel((15, 15), (121, 121, 121))
    reveal = Image.new("L", (20, 20), 0)
    reveal.putpixel((5, 5), 255)

    frame = _reveal_line_art_canvas(canvas, line_art, reveal, preserve_tones=True)

    assert frame.getpixel((5, 5)) == (73, 73, 73)
    assert frame.getpixel((15, 15)) == (255, 255, 255)


def test_tone_aware_reveal_accepts_precomputed_exact_layer():
    canvas = Image.new("RGB", (20, 20), "white")
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((5, 5), (73, 73, 73))
    reveal = Image.new("L", (20, 20), 0)
    reveal.putpixel((5, 5), 255)
    exact_layer = _complete_line_art_canvas(canvas, line_art, preserve_tones=True)

    frame = _reveal_line_art_canvas(
        canvas,
        None,
        reveal,
        preserve_tones=True,
        exact_layer=exact_layer,
    )

    assert frame.getpixel((5, 5)) == (73, 73, 73)


def test_line_art_canvas_accepts_precomputed_ink_mask():
    canvas = Image.new("RGB", (20, 20), "white")
    line_art = Image.new("RGB", (20, 20), "white")
    line_art.putpixel((5, 5), (0, 0, 0))
    line_art.putpixel((8, 8), (0, 0, 0))
    reveal = Image.new("L", (20, 20), 0)
    reveal.putpixel((5, 5), 255)
    ink_mask = _line_art_ink_mask(line_art, (20, 20))

    completed = _complete_line_art_canvas(canvas, line_art)
    completed_cached = _complete_line_art_canvas(canvas, None, ink_mask=ink_mask)
    revealed = _reveal_line_art_canvas(canvas, line_art, reveal)
    revealed_cached = _reveal_line_art_canvas(canvas, None, reveal, ink_mask=ink_mask)

    assert completed.tobytes() == completed_cached.tobytes()
    assert revealed.tobytes() == revealed_cached.tobytes()


def test_estimate_line_art_width_from_ink_area():
    line_art = Image.new("RGB", (40, 40), "white")
    for y in range(19, 22):
        for x in range(5, 35):
            line_art.putpixel((x, y), (0, 0, 0))

    assert _estimate_line_art_width(line_art) >= 3


def test_line_art_ink_mask_keeps_solid_ink_body():
    """Snap mask must be solid ink mass, never a skeleton centerline."""
    line_art = Image.new("RGB", (40, 40), "white")
    for y in range(18, 23):
        for x in range(5, 35):
            line_art.putpixel((x, y), (0, 0, 0))

    mask = _line_art_ink_mask(line_art, (40, 40))

    # Full thickness of the bar stays ink (not thinned to a 1px bone).
    assert mask.getpixel((20, 20)) == 255
    assert mask.getpixel((20, 19)) == 255
    assert mask.getpixel((20, 18)) == 255
    assert mask.getpixel((20, 17)) == 0


def test_tone_aware_line_art_snap_preserves_ink_layers():
    canvas = Image.new("RGB", (24, 24), "white")
    line_art = Image.new("L", (24, 24), 255)
    line_art.putpixel((4, 4), 0)      # 浓墨
    line_art.putpixel((12, 12), 55)   # 中墨
    line_art.putpixel((20, 20), 200)  # 淡墨

    result = _complete_line_art_canvas(canvas, line_art.convert("RGB"), preserve_tones=True)
    dark = result.getpixel((4, 4))[0]
    mid = result.getpixel((12, 12))[0]
    wash = result.getpixel((20, 20))[0]

    assert dark < mid < wash < 255


def test_text_to_strokes_generates_drawable_paths():
    strokes = _text_to_strokes("Hi", (240, 160))

    assert strokes
    assert all(len(stroke.points) >= 2 for stroke in strokes)


def test_theme_fonts_create_distinct_chinese_brush_paths():
    mao = _text_to_strokes("嫦娥", (398, 548), position="top", font_style="mao")
    cursive = _text_to_strokes("嫦娥", (398, 548), position="top", font_style="cursive")

    assert mao and cursive
    assert [stroke.points for stroke in mao] != [stroke.points for stroke in cursive]


def test_theme_writing_gets_a_readable_dedicated_duration():
    assert _theme_duration_seconds(None) == 0
    assert 1.8 <= _theme_duration_seconds("嫦娥") <= 2.5
    assert _theme_duration_seconds("嫦娥奔月") > _theme_duration_seconds("嫦娥")


def test_theme_font_preview_uses_the_selected_real_font():
    mao = _theme_preview_image("嫦娥", "mao")
    cursive = _theme_preview_image("嫦娥", "cursive")

    assert mao.size == cursive.size == (320, 104)
    assert ImageChops.difference(mao, cursive).getbbox() is not None


def test_theme_text_preserves_manual_line_breaks_in_preview_and_render():
    preview = _theme_preview_image("嫦娥\n奔月", "mao", 72)
    single_line = _theme_preview_image("嫦娥奔月", "mao", 72)
    multiline_mask = _theme_text_mask("嫦娥\n奔月", (398, 548), "mao", 72, position="top")
    single_line_mask = _theme_text_mask("嫦娥奔月", (398, 548), "mao", 72, position="top")

    assert ImageChops.difference(preview, single_line).getbbox() is not None
    assert ImageChops.difference(multiline_mask, single_line_mask).getbbox() is not None


def test_theme_font_size_changes_the_drawn_title_bounds():
    small = _text_to_strokes(
        "嫦娥", (1080, 1920), position="right", font_style="mao", font_size=48
    )
    large = _text_to_strokes(
        "嫦娥", (1080, 1920), position="right", font_style="mao", font_size=112
    )

    small_height = max(y for stroke in small for _, y in stroke.points) - min(y for stroke in small for _, y in stroke.points)
    large_height = max(y for stroke in large for _, y in stroke.points) - min(y for stroke in large for _, y in stroke.points)
    assert large_height > small_height * 1.6


def test_theme_writing_uses_supersampled_antialiasing():
    assert THEME_AA_SCALE >= 3


def test_theme_text_mask_preserves_complete_font_glyphs_at_render_scale():
    base = _theme_text_mask("嫦娥", (398, 548), "mao", 72)
    supersampled = _theme_text_mask("嫦娥", (398, 548), "mao", 72, scale=THEME_AA_SCALE)
    reduced = supersampled.resize(base.size, Image.Resampling.LANCZOS)

    assert base.getbbox() is not None
    assert reduced.getbbox() is not None
    assert supersampled.size == (398 * THEME_AA_SCALE, 548 * THEME_AA_SCALE)
    reduced_values = np.asarray(reduced, dtype=np.uint8)
    assert np.any((reduced_values > 0) & (reduced_values < 255))


def test_theme_text_mask_honors_user_selected_position():
    resolution = (398, 548)
    boxes = {
        position: _theme_text_mask("嫦娥", resolution, "mao", 72, position=position).getbbox()
        for position in ("right", "left", "top", "bottom", "center")
    }

    assert boxes["right"][0] > resolution[0] / 2
    assert boxes["left"][2] < resolution[0] / 2
    assert boxes["top"][1] < resolution[1] / 3
    assert boxes["bottom"][3] > resolution[1] * 2 / 3
    center_y = (boxes["center"][1] + boxes["center"][3]) / 2
    assert abs(center_y - resolution[1] / 2) < 2
