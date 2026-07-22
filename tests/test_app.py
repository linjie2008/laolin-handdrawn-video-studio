import ast
from pathlib import Path


def test_animated_characters_are_available_as_coloring_tools():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "COLOR_TOOL_CONFIG" for target in node.targets)
    )
    color_tool_config = ast.literal_eval(assignment.value)

    expected = {
        "wukong-run",
        "zhubajie-run",
        "tangsanzang-run",
        "guanyu-run",
        "zhugeliang-run",
    }

    assert expected <= color_tool_config.keys()
    assert {color_tool_config[name][0] for name in expected} == expected


def test_inkwash_mode_shows_preset_and_all_advanced_controls():
    source = Path("app.py").read_text(encoding="utf-8")
    switch_start = source.index("    def switch_style(mode):")
    switch_end = source.index("    style_mode.change(", switch_start)
    ink_branch = source[switch_start:switch_end]

    assert ink_branch.count("gr.update(visible=True)") >= 2
    assert 'gr.update(visible=(provider == "ink-wash"))' in source


def test_common_video_settings_appear_before_optional_inkwash_controls():
    source = Path("app.py").read_text(encoding="utf-8")

    assert source.index('gr.Accordion("视频输出"') < source.index('gr.Accordion("水墨高级参数"')


def test_settings_panel_keeps_sections_accessible_without_page_scrollbars():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "height: calc(100vh - 91px) !important;" in source
    assert "max-height: calc(100vh - 91px) !important;" in source
    assert "overflow-y: auto !important;" in source
    assert "flex-wrap: nowrap !important;" in source
    assert ".settings-panel::-webkit-scrollbar { display: none !important;" in source


def test_theme_input_accepts_manual_line_breaks():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'placeholder="例如：嫦娥（按 Enter 换行）"' in source
    assert "max_lines=4" in source
    assert 'cleaned_theme = "\\n".join(' in source


def test_frontend_uses_laolin_product_branding():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'title="老林手绘视频工坊"' in source
    assert "LAOLIN HAND-DRAWN VIDEO STUDIO" in source
