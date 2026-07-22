"""Pure UI-state rules shared by the Gradio frontend and tests."""

DEFAULT_DRAWING_TOOL = "zhubajie-run"

DRAW_TO_COLOR_TOOL = {
    "quill": "quill",
    "rooster-quill": "rooster-quill",
    "brush": "soft-goat",
    "wukong-run": "wukong-run",
    "zhubajie-run": "zhubajie-run",
    "tangsanzang-run": "tangsanzang-run",
    "guanyu-run": "guanyu-run",
    "zhugeliang-run": "zhugeliang-run",
    "ip-signature": "ip-signature",
    "ip-stamp": "ip-stamp",
    "ip-spark": "ip-spark",
    "none": "none",
}


def matching_color_tool(
    drawing_tool: str,
    current_color_tool: str,
    manually_overridden: bool,
) -> str:
    """Follow the drawing tool until the user explicitly picks a color tool."""

    if manually_overridden:
        return current_color_tool
    return DRAW_TO_COLOR_TOOL.get(drawing_tool, current_color_tool)
