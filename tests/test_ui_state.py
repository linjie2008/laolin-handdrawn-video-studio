from whiteboard_skill.ui_state import DEFAULT_DRAWING_TOOL, matching_color_tool


def test_pigsy_is_the_default_drawing_tool():
    assert DEFAULT_DRAWING_TOOL == "zhubajie-run"


def test_coloring_tool_follows_drawing_tool_by_default():
    assert matching_color_tool("zhubajie-run", "soft-goat", manually_overridden=False) == "zhubajie-run"
    assert matching_color_tool("brush", "quill", manually_overridden=False) == "soft-goat"
    assert matching_color_tool("none", "quill", manually_overridden=False) == "none"


def test_manual_coloring_tool_selection_stops_automatic_following():
    assert matching_color_tool("guanyu-run", "eraser", manually_overridden=True) == "eraser"
