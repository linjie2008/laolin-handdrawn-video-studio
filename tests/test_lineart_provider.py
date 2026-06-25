from pathlib import Path
import sys

from PIL import Image, ImageDraw
import pytest

from whiteboard_skill.cli import _normalize_lineart
from whiteboard_skill.providers.lineart import get_lineart_provider


def test_auto_provider_uses_configured_neural_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tool = tmp_path / "fake_lineart_tool.py"
    tool.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())\n",
        encoding="utf-8",
    )
    source = Image.new("RGB", (120, 90), "white")
    draw = ImageDraw.Draw(source)
    draw.ellipse((30, 20, 90, 80), outline="black", width=4)
    source_path = tmp_path / "source.png"
    out_path = tmp_path / "lineart.png"
    source.save(source_path)

    monkeypatch.setenv("WHITEBOARD_INFORMATIVE_DRAWINGS_CMD", f"{sys.executable} {tool} {{input}} {{output}}")

    get_lineart_provider("auto").extract(source_path, out_path)

    assert out_path.exists()


def test_xdog_provider_is_not_available():
    with pytest.raises(ValueError):
        get_lineart_provider("xdog")


def test_normalize_lineart_clears_canvas_edges_without_alignment(tmp_path: Path):
    raw = Image.new("RGB", (40, 40), "white")
    draw = ImageDraw.Draw(raw)
    draw.rectangle((0, 0, 39, 39), outline="black", width=2)
    draw.line((10, 20, 30, 20), fill="black", width=1)
    raw_path = tmp_path / "raw.png"
    out_path = tmp_path / "normalized.png"
    raw.save(raw_path)

    _normalize_lineart(raw_path, out_path, threshold=224, clear_edge=4)

    result = Image.open(out_path).convert("RGB")
    assert result.getpixel((1, 1)) == (255, 255, 255)
    assert result.getpixel((20, 20)) == (0, 0, 0)
