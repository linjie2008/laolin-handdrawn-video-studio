"""Local image-to-line-art providers.

The production path is explicit: use a locally installed neural extractor and
fail fast when no model-backed provider is available. This avoids silently
falling back to weak edge detection for uploaded photos.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class LineArtProvider(ABC):
    """Extract a black-on-white line-art bitmap from a color image."""

    @abstractmethod
    def extract(self, color_png: Path, out_png: Path) -> Path:
        """Write line art to `out_png` and return it."""


class ExternalCommandLineArt(LineArtProvider):
    """Line-art provider backed by an external command.

    The command may contain `{input}` and `{output}` placeholders. If it does
    not, the input and output paths are appended as positional arguments.
    """

    def __init__(self, command: str, name: str) -> None:
        self.command = command
        self.name = name

    def extract(self, color_png: Path, out_png: Path) -> Path:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        if "{input}" in self.command or "{output}" in self.command:
            command = self.command.format(input=str(color_png), output=str(out_png))
            args = shlex.split(command)
        else:
            args = [*shlex.split(self.command), str(color_png), str(out_png)]
        subprocess.run(args, check=True)
        if not out_png.exists():
            raise RuntimeError(f"{self.name} did not create output: {out_png}")
        _clean_canvas_edges(out_png)
        return out_png


class InformativeDrawingsLineArt(ExternalCommandLineArt):
    """Wrapper for a locally installed Informative-Drawings extractor."""

    def __init__(self, command: str) -> None:
        super().__init__(command, "informative-drawings")


class Anime2SketchLineArt(ExternalCommandLineArt):
    """Wrapper for a locally installed Anime2Sketch extractor."""

    def __init__(self, command: str) -> None:
        super().__init__(command, "anime2sketch")


def get_lineart_provider(name: str = "auto") -> LineArtProvider:
    """Return a configured local line-art provider."""

    normalized = name.lower().strip()
    if normalized == "auto":
        informative = _command_from_env_path_or_local_wrapper(
            "WHITEBOARD_INFORMATIVE_DRAWINGS_CMD",
            "informative-drawings",
            "run_informative_drawings.py",
            _informative_weights_ready,
        )
        if informative:
            return InformativeDrawingsLineArt(informative)
        anime = _command_from_env_path_or_local_wrapper(
            "WHITEBOARD_ANIME2SKETCH_CMD",
            "anime2sketch-lineart",
            "run_anime2sketch.py",
            _anime2sketch_weights_ready,
        )
        if anime:
            return Anime2SketchLineArt(anime)
        raise RuntimeError(
            "No neural line-art provider is installed. Install Informative Drawings weights "
            "at tools/informative-drawings/checkpoints/anime_style/netG_A_latest.pth "
            "or tools/informative-drawings/checkpoints/model/anime_style/netG_A_latest.pth; install "
            "Anime2Sketch weights at tools/Anime2Sketch/weights/netG.pth / improved.bin. "
            "You can also set WHITEBOARD_INFORMATIVE_DRAWINGS_CMD or WHITEBOARD_ANIME2SKETCH_CMD."
        )
    if normalized in {"informative", "informative-drawings"}:
        command = _command_from_env_path_or_local_wrapper(
            "WHITEBOARD_INFORMATIVE_DRAWINGS_CMD",
            "informative-drawings",
            "run_informative_drawings.py",
            _informative_weights_ready,
            strict_local=True,
        )
        if not command:
            raise RuntimeError("Informative-Drawings command not found. Set WHITEBOARD_INFORMATIVE_DRAWINGS_CMD.")
        return InformativeDrawingsLineArt(command)
    if normalized in {"anime", "anime2sketch", "manga", "manga-line", "manga-line-extraction"}:
        command = _command_from_env_path_or_local_wrapper(
            "WHITEBOARD_ANIME2SKETCH_CMD",
            "anime2sketch-lineart",
            "run_anime2sketch.py",
            _anime2sketch_weights_ready,
            strict_local=True,
        )
        if not command:
            raise RuntimeError("Anime2Sketch command not found. Set WHITEBOARD_ANIME2SKETCH_CMD.")
        return Anime2SketchLineArt(command)
    raise ValueError(f"Unknown line-art provider: {name}")


def vectorize_with_vtracer(line_png: Path, svg_path: Path) -> Path:
    """Vectorize a line-art bitmap with vtracer when the CLI is installed."""

    executable = shutil.which("vtracer")
    if not executable:
        raise RuntimeError("vtracer CLI not found on PATH")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            executable,
            "--input",
            str(line_png),
            "--output",
            str(svg_path),
            "--mode",
            "polygon",
            "--filter_speckle",
            "4",
            "--color_precision",
            "1",
        ],
        check=True,
    )
    return svg_path


def _command_from_env_path_or_local_wrapper(
    env_name: str,
    executable: str,
    wrapper_name: str,
    readiness_check,
    strict_local: bool = False,
) -> str | None:
    value = os.getenv(env_name)
    if value:
        return value
    found = shutil.which(executable)
    if found:
        return found
    wrapper = _find_local_wrapper(wrapper_name)
    if wrapper and readiness_check(wrapper):
        python = _lineart_python(wrapper)
        return f"{shlex.quote(str(python))} {shlex.quote(str(wrapper))} {{input}} {{output}}"
    if strict_local and wrapper:
        return None
    return None


def _find_local_wrapper(wrapper_name: str) -> Path | None:
    search_roots = [Path.cwd()]
    module_root = Path(__file__).resolve()
    search_roots.extend(module_root.parents)
    for root in search_roots:
        candidate = root / "tools" / "lineart" / wrapper_name
        if candidate.exists():
            return candidate
    return None


def _lineart_python(wrapper: Path) -> Path:
    for root in [wrapper.parents[2], *wrapper.parents]:
        candidate = root / ".venv-lineart" / "bin" / "python"
        if candidate.exists():
            return candidate
    return Path(os.getenv("WHITEBOARD_LINEART_PYTHON", "python3"))


def _informative_weights_ready(wrapper: Path) -> bool:
    root = wrapper.parents[1] / "informative-drawings"
    return (
        (root / "checkpoints" / "anime_style" / "netG_A_latest.pth").exists()
        or (root / "checkpoints" / "model" / "anime_style" / "netG_A_latest.pth").exists()
    )


def _anime2sketch_weights_ready(wrapper: Path) -> bool:
    root = wrapper.parents[1] / "Anime2Sketch"
    weights = root / "weights"
    return (weights / "netG.pth").exists() or (weights / "improved.bin").exists()


def _clean_canvas_edges(image_path: Path, threshold: int = 250) -> None:
    from ..preprocess import suppress_canvas_border

    image = Image.open(image_path).convert("RGB")
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = gray < threshold
    cleaned = suppress_canvas_border(ink)
    if np.array_equal(ink, cleaned):
        return
    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[ink & ~cleaned] = 255
    Image.fromarray(arr, mode="RGB").save(image_path)
