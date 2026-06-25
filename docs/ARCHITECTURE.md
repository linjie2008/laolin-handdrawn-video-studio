# Architecture

This project has two layers:

1. Python engine: reusable package in `src/whiteboard_skill/`.
2. Codex skill adapter: package in `whiteboard-video/`.

The engine should stay independent from Codex. The skill should only describe
how Codex should invoke the engine.

## Pipeline

```text
script or image
  |
  | script path
  v
scene_split -> image_gen -> local line-art extraction
  |
  | image path
  v
line-art normalization -> raster/SVG stroke extraction -> stroke ordering
  |
  v
whiteboard renderer -> optional color fill -> FFmpeg MP4
```

For uploaded photos:

```text
source image
  -> local neural line-art provider
  -> extracted line-art PNG
  -> skeleton tracing / SVG parsing
  -> hand-drawn render
  -> contour-aware color fill from source image
```

## Core Modules

### `cli.py`

Main command-line interface:

- `extract-lineart`
- `render-photo`
- `render-image`
- `analyze-image`
- `plan-script`
- `run`
- `compose`
- `list-hands`
- `doctor`

### `providers/lineart.py`

Local model provider discovery and execution.

Supported providers:

- Informative Drawings
- Anime2Sketch

Provider commands can be discovered from:

1. Environment variables.
2. Executables on `PATH`.
3. Local wrappers under `tools/lineart/`.

No edge-only fallback is used in production. Missing neural providers should
fail loudly.

### `preprocess.py`

Converts SVG or raster line art into ordered drawable strokes.

Important steps:

- fit line art into the render canvas
- remove frame-like canvas borders
- threshold grayscale line-art pixels
- Zhang-Suen skeletonization to one-pixel-wide paths
- 8-neighbor path tracing
- frame stroke filtering
- endpoint-based stroke merging
- smoothing and resampling
- top-to-bottom, left-to-right ordering with local continuity

Stroke detail presets:

- `balanced`: fewer strokes, more continuity, lower detail.
- `rich`: default; keeps semantic detail while merging short fragments into
  longer strokes.
- `max`: keeps tiny marks and logos, but can look jumpier.

### `whiteboard.py`

Frame renderer.

Features:

- progressive stroke drawing with easing
- antialiased line rendering
- fixed-orientation hand cursor
- hand-drawn text strokes
- line-art snap completion with default threshold `170`
- contour-aware color fill
- FFmpeg MP4 encoding

### `pipeline.py`

Resumable script-to-video orchestration:

- split script into scenes
- generate scene images
- render scene clips
- compose final MP4

### `compose.py`

FFmpeg helper functions for clip concatenation.

## Stroke Continuity Strategy

The key renderer quality issue is not only line-art quality, but how many tiny
fragments the engine asks the hand to draw.

The current strategy:

- trace skeleton paths conservatively
- merge endpoints when the gap is small and tangent direction is compatible
- allow stronger merging for touching endpoints and short fragments
- filter very small isolated fragments in `rich`
- keep all tiny detail in `max`
- prefer longer strokes during ordering

This creates a long/medium/short stroke mix:

- long strokes for main contours
- medium strokes for hair, clothing folds, limbs, facial structure
- short strokes for expression, logos, fingers, shoes, and accent marks

## Color Fill

The renderer uses the original image as the color source when dimensions match.

Recommended path:

```bash
whiteboard render-image lineart.png \
  --source-image source.jpg \
  --source-fit exact \
  --size-from-image \
  --color-fill contour-wipe
```

The line-art and source image should share the same canvas size to avoid color
misalignment. Local line-art extraction preserves source dimensions.

## Model Integration Boundary

The engine does not import upstream model code as package dependencies. It runs
external wrappers through subprocesses. This keeps the core package light and
avoids forcing PyTorch on users who only render SVG or existing line-art PNGs.

Wrappers live in `tools/lineart/` and expect upstream repositories under:

```text
tools/informative-drawings/
tools/Anime2Sketch/
```

Weights are intentionally not part of this repository.

## Testing

Core tests should avoid requiring neural model weights. Model tests should be
manual or integration tests guarded by local file existence.

Recommended test categories:

- pure preprocessing tests
- renderer tests with tiny fixtures
- CLI smoke tests with SVG input
- optional model smoke tests when weights exist

Run:

```bash
MOCK=1 PYTHONPATH=src python3 -m pytest -q
```
