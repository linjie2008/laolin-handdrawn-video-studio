# Model Setup

The engine does not include neural model code or model weights. Install them
locally under `tools/` when you want to use `render-photo` or
`extract-lineart`.

The engine auto-discovers these wrapper scripts:

- `tools/lineart/run_informative_drawings.py`
- `tools/lineart/run_anime2sketch.py`

It also accepts explicit commands:

- `WHITEBOARD_INFORMATIVE_DRAWINGS_CMD`
- `WHITEBOARD_ANIME2SKETCH_CMD`

Commands may contain `{input}` and `{output}` placeholders.

## Python Environment For Line-Art Models

Create a separate environment for PyTorch model inference:

```bash
python3 -m venv .venv-lineart
.venv-lineart/bin/python -m pip install --upgrade pip setuptools wheel
.venv-lineart/bin/python -m pip install torch torchvision Pillow numpy tqdm
```

The engine will use `.venv-lineart/bin/python` automatically when it exists.

## Informative Drawings

Best default for photos and semantic line art.

Upstream:

- GitHub: https://github.com/carolineec/informative-drawings
- Project page: https://carolineec.github.io/informative_drawings/
- Paper: https://arxiv.org/abs/2203.12691
- Demo: https://huggingface.co/spaces/carolineec/informativedrawings

Install code:

```bash
mkdir -p tools
git clone https://github.com/carolineec/informative-drawings.git tools/informative-drawings
```

Download the pretrained model from the upstream README:

- https://drive.google.com/file/d/1MIdHzecxz-z0uY3ARL_R40DlKcuQxiDk/view?usp=sharing

Expected weight path, either of these:

```text
tools/informative-drawings/checkpoints/anime_style/netG_A_latest.pth
tools/informative-drawings/checkpoints/model/anime_style/netG_A_latest.pth
```

The second layout is common when the upstream `model.zip` is unzipped directly
inside `checkpoints/`.

Manual test:

```bash
.venv-lineart/bin/python tools/lineart/run_informative_drawings.py \
  input.jpg \
  out/lineart-informative.png \
  --style anime_style \
  --size 768
```

## Anime2Sketch

Best for anime, manga, illustration, and clean white-background art. It is not
the best default for real photos with dark or low-contrast backgrounds.

Upstream:

- GitHub: https://github.com/Mukosame/Anime2Sketch
- Paper reference in upstream README: https://arxiv.org/abs/2104.05703

Install code:

```bash
mkdir -p tools
git clone https://github.com/Mukosame/Anime2Sketch.git tools/Anime2Sketch
```

Download weights from the upstream README:

- Default weights folder: https://drive.google.com/drive/folders/1Srf-WYUixK0wiUddc9y3pNKHHno5PN6R?usp=sharing
- Artifact-free weight for dark / low-contrast images: https://drive.google.com/file/d/1cf90_fPW-elGOKu5mTXT5N1dum-XY_46/view?usp=sharing

Expected weight path:

```text
tools/Anime2Sketch/weights/netG.pth
tools/Anime2Sketch/weights/improved.bin
```

If `improved.bin` exists, the wrapper uses it automatically. Otherwise it uses
`netG.pth`.

Manual test:

```bash
.venv-lineart/bin/python tools/lineart/run_anime2sketch.py \
  input.png \
  out/lineart-anime2sketch.png \
  --load-size 768
```

## Provider Selection

```bash
whiteboard extract-lineart input.jpg -o out/lineart.png --provider auto
```

`auto` resolves in this order:

1. Informative Drawings, when installed and weighted.
2. Anime2Sketch, when installed and weighted.
3. Fail loudly.

There is no Canny, XDoG, or edge-only production fallback.

Use explicit providers when comparing quality:

```bash
whiteboard extract-lineart input.jpg -o out/informative.png --provider informative
whiteboard extract-lineart input.jpg -o out/anime2sketch.png --provider anime2sketch
```

## Version-Control Policy

Do not commit:

- model repositories under `tools/informative-drawings/` or `tools/Anime2Sketch/`
- model weights
- `.venv-lineart/`
- generated line art or videos

Keep only the wrapper scripts under `tools/lineart/`.
