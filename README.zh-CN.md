# Whiteboard Video Engine

[English](README.md)

本项目是一个本地优先的白板手绘视频引擎，用于把 SVG、线稿 PNG、插画或照片转换成逐笔绘制的 MP4 视频。它负责真正的渲染、线稿提取、路径追踪、手势跟随和颜色填充；Codex Skill 是独立仓库，只作为调用入口。

- Engine: <https://github.com/gnipbao/whiteboard-video-engine>
- Codex Skill: <https://github.com/gnipbao/codex-whiteboard-video-skill>
- Author: <https://github.com/gnipbao>
- 个人介绍：<https://ycnj2htgnvdy.feishu.cn/wiki/DOYRws0FmizhDAkkKGicvlpzndh?from=from_copylink>

## Demo

GitHub README 对 `<video>` 内联视频支持不稳定，所以这里使用 GIF 预览，并保留完整 MP4 链接。

<table>
  <tr>
    <td width="50%">
      <strong>输入图</strong><br>
      <img src="examples/cases/sports-illustration-anime2sketch/input.jpg" alt="Sports illustration input" width="360">
    </td>
    <td width="50%">
      <strong>输出预览</strong><br>
      <a href="examples/cases/sports-illustration-anime2sketch/output.mp4">
        <img src="examples/cases/sports-illustration-anime2sketch/output-preview.gif" alt="Whiteboard animation output preview" width="360">
      </a><br>
      <a href="examples/cases/sports-illustration-anime2sketch/output.mp4">查看完整 MP4</a>
    </td>
  </tr>
</table>

案例文件：

- `examples/cases/sports-illustration-anime2sketch/input.jpg`
- `examples/cases/sports-illustration-anime2sketch/output-preview.gif`
- `examples/cases/sports-illustration-anime2sketch/output.mp4`
- `examples/cases/sports-illustration-anime2sketch/README.md`

## 能做什么

- 将 SVG 或线稿 PNG 渲染成逐笔绘制的白板视频。
- 使用本地神经网络模型从上传图片中提取语义线稿。
- 将 raster skeleton 拆成可绘制 stroke，并尽量合并短线为长线。
- 支持 `asian`、`black`、`children`、`white` 四种内置手势 PNG。
- 支持固定手势方向，只跟随笔尖平移，减少旋转抖动。
- 支持 `--draw-text "短标题"`，把短文字转成手写路径。
- 支持轮廓感上色，让颜色从线稿边界内逐步填充到原图效果。

## 安装

从 GitHub 安装：

```bash
python3 -m pip install "git+https://github.com/gnipbao/whiteboard-video-engine.git"
```

本地开发安装：

```bash
git clone https://github.com/gnipbao/whiteboard-video-engine.git
cd whiteboard-video-engine
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

如果网络受限，先安装基础依赖，再关闭 build isolation：

```bash
pip install numpy Pillow pydantic
pip install --no-build-isolation --no-deps -e .
```

检查运行环境：

```bash
whiteboard doctor
```

## 快速开始

渲染 SVG：

```bash
whiteboard render-image tests/fixtures/apple.svg \
  -o out/apple.mp4 \
  --duration 2 \
  --fps 24 \
  --width 640 \
  --height 360 \
  --hand asian
```

渲染上传图片：

```bash
whiteboard render-photo input.jpg \
  -o out/whiteboard.mp4 \
  --duration 15 \
  --fps 30 \
  --lineart-provider auto \
  --stroke-detail rich \
  --hand asian \
  --color-fill contour-wipe
```

复现 demo：

```bash
whiteboard render-photo examples/cases/sports-illustration-anime2sketch/input.jpg \
  -o out/sports-illustration-anime2sketch-longmix-15s.mp4 \
  --duration 15 \
  --fps 30 \
  --lineart-provider anime2sketch \
  --stroke-detail rich \
  --hand asian \
  --tail-color 4.5 \
  --color-fill contour-wipe
```

如果没有安装 console script，也可以直接调用模块：

```bash
PYTHONPATH=src python3 -m whiteboard_skill.cli render-image tests/fixtures/apple.svg \
  -o out/apple.mp4 \
  --duration 2 \
  --hand asian
```

## 本地线稿模型

模型代码和权重不提交到本仓库。用户需要按照上游项目说明自行下载。

支持：

- Informative Drawings：更适合真实照片和语义线稿。
- Anime2Sketch：更适合动漫、漫画、插画和白底干净图片。

详见 [docs/MODELS.md](docs/MODELS.md)。

## CLI 概览

```bash
whiteboard extract-lineart image.jpg -o lineart.png --provider auto
whiteboard render-photo image.jpg -o output.mp4 --duration 15 --lineart-provider auto
whiteboard render-image lineart.png -o output.mp4 --source-image image.jpg --size-from-image
whiteboard analyze-image lineart.png -o analysis.json --stroke-detail rich
whiteboard list-hands
whiteboard doctor
```

常用参数：

- `--stroke-detail balanced|rich|max`
- `--hand asian|black|children|white|procedural|none`
- `--draw-text "Title"`
- `--color-fill contour-wipe|brush-scan|top-down-blocks|fade`
- `--lineart-snap-threshold 170`
- `--no-lineart-snap`

## 技术栈

核心依赖：

- Python
- Pillow
- NumPy
- Pydantic
- FFmpeg

渲染算法：

- Zhang-Suen skeletonization
- 8-neighbor stroke tracing
- endpoint-based stroke merging
- long / medium / short stroke mix
- fixed-orientation PNG hand cursor
- contour-aware color fill

可选模型推理：

- PyTorch
- torchvision
- Informative Drawings
- Anime2Sketch
- optional VTracer CLI

架构说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## Codex Skill 集成

先安装引擎，再安装 Skill：

```bash
mkdir -p ~/.codex/skills
git clone https://github.com/gnipbao/codex-whiteboard-video-skill.git \
  ~/.codex/skills/whiteboard-video
```

Skill 仓库不会 vendor `src/whiteboard_skill`，只通过 wrapper 调用已安装的 engine package。

## 版本管理规则

不要提交：

- 大量生成视频、临时输出、`out/`、`work/`
- `.venv/`、`.venv-lineart/`
- `tools/informative-drawings/`
- `tools/Anime2Sketch/`
- `*.pth`、`*.pt`、`*.ckpt`、`*.safetensors`、`*.onnx`、`*.bin`
- 未授权用户上传素材

少量 curated example 可放在 `examples/cases/`，用于公开演示。

## License

MIT. 上游模型项目和权重请分别查看各自许可证。
