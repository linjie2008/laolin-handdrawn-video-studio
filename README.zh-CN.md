<p align="center">
  <img src="docs/assets/hero.png" alt="白板手绘视频引擎" width="960">
</p>

# 白板手绘视频引擎

[English](README.md)

一个本地优先的白板手绘视频引擎，可将 SVG、线稿图、插画和照片转换为逐笔绘制的 MP4 视频。

本仓库专注于底层渲染能力：语义线稿输入、笔画追踪、路径排序、手势跟随和轮廓感上色。Codex Skill 独立维护在 [gnipbao/codex-whiteboard-video-skill](https://github.com/gnipbao/codex-whiteboard-video-skill)。

## 核心能力

- 支持 SVG 和栅格线稿逐笔绘制。
- 支持本地神经网络线稿提取，适配照片、插画和动漫图。
- 支持骨架追踪、路径平滑和短线合并。
- 内置固定角度手势：`asian`、`black`、`children`、`white`。
- 支持 `--draw-text` 将短标题转换为手写路径。
- 支持基于原图的轮廓感上色。
- CLI 优先，方便脚本化、自动化和 Codex 集成。

## 效果演示

<table>
  <tr>
    <td width="50%">
      <strong>输入图</strong><br>
      <img src="examples/cases/sports-illustration-anime2sketch/input.jpg" alt="输入插画" width="360">
    </td>
    <td width="50%">
      <strong>输出预览</strong><br>
      <a href="examples/cases/sports-illustration-anime2sketch/output.mp4">
        <img src="examples/cases/sports-illustration-anime2sketch/output-preview.gif" alt="白板动画预览" width="360">
      </a><br>
      <a href="examples/cases/sports-illustration-anime2sketch/output.mp4">查看 MP4</a>
    </td>
  </tr>
</table>

后续案例可继续放入 `examples/cases/<case-name>/`。

## 安装

```bash
python3 -m pip install "git+https://github.com/gnipbao/whiteboard-video-engine.git"
```

本地开发：

```bash
git clone https://github.com/gnipbao/whiteboard-video-engine.git
cd whiteboard-video-engine
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

检查环境：

```bash
whiteboard doctor
```

## 快速开始

渲染照片或插画：

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

渲染已有 SVG 或线稿图：

```bash
whiteboard render-image lineart.png \
  -o out/whiteboard.mp4 \
  --source-image input.jpg \
  --source-fit exact \
  --duration 15 \
  --fps 30
```

复现内置案例：

```bash
whiteboard render-photo examples/cases/sports-illustration-anime2sketch/input.jpg \
  -o out/sports-illustration-anime2sketch.mp4 \
  --duration 15 \
  --fps 30 \
  --lineart-provider anime2sketch \
  --stroke-detail rich \
  --hand asian \
  --tail-color 4.5 \
  --color-fill contour-wipe
```

## 命令行

```bash
whiteboard extract-lineart image.jpg -o lineart.png --provider auto
whiteboard render-photo image.jpg -o output.mp4 --duration 15 --lineart-provider auto
whiteboard render-image lineart.png -o output.mp4 --source-image image.jpg --source-fit exact
whiteboard analyze-image lineart.png -o analysis.json --stroke-detail rich
whiteboard list-hands
whiteboard doctor
```

常用参数：

- `--stroke-detail balanced|rich|max`
- `--hand asian|black|children|white|procedural|none`
- `--draw-text "标题"`
- `--color-fill contour-wipe|brush-scan|top-down-blocks|fade`
- `--lineart-provider auto|informative|anime2sketch`

## 线稿模型

本仓库不提交模型仓库和模型权重。用户需要本地安装模型，并将 wrapper 放在 `tools/lineart/` 下。

支持的线稿模型：

- [Informative Drawings](https://github.com/carolineec/informative-drawings)：适合照片和语义线稿。
- [Anime2Sketch](https://github.com/Mukosame/Anime2Sketch)：适合动漫、漫画和白底插画。

模型路径、环境变量和 wrapper 命令见 [docs/MODELS.md](docs/MODELS.md)。

## 架构

```text
原图 / SVG
  -> 本地线稿模型
  -> 骨架提取 / SVG 路径解析
  -> 笔画排序与路径平滑
  -> 手势跟随渲染
  -> 轮廓感上色
  -> FFmpeg 输出 MP4
```

核心依赖：

- Python、Pillow、NumPy、Pydantic
- FFmpeg
- 可选 PyTorch 线稿模型栈

架构细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## Codex Skill

安装引擎后，可继续安装配套 Skill：

```bash
mkdir -p ~/.codex/skills
git clone https://github.com/gnipbao/codex-whiteboard-video-skill.git \
  ~/.codex/skills/whiteboard-video
```

Skill 仓库只包含 Codex 指令和 wrapper 脚本，渲染能力以本仓库为准。

## 案例库

| 案例 | 线稿模型 | 说明 |
| --- | --- | --- |
| `sports-illustration-anime2sketch` | Anime2Sketch | 白底插画、丰富笔画、轮廓感上色 |

新增案例建议使用：

```text
examples/cases/<case-name>/
  README.md
  input.jpg
  output-preview.gif
  output.mp4
```

## 仓库边界

不要提交模型仓库、模型权重、虚拟环境、生成过程目录，或没有分发授权的用户上传素材。

少量精选演示素材可放在 `examples/cases/`。

## 许可证

MIT。上游模型代码和权重遵循各自许可证。
