<p align="center">
  <img src="docs/assets/hero.png" alt="白板手绘视频生成器" width="960">
</p>

# 白板手绘视频生成器

[English](README.en.md)

一个面向水墨画、涂鸦、普通插画和已有线稿的本地手绘视频创作工具。上传图片后，系统会自动处理背景、提取线稿与颜色，并按照真实绘画顺序生成包含角色运笔、自然上色、主题书写和盖章动作的 MP4 视频。

项目以可视化前端为主要入口，不需要手动组合复杂命令。所有识别与渲染均可在本地完成。

## 最新水墨演示

<table>
  <tr>
    <td width="50%">
      <strong>水墨原图</strong><br>
      <img src="examples/cases/ink-wash-ancient-town/input.png" alt="古城水墨原图" width="360">
    </td>
    <td width="50%">
      <strong>完整绘制过程</strong><br>
      <a href="examples/cases/ink-wash-ancient-town/output.mp4">
        <img src="examples/cases/ink-wash-ancient-town/output-preview.gif" alt="古城水墨绘制动画预览" width="360">
      </a><br>
      <a href="examples/cases/ink-wash-ancient-town/output.mp4">查看 19 秒 MP4</a>
      · <a href="examples/cases/ink-wash-ancient-town/lineart.png">查看提取线稿</a>
    </td>
  </tr>
</table>

## 核心流程

```text
上传或粘贴图片
  -> 自动转换纯白背景
  -> 选择处理类型并提取线稿/颜色
  -> 检查识别预览
  -> 设置绘制角色、上色、主题与印章
  -> 生成并预览 MP4
```

未完成线稿提取时不能生成视频，避免使用过期或不匹配的识别结果。

## 主要功能

### 图片识别

- **普通线稿**：适合人物、动漫、插画和常规照片。
- **水墨线稿**：识别浓墨、中墨、浅墨、飞白、墨块和晕染边缘。
- **涂鸦作品**：保留原图中的复杂颜色区域，为后续上色提供依据。
- **已经是线稿**：规范化现有线稿，同时尽量保持原始线宽和笔触风格。
- 上传后自动识别，非白色背景自动处理为纯白色。

### 真人感绘制

- 按局部连续性排序笔画，减少跨区域乱跳。
- 绘制过程严格跟随已提取线稿，完成后再进行轻微对齐修复。
- 墨块只在工具经过的位置逐步出现，支持直接铺墨和自然扩散感。
- 线宽自动适配原线稿，减少视频线条粗糙、悬浮或骨骼化的问题。

### 绘制角色与工具

- 猪八戒：步行入场并挥舞钉耙，当前默认工具。
- 孙悟空：筋斗云入场并动态绘制。
- 唐僧、诸葛亮：步行入场。
- 关羽：骑马入场并持刀绘制。
- 鹅毛笔、大公鸡羽毛笔、毛笔和品牌特色工具。
- 绘制工具与上色工具默认自动保持一致，也可以单独覆盖上色工具。

### 自然上色

- 根据原图与线稿差异识别需要补色的区域。
- 根据颜色复杂度自动建议上色时长。
- 支持自然修复、自然涂鸦、笔触跟随、轮廓渐染、分块填充等方式。
- 水墨、涂鸦和普通线稿使用不同的颜色恢复策略。
- 上色过程逐步呈现，最后阶段平滑过渡到原图，避免成品突然出现。

### 主题与印章

- 盖章前可书写自定义主题，支持手动换行。
- 提供毛体与草书、字体预览、字号和位置设置。
- 印章文字默认“老林涂鸦”，内容和位置均可自定义。
- 支持白文印、复古玉玺、水墨印等风格。
- 盖章采用垂直方向的立体动作：先由印章实物遮挡，按压后抬起并显示印迹。

### 视频输出

- 支持原图尺寸、横屏和竖屏预设。
- 视频时长 5–120 秒，帧率 15–60 FPS。
- 生成进度按 20% 阶段变色显示。
- 视频窗口、状态和进度集中显示，生成期间保持稳定，不反复闪烁。

## 安装

环境要求：

- Python 3.11+
- FFmpeg（需要包含 `libx264`）
- 推荐使用支持 PyTorch 的本地环境运行神经网络线稿模型

```bash
git clone https://github.com/linjie2008/whiteboard-video-engine.git
cd whiteboard-video-engine

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,lineart]"
```

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

## 启动前端

```bash
python app.py
```

浏览器访问：

```text
http://127.0.0.1:7860
```

## 线稿模型

本项目支持以下本地识别方式：

- Anime2Sketch：动漫、人物和白底插画。
- Informative Drawings：照片和物体轮廓。
- 水墨专用 CV 流程：墨色分层、飞白和墨块识别。
- 涂鸦全色识别：复杂色块与调色板提取。

第三方模型仓库和权重不会提交到 Git。安装位置与环境变量配置见 [docs/MODELS.md](docs/MODELS.md)。

## 命令行

前端是推荐入口，同时保留 CLI 供批处理和自动化调用：

```bash
whiteboard doctor
whiteboard extract-lineart input.png -o lineart.png --provider ink-wash
whiteboard render-photo input.png -o output.mp4 --lineart-provider ink-wash
whiteboard render-image lineart.png -o output.mp4 --source-image input.png
```

## 项目结构

```text
app.py                         Gradio 前端与任务状态管理
src/whiteboard_skill/          线稿、路径、上色和视频渲染核心
tools/lineart/                 水墨、涂鸦及第三方模型适配器
assets/cursors/                绘制角色与工具动画素材
assets/stamps/                 印章实物与印迹素材
assets/fonts/                  主题书写字体
examples/cases/                可公开复现的图片、线稿、GIF 和 MP4
tests/                         前端状态与渲染回归测试
```

## 测试

```bash
python -m pytest -q
```

## 示例素材

`examples/cases/` 中的精选案例会同时保留输入图片、动态预览和 MP4。运行过程中产生的临时文件仍保存在 `out/`，不会上传到 GitHub。

## 开源致谢

本项目在 [@gnipbao](https://github.com/gnipbao) 开源的
[whiteboard-video-engine](https://github.com/gnipbao/whiteboard-video-engine) 基础上持续开发，并围绕可视化工作流、水墨与涂鸦识别、角色动作、自然上色、主题书写和立体盖章进行了扩展。感谢原作者及相关开源模型作者。

## 许可证

项目代码采用 MIT License。第三方模型、字体和素材遵循各自许可证。
