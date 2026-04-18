# ai-auto-ps

一个基于轻量化开源视觉大模型的 AI 自动 P 图/调色工具，支持常见图片和视频格式，并提供图形化操作界面。

## 已实现功能

- 选用开源轻量模型：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  - 开源仓库：<https://github.com/huggingface/smollm>
- 自动内容分析与风格路由
  - 模型可用时走模型推理
  - 模型不可用时自动回退到本地启发式分析，保证流程可用
- 预设多种修图调色风格
  - `portrait_soft`
  - `landscape_vivid`
  - `night_clarity`
  - `cinematic`
  - `food_fresh`
  - `clean_natural`
- 支持常见图片格式（基础）
  - `jpg/jpeg/jpe/jfif/png/bmp/dib/tif/tiff/webp/heic/gif/ppm/pgm/pbm/pnm`
- 支持常见视频格式（基础）
  - `mp4/mov/mkv/avi/webm/m4v/flv/wmv/mpeg/mpg/ts/m2ts/mts/3gp/3g2/ogv/ogm/asf/vob`
- 支持环境可选图片格式（按 Pillow 编解码能力自动启用）
  - `avif/jp2/j2k/jpf/jpx/jxl/ico/icns/pcx/dds`
- 图形化界面（Gradio）
  - 上传图片/视频
  - 自动/手动风格选择
  - 输出处理结果、风格决策依据、双轮完整性检查结果

## 当前任务流程

1. 上传媒体文件
2. 媒体类型识别（扩展名 -> MIME -> 解码探测）
3. 风格路由（手动风格直通；自动风格走模型/启发式）
4. 应用调色并输出文件
5. 双轮完整性检查并返回结果

## 流程优化说明

- 模型顾问实例复用：避免每次请求重复初始化模型，提升连续处理流畅性。
- 媒体识别链路统一：先快速判断扩展名，再回退 MIME 与解码探测，降低误判。
- 视频处理资源安全释放：写入器与读取器统一在 `finally` 中释放，减少中途异常时资源泄漏风险。
- 格式扩展机制升级：基础格式稳定支持，可选格式根据运行环境自动增量启用。

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ai_auto_ps.py
```

打开浏览器访问：<http://127.0.0.1:7860>

## Windows 一键启动

项目根目录已提供一键启动脚本：`start_gui.bat`

使用方式：

1. 在资源管理器中双击 `start_gui.bat`
2. 首次运行会自动创建 `.venv` 并安装依赖
3. 服务就绪后自动打开浏览器（默认地址：<http://127.0.0.1:7860>）
4. 若默认端口被占用，程序会自动切换到后续可用端口并在终端提示

可选：你也可以先设置环境变量自定义地址和端口，再运行脚本。

- `AI_AUTO_PS_HOST`（默认 `127.0.0.1`）
- `AI_AUTO_PS_PORT`（默认 `7860`）
- `AI_AUTO_PS_OPEN_BROWSER`（默认 `1`，可选 `0/1`）

## 双轮完整性检查

应用内置 `double_check_implementation()`，每次处理后都会进行两轮检查：

1. 第一轮（配置检查）
   - 是否已配置轻量开源模型
   - 是否已配置多风格预设
   - 是否同时具备图片与视频格式支持
2. 第二轮（运行时检查）
   - 运行一个最小样例图像的分析与调色流程
   - 校验返回风格与预设一致
   - 再次校验格式声明规范

两轮都通过才返回“检查通过”。
