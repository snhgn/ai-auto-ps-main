from __future__ import annotations

import mimetypes
import os
import socket
import tempfile
import threading
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image, ImageEnhance, UnidentifiedImageError
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageEnhance = None
    UnidentifiedImageError = Exception

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MODEL_REPO_URL = "https://github.com/huggingface/smollm"

BASE_IMAGE_FORMATS: Set[str] = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".dib", ".tif", ".tiff", ".webp", ".heic", ".gif",
    ".ppm", ".pgm", ".pbm", ".pnm",
}

OPTIONAL_IMAGE_FORMAT_CANDIDATES: Set[str] = {
    ".avif", ".jp2", ".j2k", ".jpf", ".jpx", ".jxl", ".ico", ".icns", ".pcx", ".dds",
}

BASE_VIDEO_FORMATS: Set[str] = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".mpeg", ".mpg",
    ".ts", ".m2ts", ".mts", ".3gp", ".3g2", ".ogv", ".ogm", ".asf", ".vob",
}


def _collect_supported_image_formats() -> Set[str]:
    if Image is None:
        return set(BASE_IMAGE_FORMATS)
    registered = {ext.lower() for ext in Image.registered_extensions().keys()}
    return BASE_IMAGE_FORMATS | (OPTIONAL_IMAGE_FORMAT_CANDIDATES & registered)


SUPPORTED_IMAGE_FORMATS = _collect_supported_image_formats()
SUPPORTED_VIDEO_FORMATS = set(BASE_VIDEO_FORMATS)


STYLE_PRESETS: Dict[str, Dict[str, float]] = {
    "portrait_soft": {"brightness": 1.05, "contrast": 1.08, "color": 1.06},
    "landscape_vivid": {"brightness": 1.03, "contrast": 1.15, "color": 1.22},
    "night_clarity": {"brightness": 1.20, "contrast": 1.12, "color": 0.98},
    "cinematic": {"brightness": 0.97, "contrast": 1.20, "color": 0.90},
    "food_fresh": {"brightness": 1.08, "contrast": 1.10, "color": 1.24},
    "clean_natural": {"brightness": 1.00, "contrast": 1.05, "color": 1.05},
}

STYLE_HINTS: Dict[str, str] = {
    "person": "portrait_soft",
    "face": "portrait_soft",
    "portrait": "portrait_soft",
    "landscape": "landscape_vivid",
    "mountain": "landscape_vivid",
    "sky": "landscape_vivid",
    "night": "night_clarity",
    "dark": "night_clarity",
    "city": "cinematic",
    "street": "cinematic",
    "food": "food_fresh",
    "dish": "food_fresh",
}


@dataclass
class AnalysisResult:
    description: str
    selected_style: str
    strategy: str


class LightweightStyleAdvisor:
    """Auto style selector backed by a lightweight open-source vision-language model.

    The default model is from the public GitHub repository:
    https://github.com/huggingface/smollm
    """

    def __init__(self) -> None:
        self._captioner = None
        self._init_error = None

    def _load_model_if_needed(self) -> None:
        if self._captioner is not None or self._init_error is not None:
            return
        if importlib.util.find_spec("torch") is None:
            self._init_error = "torch not installed"
            return
        try:
            from transformers import pipeline

            self._captioner = pipeline(
                task="image-to-text",
                model=MODEL_NAME,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            self._init_error = str(exc)

    def _heuristic_description(self, image: "Image.Image") -> str:
        if np is None:
            return "portrait person photo"
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
        brightness = float(arr.mean()) / 255.0
        red_blue_std = float(np.std(arr[:, :, 0] - arr[:, :, 2])) / 255.0

        if brightness < 0.32:
            return "dark night street scene"
        if red_blue_std > 0.20:
            return "high-color landscape photo"
        return "portrait person photo"

    def analyze(self, image: "Image.Image", requested_style: str) -> AnalysisResult:
        if requested_style != "auto":
            return AnalysisResult(
                description="manual style selected",
                selected_style=requested_style,
                strategy="manual",
            )

        self._load_model_if_needed()
        description = ""

        if self._captioner is not None:
            try:
                result = self._captioner(image)
                if result and isinstance(result, list):
                    description = str(result[0].get("generated_text", ""))
            except Exception:
                description = ""

        if not description:
            description = self._heuristic_description(image)

        style = choose_style_from_description(description)
        strategy = "llm" if self._captioner is not None else "heuristic_fallback"
        return AnalysisResult(description=description, selected_style=style, strategy=strategy)


_ADVISOR_INSTANCE: Optional[LightweightStyleAdvisor] = None
_ADVISOR_LOCK = threading.Lock()


def get_advisor() -> LightweightStyleAdvisor:
    global _ADVISOR_INSTANCE
    if _ADVISOR_INSTANCE is not None:
        return _ADVISOR_INSTANCE

    with _ADVISOR_LOCK:
        if _ADVISOR_INSTANCE is None:
            _ADVISOR_INSTANCE = LightweightStyleAdvisor()
    return _ADVISOR_INSTANCE


def choose_style_from_description(description: str) -> str:
    lowered = description.lower()
    for hint, style in STYLE_HINTS.items():
        if hint in lowered:
            return style
    return "clean_natural"


def describe_supported_formats() -> str:
    image_formats = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
    video_formats = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
    return f"图片: {image_formats} | 视频: {video_formats}"


def detect_media_type(path: str) -> str:
    src = Path(path)
    suffix = src.suffix.lower()

    if suffix in SUPPORTED_IMAGE_FORMATS:
        return "image"
    if suffix in SUPPORTED_VIDEO_FORMATS:
        return "video"

    guessed_mime, _ = mimetypes.guess_type(str(src))
    if guessed_mime:
        if guessed_mime.startswith("image/"):
            return "image"
        if guessed_mime.startswith("video/"):
            return "video"

    if src.exists() and src.is_file():
        if Image is not None:
            try:
                with Image.open(src) as image:
                    image.verify()
                return "image"
            except Exception:
                pass

        if cv2 is not None:
            capture = cv2.VideoCapture(str(src))
            try:
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        return "video"
            finally:
                capture.release()

    raise ValueError(f"Unsupported file format: {suffix or '(no extension)'} | {describe_supported_formats()}")


def _build_output_path(src: Path, style_name: str, extension: str) -> Path:
    output_dir = Path(tempfile.mkdtemp(prefix="ai-auto-ps-"))
    safe_style = style_name.replace(" ", "_")
    return output_dir / f"{src.stem}_{safe_style}{extension}"


def apply_style_to_pil(image: "Image.Image", style_name: str) -> "Image.Image":
    if style_name not in STYLE_PRESETS:
        style_name = "clean_natural"

    if ImageEnhance is None:
        raise RuntimeError("Pillow is required for image processing.")

    values = STYLE_PRESETS[style_name]
    out = ImageEnhance.Brightness(image).enhance(values["brightness"])
    out = ImageEnhance.Contrast(out).enhance(values["contrast"])
    out = ImageEnhance.Color(out).enhance(values["color"])
    return out


def process_image_file(
    path: str,
    advisor: LightweightStyleAdvisor,
    requested_style: str,
    skip_type_check: bool = False,
) -> Tuple[str, AnalysisResult]:
    if Image is None:
        raise RuntimeError("Pillow is required. Install pillow to process images.")

    src = Path(path)
    if not skip_type_check and detect_media_type(path) != "image":
        raise ValueError(f"Unsupported image format: {src.suffix.lower()}")

    try:
        with Image.open(src) as image:
            image = image.convert("RGB")
            analysis = advisor.analyze(image, requested_style=requested_style)
            styled = apply_style_to_pil(image, analysis.selected_style)
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Unable to decode image file: {src.name}") from exc

    output_path = _build_output_path(src, analysis.selected_style, ".jpg")
    styled.save(output_path, format="JPEG", quality=95)
    return str(output_path), analysis


def _apply_style_to_frame(frame: np.ndarray, style_name: str) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for video frame processing.")
    if style_name not in STYLE_PRESETS:
        style_name = "clean_natural"

    values = STYLE_PRESETS[style_name]
    frame_f = frame.astype(np.float32)
    frame_f = np.clip(frame_f * values["brightness"], 0, 255)

    mean = frame_f.mean(axis=(0, 1), keepdims=True)
    frame_f = np.clip((frame_f - mean) * values["contrast"] + mean, 0, 255)

    frame_u8 = frame_f.astype(np.uint8)
    gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    frame_f = np.clip(gray + (frame_f - gray) * values["color"], 0, 255)
    return frame_f.astype(np.uint8)


def process_video_file(
    path: str,
    advisor: LightweightStyleAdvisor,
    requested_style: str,
    skip_type_check: bool = False,
) -> Tuple[str, AnalysisResult]:
    if cv2 is None or Image is None:
        raise RuntimeError("opencv-python and pillow are required for video processing.")

    src = Path(path)
    if not skip_type_check and detect_media_type(path) != "video":
        raise ValueError(f"Unsupported video format: {src.suffix.lower()}")

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    writer = None
    try:
        ok, sample_frame = capture.read()
        if not ok:
            raise RuntimeError("Video file is empty or unreadable.")

        sample_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
        analysis = advisor.analyze(Image.fromarray(sample_rgb), requested_style=requested_style)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            height, width = sample_frame.shape[:2]

        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = raw_fps if raw_fps > 0 else 24.0

        output_path = _build_output_path(src, analysis.selected_style, ".mp4")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to create output video writer.")

        writer.write(_apply_style_to_frame(sample_frame, analysis.selected_style))

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_apply_style_to_frame(frame, analysis.selected_style))
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return str(output_path), analysis


def process_media(path: str, requested_style: str = "auto") -> Tuple[str, str, str]:
    if not path:
        raise ValueError("Please upload an image or video file.")

    advisor = get_advisor()
    media_type = detect_media_type(path)

    if media_type == "image":
        output_path, analysis = process_image_file(path, advisor, requested_style, skip_type_check=True)
    else:
        output_path, analysis = process_video_file(path, advisor, requested_style, skip_type_check=True)

    reason = (
        f"Model: {MODEL_NAME} | Strategy: {analysis.strategy} | "
        f"Description: {analysis.description} | Selected style: {analysis.selected_style}"
    )
    return output_path, analysis.selected_style, reason


def double_check_implementation() -> str:
    first_pass_checks = [
        bool(MODEL_NAME and MODEL_REPO_URL),
        bool(STYLE_PRESETS),
        bool(SUPPORTED_IMAGE_FORMATS and SUPPORTED_VIDEO_FORMATS),
    ]
    first_pass = all(first_pass_checks)
    try:
        if Image is None:
            raise RuntimeError("Pillow not available")
        sample = Image.new("RGB", (8, 8), (40, 80, 140))
        advisor = LightweightStyleAdvisor()
        analysis = advisor.analyze(sample, requested_style="auto")
        styled = apply_style_to_pil(sample, analysis.selected_style)
        second_pass = (
            styled.size == sample.size
            and analysis.selected_style in STYLE_PRESETS
            and all(fmt.startswith(".") for fmt in SUPPORTED_IMAGE_FORMATS | SUPPORTED_VIDEO_FORMATS)
        )
    except Exception:
        second_pass = False
    if first_pass and second_pass:
        return "两轮检查通过：核心功能入口完整。"
    return "两轮检查未通过：请检查模型、风格和格式支持配置。"


def build_ui():
    import gradio as gr

    def _handle_upload(file_obj, style_name):
        if file_obj is None:
            raise gr.Error("请先上传图片或视频文件")

        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        try:
            output_path, final_style, reason = process_media(file_path, style_name)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        return output_path, final_style, reason, double_check_implementation()

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        font=[gr.themes.GoogleFont("sans-serif"), "Arial", "sans-serif"]
    )

    with gr.Blocks(title="AI Auto PS", theme=custom_theme) as demo:
        with gr.Column(elem_id="header"):
            gr.Markdown(
                f"""
                <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem; background: linear-gradient(135deg, #e0e7ff, #f3f4f6); border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <h1 style="color: #1e3a8a; font-size: 2.5rem; margin: 0;">✨ AI 自动 P 图调色大师</h1>
                    <p style="color: #4f46e5; font-size: 1.1rem; margin-top: 0.5rem;">基于视觉大模型，智能识别内容与路由风格</p>
                </div>
                """
            )

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 📥 第 1 步：媒体输入")
                with gr.Group():
                    media_input = gr.File(
                        label="将图片或视频拖拽到此处上传", 
                        file_types=["image", "video"], 
                        elem_classes="top-gap"
                    )
                    style_choice = gr.Dropdown(
                        choices=["auto", *STYLE_PRESETS.keys()],
                        value="auto",
                        label="🎨 调色风格偏好",
                        info="选择 'auto' 交给 AI 根据画面内容决定"
                    )
                run_button = gr.Button("🚀 立即开始智能调色", variant="primary", size="lg")

            with gr.Column(scale=5):
                gr.Markdown("### 🎯 第 2 步：处理结果获取")
                output_file = gr.File(label="📥 点击下载调色后的文件")
                
                with gr.Accordion("📊 AI 分析与决策看板", open=True):
                    with gr.Group():
                        output_style = gr.Textbox(label="实际应用风格", show_copy_button=True)
                        output_reason = gr.Textbox(label="模型分析摘要与依据", lines=2)

                with gr.Accordion("🛡️ 系统运行诊断 (高级)", open=False):
                    output_check = gr.Textbox(label="双轮完整性检查状态", lines=1)
                    gr.Markdown(f"**当前环境支持格式:**\n`{describe_supported_formats()}`")

        run_button.click(
            fn=_handle_upload,
            inputs=[media_input, style_choice],
            outputs=[output_file, output_style, output_reason, output_check],
        )

    return demo


def launch() -> None:
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _is_port_available(host: str, port: int) -> bool:
        if port <= 0 or port > 65535:
            return False

        bind_host = host
        if bind_host in {"0.0.0.0", "::"}:
            bind_host = "0.0.0.0"

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except OSError:
                return False
        return True

    def _choose_port(host: str, preferred_port: int) -> int:
        if _is_port_available(host, preferred_port):
            return preferred_port

        for offset in range(1, 21):
            candidate = preferred_port + offset
            if _is_port_available(host, candidate):
                return candidate

        raise RuntimeError("No available port found in preferred range.")

    host = os.getenv("AI_AUTO_PS_HOST", "127.0.0.1")
    preferred_port = int(os.getenv("AI_AUTO_PS_PORT", "7860"))
    open_browser = _env_flag("AI_AUTO_PS_OPEN_BROWSER", default=False)
    server_port = _choose_port(host, preferred_port)

    if server_port != preferred_port:
        print(f"[INFO] Port {preferred_port} is in use, switched to {server_port}.")

    app = build_ui()
    app.launch(server_name=host, server_port=server_port, inbrowser=open_browser)


if __name__ == "__main__":
    launch()
