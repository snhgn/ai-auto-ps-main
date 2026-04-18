from __future__ import annotations

import mimetypes
import os
import socket
import tempfile
import threading
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
    "portrait_soft": {
        "brightness": 1.05,
        "contrast": 1.08,
        "color": 1.06,
        "skin_smooth": 0.52,
        "slim_face": 0.38,
        "skin_smooth_video": 0.32,
        "slim_face_video": 0.22,
    },
    "landscape_vivid": {
        "brightness": 1.03,
        "contrast": 1.15,
        "color": 1.22,
        "skin_smooth": 0.00,
        "slim_face": 0.00,
    },
    "night_clarity": {
        "brightness": 1.20,
        "contrast": 1.12,
        "color": 0.98,
        "skin_smooth": 0.12,
        "slim_face": 0.00,
        "skin_smooth_video": 0.10,
    },
    "cinematic": {
        "brightness": 0.97,
        "contrast": 1.20,
        "color": 0.90,
        "skin_smooth": 0.18,
        "slim_face": 0.10,
        "skin_smooth_video": 0.12,
        "slim_face_video": 0.06,
    },
    "food_fresh": {
        "brightness": 1.08,
        "contrast": 1.10,
        "color": 1.24,
        "skin_smooth": 0.00,
        "slim_face": 0.00,
    },
    "clean_natural": {
        "brightness": 1.00,
        "contrast": 1.05,
        "color": 1.05,
        "skin_smooth": 0.10,
        "slim_face": 0.05,
        "skin_smooth_video": 0.06,
        "slim_face_video": 0.03,
    },
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
_FACE_CASCADE = None
_FACE_CASCADE_READY = False
_FACE_CASCADE_LOCK = threading.Lock()


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
    out = _apply_advanced_retouch_to_pil(out, style_name)
    return out


def _get_face_cascade():
    global _FACE_CASCADE
    global _FACE_CASCADE_READY

    if _FACE_CASCADE_READY:
        return _FACE_CASCADE

    with _FACE_CASCADE_LOCK:
        if _FACE_CASCADE_READY:
            return _FACE_CASCADE

        cascade = None
        if cv2 is not None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                loaded = cv2.CascadeClassifier(cascade_path)
                if loaded is not None and not loaded.empty():
                    cascade = loaded
            except Exception:
                cascade = None

        _FACE_CASCADE = cascade
        _FACE_CASCADE_READY = True

    return _FACE_CASCADE


def _apply_skin_smoothing_bgr(frame: np.ndarray, strength: float) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = float(np.clip(strength, 0.0, 1.0))
    diameter = max(5, int(7 + strength * 8))
    sigma = 25 + strength * 70

    smooth = cv2.bilateralFilter(frame, d=diameter, sigmaColor=sigma, sigmaSpace=sigma)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    skin_mask = cv2.GaussianBlur(skin_mask, (0, 0), sigmaX=2.2, sigmaY=2.2).astype(np.float32) / 255.0

    blend = np.clip((0.20 + 0.60 * strength) * skin_mask, 0.0, 1.0)[..., None]
    out = frame.astype(np.float32) * (1.0 - blend) + smooth.astype(np.float32) * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def _slim_face_region_bgr(region: np.ndarray, strength: float) -> np.ndarray:
    if np is None or cv2 is None:
        return region

    h, w = region.shape[:2]
    if h < 24 or w < 24:
        return region

    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)

    radial = np.sqrt(nx * nx + (ny * 1.25) * (ny * 1.25))
    influence = np.clip(1.0 - radial, 0.0, 1.0)
    max_shift = strength * w * 0.14

    map_x = np.clip(xx + nx * influence * max_shift, 0, w - 1).astype(np.float32)
    map_y = yy.astype(np.float32)
    return cv2.remap(region, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _apply_face_slimming_bgr(frame: np.ndarray, strength: float) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = float(np.clip(strength, 0.0, 1.0))
    cascade = _get_face_cascade()
    if cascade is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(56, 56))
    if faces is None or len(faces) == 0:
        return frame

    out = frame.copy()
    for x, y, w, h in faces:
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.20)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(out.shape[1], x + w + pad_x)
        y1 = min(out.shape[0], y + h + pad_y)
        out[y0:y1, x0:x1] = _slim_face_region_bgr(out[y0:y1, x0:x1], strength)

    return out


def _apply_advanced_retouch_to_pil(image: "Image.Image", style_name: str) -> "Image.Image":
    if Image is None or np is None or cv2 is None:
        return image

    values = STYLE_PRESETS.get(style_name, STYLE_PRESETS["clean_natural"])
    smooth_strength = float(values.get("skin_smooth", 0.0))
    slim_strength = float(values.get("slim_face", 0.0))

    if smooth_strength <= 0 and slim_strength <= 0:
        return image

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _apply_skin_smoothing_bgr(bgr, smooth_strength)
    bgr = _apply_face_slimming_bgr(bgr, slim_strength)

    result_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


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

    out = frame_f.astype(np.uint8)
    smooth_strength = float(values.get("skin_smooth_video", values.get("skin_smooth", 0.0) * 0.6))
    slim_strength = float(values.get("slim_face_video", values.get("slim_face", 0.0) * 0.6))
    out = _apply_skin_smoothing_bgr(out, smooth_strength)
    out = _apply_face_slimming_bgr(out, slim_strength)
    return out


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


def normalize_uploaded_file_paths(file_obj: Any) -> List[str]:
    if file_obj is None:
        return []

    if isinstance(file_obj, (list, tuple)):
        file_items = file_obj
    else:
        file_items = [file_obj]

    paths: List[str] = []
    for item in file_items:
        if item is None:
            continue
        path = item.name if hasattr(item, "name") else str(item)
        if path:
            paths.append(path)
    return paths


def process_uploaded_files(
    file_paths: Sequence[str],
    requested_style: str = "auto",
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]], str, str, str]:
    if not file_paths:
        raise ValueError("请先上传图片或视频文件")

    advisor = get_advisor()
    image_paths: List[str] = []
    video_paths: List[str] = []

    for path in file_paths:
        media_type = detect_media_type(path)
        if media_type == "image":
            image_paths.append(path)
        else:
            video_paths.append(path)

    if video_paths and len(file_paths) > 1:
        raise ValueError("批量上传模式目前仅支持图片；视频请单独上传一个文件处理。")

    if video_paths:
        output_path, analysis = process_video_file(
            video_paths[0],
            advisor,
            requested_style,
            skip_type_check=True,
        )
        reason = (
            f"Model: {MODEL_NAME} | Strategy: {analysis.strategy} | "
            f"Description: {analysis.description} | Selected style: {analysis.selected_style}"
        )
        return [output_path], [], [], analysis.selected_style, reason, double_check_implementation()

    output_paths: List[str] = []
    before_gallery: List[Tuple[str, str]] = []
    after_gallery: List[Tuple[str, str]] = []
    style_lines: List[str] = []
    reason_lines: List[str] = []

    for path in image_paths:
        output_path, analysis = process_image_file(
            path,
            advisor,
            requested_style,
            skip_type_check=True,
        )
        src_name = Path(path).name

        output_paths.append(output_path)
        before_gallery.append((path, f"原图: {src_name}"))
        after_gallery.append((output_path, f"结果: {src_name} -> {analysis.selected_style}"))
        style_lines.append(f"{src_name}: {analysis.selected_style}")
        reason_lines.append(
            f"{src_name} | strategy={analysis.strategy} | description={analysis.description}"
        )

    return (
        output_paths,
        before_gallery,
        after_gallery,
        "\n".join(style_lines),
        "\n".join(reason_lines),
        double_check_implementation(),
    )


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
        file_paths = normalize_uploaded_file_paths(file_obj)
        if not file_paths:
            raise gr.Error("请先上传图片或视频文件")

        try:
            result = process_uploaded_files(file_paths, style_name)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        return result

    with gr.Blocks(title="AI Auto PS") as demo:
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
                        label="支持一次上传多张图片（视频请单独上传）",
                        file_types=["image", "video"],
                        file_count="multiple",
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
                output_files = gr.File(label="📥 点击下载调色后的文件", file_count="multiple")

                with gr.Row():
                    before_preview = gr.Gallery(label="处理前预览", columns=3, rows=1, height="auto")
                    after_preview = gr.Gallery(label="处理后预览", columns=3, rows=1, height="auto")
                
                with gr.Accordion("📊 AI 分析与决策看板", open=True):
                    with gr.Group():
                        output_style = gr.Textbox(label="实际应用风格", lines=4)
                        output_reason = gr.Textbox(label="模型分析摘要与依据", lines=6)

                with gr.Accordion("🛡️ 系统运行诊断 (高级)", open=False):
                    output_check = gr.Textbox(label="双轮完整性检查状态", lines=1)
                    gr.Markdown(f"**当前环境支持格式:**\n`{describe_supported_formats()}`")

        run_button.click(
            fn=_handle_upload,
            inputs=[media_input, style_choice],
            outputs=[output_files, before_preview, after_preview, output_style, output_reason, output_check],
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
