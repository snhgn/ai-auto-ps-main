from __future__ import annotations

import mimetypes
import os
import socket
import tempfile
import threading
import importlib.util
import uuid
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

try:
    from multi_solution_generator import (
        EnhancedAnalysisResult,
        SolutionVariant,
        generate_multiple_solutions,
        get_solution_by_name,
        solutions_to_ui_format,
    )
    HAS_MULTI_SOLUTION = True
except ImportError:
    HAS_MULTI_SOLUTION = False

try:
    from solution_manager import (
        SolutionSession,
        UserFeedback,
        get_manager,
    )
    HAS_SOLUTION_MANAGER = True
except ImportError:
    HAS_SOLUTION_MANAGER = False


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
        "skin_smooth": 0.58,
        "skin_whiten": 0.38,
        "acne_remove": 0.42,
        "blush": 0.24,
        "eye_brighten": 0.30,
        "lip_tint": 0.22,
        "slim_face": 0.40,
        "eye_enlarge": 0.28,
        "nose_slim": 0.24,
        "chin_refine": 0.18,
        "skin_smooth_video": 0.32,
        "skin_whiten_video": 0.20,
        "acne_remove_video": 0.22,
        "blush_video": 0.14,
        "eye_brighten_video": 0.14,
        "lip_tint_video": 0.10,
        "slim_face_video": 0.22,
        "eye_enlarge_video": 0.12,
        "nose_slim_video": 0.10,
        "chin_refine_video": 0.08,
    },
    "landscape_vivid": {
        "brightness": 1.03,
        "contrast": 1.15,
        "color": 1.22,
        "skin_whiten": 0.00,
        "acne_remove": 0.00,
        "blush": 0.00,
        "eye_brighten": 0.00,
        "lip_tint": 0.00,
        "skin_smooth": 0.00,
        "slim_face": 0.00,
        "eye_enlarge": 0.00,
        "nose_slim": 0.00,
        "chin_refine": 0.00,
    },
    "night_clarity": {
        "brightness": 1.20,
        "contrast": 1.12,
        "color": 0.98,
        "skin_smooth": 0.18,
        "skin_whiten": 0.16,
        "acne_remove": 0.14,
        "blush": 0.08,
        "eye_brighten": 0.20,
        "lip_tint": 0.06,
        "slim_face": 0.00,
        "eye_enlarge": 0.05,
        "nose_slim": 0.00,
        "chin_refine": 0.00,
        "skin_smooth_video": 0.10,
        "skin_whiten_video": 0.10,
        "acne_remove_video": 0.08,
        "eye_brighten_video": 0.12,
    },
    "cinematic": {
        "brightness": 0.97,
        "contrast": 1.20,
        "color": 0.90,
        "skin_smooth": 0.18,
        "skin_whiten": 0.10,
        "acne_remove": 0.12,
        "blush": 0.10,
        "eye_brighten": 0.12,
        "lip_tint": 0.10,
        "slim_face": 0.10,
        "eye_enlarge": 0.08,
        "nose_slim": 0.08,
        "chin_refine": 0.06,
        "skin_smooth_video": 0.12,
        "skin_whiten_video": 0.08,
        "acne_remove_video": 0.08,
        "blush_video": 0.06,
        "eye_brighten_video": 0.08,
        "lip_tint_video": 0.06,
        "slim_face_video": 0.06,
        "eye_enlarge_video": 0.04,
        "nose_slim_video": 0.04,
        "chin_refine_video": 0.04,
    },
    "food_fresh": {
        "brightness": 1.08,
        "contrast": 1.10,
        "color": 1.24,
        "skin_whiten": 0.00,
        "acne_remove": 0.00,
        "blush": 0.00,
        "eye_brighten": 0.00,
        "lip_tint": 0.00,
        "skin_smooth": 0.00,
        "slim_face": 0.00,
        "eye_enlarge": 0.00,
        "nose_slim": 0.00,
        "chin_refine": 0.00,
    },
    "clean_natural": {
        "brightness": 1.00,
        "contrast": 1.05,
        "color": 1.05,
        "skin_smooth": 0.18,
        "skin_whiten": 0.12,
        "acne_remove": 0.16,
        "blush": 0.10,
        "eye_brighten": 0.12,
        "lip_tint": 0.08,
        "slim_face": 0.08,
        "eye_enlarge": 0.08,
        "nose_slim": 0.06,
        "chin_refine": 0.06,
        "skin_smooth_video": 0.06,
        "skin_whiten_video": 0.05,
        "acne_remove_video": 0.06,
        "blush_video": 0.04,
        "eye_brighten_video": 0.05,
        "lip_tint_video": 0.03,
        "slim_face_video": 0.03,
        "eye_enlarge_video": 0.02,
        "nose_slim_video": 0.02,
        "chin_refine_video": 0.02,
    },
}

RETOUCH_CONTROL_KEYS: Tuple[str, ...] = (
    "skin_smooth",
    "skin_whiten",
    "acne_remove",
    "blush",
    "eye_brighten",
    "lip_tint",
    "slim_face",
    "eye_enlarge",
    "nose_slim",
    "chin_refine",
)

RETOUCH_CONTROL_LABELS: Dict[str, str] = {
    "skin_smooth": "磨皮",
    "skin_whiten": "美白",
    "acne_remove": "祛痘",
    "blush": "红润",
    "eye_brighten": "亮眼",
    "lip_tint": "唇色",
    "slim_face": "瘦脸",
    "eye_enlarge": "大眼",
    "nose_slim": "窄鼻",
    "chin_refine": "下巴",
}

RETOUCH_PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "自然韩系": {
        "skin_smooth": 0.62,
        "skin_whiten": 0.34,
        "acne_remove": 0.45,
        "blush": 0.28,
        "eye_brighten": 0.32,
        "lip_tint": 0.28,
        "slim_face": 0.36,
        "eye_enlarge": 0.30,
        "nose_slim": 0.22,
        "chin_refine": 0.20,
    },
    "清透日系": {
        "skin_smooth": 0.48,
        "skin_whiten": 0.26,
        "acne_remove": 0.32,
        "blush": 0.16,
        "eye_brighten": 0.25,
        "lip_tint": 0.15,
        "slim_face": 0.20,
        "eye_enlarge": 0.18,
        "nose_slim": 0.10,
        "chin_refine": 0.10,
    },
    "轻欧式": {
        "skin_smooth": 0.40,
        "skin_whiten": 0.18,
        "acne_remove": 0.22,
        "blush": 0.18,
        "eye_brighten": 0.22,
        "lip_tint": 0.20,
        "slim_face": 0.24,
        "eye_enlarge": 0.22,
        "nose_slim": 0.18,
        "chin_refine": 0.16,
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

NIGHT_BRIGHTNESS_THRESHOLD = 0.28
BRIGHT_LANDSCAPE_THRESHOLD = 0.75
AUTO_CROP_DEFAULT_FACTOR = 1.0

PREFERENCE_BRIGHTNESS_LOWER_KEYWORDS = ("太亮", "过曝", "刺眼", "too bright")
PREFERENCE_BRIGHTNESS_HIGHER_KEYWORDS = ("太暗", "偏暗", "提亮", "too dark")
PREFERENCE_SATURATION_LOWER_KEYWORDS = ("太艳", "太饱和", "过饱和", "oversaturated")
PREFERENCE_SATURATION_HIGHER_KEYWORDS = ("不够鲜艳", "太灰", "发灰", "desaturated")

NEGATIVE_STYLE_INTENSITY_MULTIPLIER = 0.88
NEGATIVE_STYLE_MIN_INTENSITY = 0.45
NEGATIVE_STYLE_CONTRAST_MULTIPLIER = 0.92
NEGATIVE_STYLE_CONTRAST_MIN = 1.0
NEGATIVE_STYLE_COLOR_MULTIPLIER = 0.90
NEGATIVE_STYLE_COLOR_MIN = 0.85


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
            
            # 尝试加载更强的模型（如果启用）
            use_large = os.getenv("AI_AUTO_PS_USE_LARGE_MODEL", "0").lower() in {"1", "true", "yes"}
            model_name = "llava-hf/llava-1.5-13b-hf" if use_large else MODEL_NAME
            
            try:
                self._captioner = pipeline(
                    task="image-to-text",
                    model=model_name,
                )
                self._model_type = "llava" if use_large else "smolvlm2"
            except Exception:
                # Fallback to SmolVLM2 if LLaVA not available
                self._captioner = pipeline(
                    task="image-to-text",
                    model=MODEL_NAME,
                )
                self._model_type = "smolvlm2"
                
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

    def analyze(self, image: "Image.Image", requested_style: str) -> AnalysisResult | EnhancedAnalysisResult:
        if requested_style != "auto":
            result = AnalysisResult(
                description="manual style selected",
                selected_style=requested_style,
                strategy="manual",
            )
            if HAS_MULTI_SOLUTION:
                return _convert_to_enhanced(result)
            return result

        self._load_model_if_needed()
        primary_description = ""
        secondary_description = self._heuristic_description(image)

        if self._captioner is not None:
            try:
                result = self._captioner(image)
                if result and isinstance(result, list):
                    primary_description = _sanitize_analysis_text(
                        _extract_text_from_model_output(result[0].get("generated_text", ""))
                    )
            except Exception:
                primary_description = ""

        if not primary_description:
            primary_description = secondary_description

        primary_style = choose_style_from_description(primary_description)
        secondary_style = choose_style_from_description(secondary_description)
        style = _merge_collaborative_style(primary_style, secondary_style, image)
        strategy = "dual_model_collaboration"
        combined_description = f"primary={primary_description} || secondary={secondary_description}"

        basic_result = AnalysisResult(description=combined_description, selected_style=style, strategy=strategy)
        
        if HAS_MULTI_SOLUTION:
            enhanced = _convert_to_enhanced(basic_result, primary_description)
            enhanced.analysis_reasoning = (
                f"双模型协作：主模型建议 {primary_style}，辅助模型建议 {secondary_style}，最终采用 {style}"
            )
            enhanced.raw_description = combined_description
            return enhanced
        
        return basic_result


_ADVISOR_INSTANCE: Optional[LightweightStyleAdvisor] = None
_ADVISOR_LOCK = threading.Lock()
_FACE_CASCADE = None
_FACE_CASCADE_READY = False
_FACE_CASCADE_LOCK = threading.Lock()
_EYE_CASCADE = None
_EYE_CASCADE_READY = False
_EYE_CASCADE_LOCK = threading.Lock()


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


def _merge_collaborative_style(primary_style: str, secondary_style: str, image: "Image.Image") -> str:
    if primary_style == secondary_style:
        return primary_style
    if np is None:
        return primary_style
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    brightness = float(arr.mean()) / 255.0
    if brightness < NIGHT_BRIGHTNESS_THRESHOLD:
        return "night_clarity"
    if brightness > BRIGHT_LANDSCAPE_THRESHOLD and secondary_style == "landscape_vivid":
        return secondary_style
    return primary_style


def _extract_text_from_model_output(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, bytes):
        for encoding in ("utf-8", "gb18030", "latin-1"):
            try:
                return payload.decode(encoding)
            except Exception:
                continue
        return str(payload)
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "content", "caption", "answer"):
            if key in payload:
                text = _extract_text_from_model_output(payload.get(key))
                if text:
                    return text
        return str(payload)
    if isinstance(payload, (list, tuple)):
        parts: List[str] = []
        for item in payload:
            text = _extract_text_from_model_output(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    return str(payload)


def _sanitize_analysis_text(text: str) -> str:
    normalized = str(text or "").replace("\x00", " ")
    normalized = "".join(ch if ch.isprintable() else " " for ch in normalized)
    return " ".join(normalized.split())


def _build_analysis_reason(
    analysis: Any,
    retouch_controls: Optional[Dict[str, float]],
    src_name: Optional[str] = None,
) -> str:
    strategy = getattr(analysis, "strategy", "unknown")
    selected_style = getattr(analysis, "selected_style", "unknown")
    raw_description = getattr(analysis, "raw_description", None) or getattr(analysis, "description", "")
    description = _sanitize_analysis_text(_extract_text_from_model_output(raw_description)) or "暂无可用模型描述"
    base = (
        f"strategy={strategy} | selected_style={selected_style} | "
        f"description={description} | retouch={summarize_retouch_controls(retouch_controls)}"
    )
    geometry = getattr(analysis, "auto_geometry_decision", "")
    if geometry:
        base = f"{base} | geometry={_sanitize_analysis_text(_extract_text_from_model_output(geometry))}"
    return f"{src_name} | {base}" if src_name else base


def _decide_auto_geometry(description: str) -> Dict[str, float]:
    lowered = _sanitize_analysis_text(description).lower()
    rotation = 0
    crop_factor = AUTO_CROP_DEFAULT_FACTOR

    rotate_left_hints = ("向左旋转", "左转", "逆时针", "rotate left", "counterclockwise", "rotate ccw")
    rotate_right_hints = ("向右旋转", "右转", "顺时针", "rotate right", "clockwise", "rotate cw")
    rotate_180_hints = ("180", "倒置", "颠倒", "upside down", "upside-down")
    crop_hints = ("裁切", "裁剪", "crop", "reframe", "too much background", "空白太多", "主体太小")
    tight_crop_hints = ("紧凑裁切", "紧一点", "特写", "close-up", "tight crop")

    if any(h in lowered for h in rotate_left_hints):
        rotation = 90
    elif any(h in lowered for h in rotate_right_hints):
        rotation = 270
    elif any(h in lowered for h in rotate_180_hints):
        rotation = 180

    if any(h in lowered for h in crop_hints):
        crop_factor = 0.90
    if any(h in lowered for h in tight_crop_hints):
        crop_factor = 0.82

    crop_factor = max(0.5, min(1.0, crop_factor))
    return {"rotation": float(rotation), "crop_factor": float(crop_factor)}


def _apply_auto_geometry_to_pil(image: "Image.Image", decision: Dict[str, float]) -> "Image.Image":
    rotation = int(decision.get("rotation", 0))
    crop_factor = float(decision.get("crop_factor", AUTO_CROP_DEFAULT_FACTOR))

    out = image
    if rotation in {90, 180, 270}:
        out = out.rotate(rotation, expand=True)

    crop_factor = max(0.5, min(1.0, crop_factor))
    if crop_factor < 1.0:
        width, height = out.size
        target_w = max(1, int(round(width * crop_factor)))
        target_h = max(1, int(round(height * crop_factor)))
        left = max(0, (width - target_w) // 2)
        top = max(0, (height - target_h) // 2)
        cropped = out.crop((left, top, left + target_w, top + target_h))
        if hasattr(Image, "Resampling"):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.LANCZOS
        out = cropped.resize((width, height), resample=resample)

    return out


def _convert_to_enhanced(basic: AnalysisResult, description: str = "") -> EnhancedAnalysisResult:
    """将基础分析结果转换为增强格式"""
    if not description:
        description = basic.description
    
    lowered = description.lower()
    scene = "portrait"  # 默认
    
    if any(w in lowered for w in ["landscape", "mountain", "sky", "outdoor", "river", "forest"]):
        scene = "landscape"
    elif any(w in lowered for w in ["food", "dish", "meal", "restaurant", "cake", "fruit"]):
        scene = "food"
    elif any(w in lowered for w in ["night", "dark", "street", "city", "urban"]):
        scene = "night"
    elif any(w in lowered for w in ["product", "item", "object", "good"]):
        scene = "product"
    
    # 提取subjects
    subjects = []
    if any(w in lowered for w in ["person", "people", "human", "face", "portrait", "head"]):
        subjects.extend(["person", "face"])
    elif any(w in lowered for w in ["landscape", "nature", "scene"]):
        subjects.append("landscape")
    
    # 估算lighting
    lighting = {
        "brightness": 0.5,
        "direction": "neutral",
    }
    
    # 推荐方向
    recommended_dirs = []
    if "person" in subjects or "face" in subjects:
        recommended_dirs = ["skin_tone", "portrait_retouch"]
    elif scene == "landscape":
        recommended_dirs = ["vibrance", "contrast"]
    elif scene == "food":
        recommended_dirs = ["vibrance", "detail"]
    else:
        recommended_dirs = ["color_correction"]
    
    return EnhancedAnalysisResult(
        raw_description=description,
        scene=scene,
        subjects=subjects,
        lighting=lighting,
        color_profile={},
        recommended_directions=recommended_dirs,
        selected_style=basic.selected_style,
        strategy=basic.strategy,
    )


def describe_supported_formats() -> str:
    image_formats = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
    video_formats = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
    return f"图片: {image_formats} | 视频: {video_formats}"


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_retouch_controls(retouch_controls: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not retouch_controls:
        return {}

    normalized: Dict[str, float] = {}
    for key in RETOUCH_CONTROL_KEYS:
        if key not in retouch_controls:
            continue
        try:
            normalized[key] = _clamp_unit(float(retouch_controls[key]))
        except Exception:
            continue
    return normalized


def summarize_retouch_controls(retouch_controls: Optional[Dict[str, float]]) -> str:
    normalized = normalize_retouch_controls(retouch_controls)
    if not normalized:
        return "preset"

    parts: List[str] = []
    for key in RETOUCH_CONTROL_KEYS:
        if key not in normalized:
            continue
        parts.append(f"{RETOUCH_CONTROL_LABELS[key]}={normalized[key]:.2f}")
    return ", ".join(parts) if parts else "preset"


def _resolve_style_values(
    style_name: str,
    retouch_controls: Optional[Dict[str, float]] = None,
    for_video: bool = False,
) -> Dict[str, float]:
    if style_name not in STYLE_PRESETS:
        style_name = "clean_natural"

    base = STYLE_PRESETS[style_name]
    values: Dict[str, float] = {
        "brightness": float(base.get("brightness", 1.0)),
        "contrast": float(base.get("contrast", 1.0)),
        "color": float(base.get("color", 1.0)),
    }

    for key in RETOUCH_CONTROL_KEYS:
        if for_video and f"{key}_video" in base:
            values[key] = _clamp_unit(base[f"{key}_video"])
        else:
            values[key] = _clamp_unit(base.get(key, 0.0))

    overrides = normalize_retouch_controls(retouch_controls)
    values.update(overrides)
    return values


def get_retouch_profile_values(profile_name: str) -> Dict[str, float]:
    profile = RETOUCH_PROFILE_PRESETS.get(profile_name)
    if profile:
        return normalize_retouch_controls(profile)

    portrait_defaults = _resolve_style_values("portrait_soft")
    return {key: portrait_defaults.get(key, 0.0) for key in RETOUCH_CONTROL_KEYS}


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


def apply_style_to_pil(
    image: "Image.Image",
    style_name: Optional[str] = None,
    retouch_controls: Optional[Dict[str, float]] = None,
    style_values: Optional[Dict[str, float]] = None,
) -> "Image.Image":
    if ImageEnhance is None:
        raise RuntimeError("Pillow is required for image processing.")

    if style_values is None:
        style_values = _resolve_style_values(style_name or "clean_natural", retouch_controls=retouch_controls, for_video=False)
    else:
        # 如果提供了style_values，仅混合置提提供的retouch_controls
        full_values = style_values.copy()
        if retouch_controls:
            full_values.update(retouch_controls)
        style_values = full_values
    
    out = ImageEnhance.Brightness(image).enhance(style_values["brightness"])
    out = ImageEnhance.Contrast(out).enhance(style_values["contrast"])
    out = ImageEnhance.Color(out).enhance(style_values["color"])
    return _apply_advanced_retouch_to_pil(out, style_values)


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


def _get_eye_cascade():
    global _EYE_CASCADE
    global _EYE_CASCADE_READY

    if _EYE_CASCADE_READY:
        return _EYE_CASCADE

    with _EYE_CASCADE_LOCK:
        if _EYE_CASCADE_READY:
            return _EYE_CASCADE

        cascade = None
        if cv2 is not None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
                loaded = cv2.CascadeClassifier(cascade_path)
                if loaded is not None and not loaded.empty():
                    cascade = loaded
            except Exception:
                cascade = None

        _EYE_CASCADE = cascade
        _EYE_CASCADE_READY = True

    return _EYE_CASCADE


def _detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if np is None or cv2 is None:
        return []

    cascade = _get_face_cascade()
    if cascade is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(56, 56))
    if faces is None or len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def _detect_eyes_in_face(face_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if np is None or cv2 is None:
        return []

    cascade = _get_eye_cascade()
    if cascade is None:
        return []

    eyes = cascade.detectMultiScale(face_gray, scaleFactor=1.10, minNeighbors=6, minSize=(14, 14))
    if eyes is None or len(eyes) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in eyes]


def _build_skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
    return cv2.GaussianBlur(skin, (0, 0), sigmaX=2.2, sigmaY=2.2).astype(np.float32) / 255.0


def _apply_skin_smoothing_bgr(frame: np.ndarray, strength: float) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    diameter = max(5, int(7 + strength * 8))
    sigma = 25 + strength * 70
    smooth = cv2.bilateralFilter(frame, d=diameter, sigmaColor=sigma, sigmaSpace=sigma)

    skin_mask = _build_skin_mask(frame)
    blend = np.clip((0.20 + 0.60 * strength) * skin_mask, 0.0, 1.0)[..., None]
    out = frame.astype(np.float32) * (1.0 - blend) + smooth.astype(np.float32) * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_skin_whitening_bgr(frame: np.ndarray, strength: float) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    skin_mask = _build_skin_mask(frame)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    l_channel = ycrcb[:, :, 0].astype(np.float32)
    boosted = np.clip(l_channel + (10 + 30 * strength), 0, 255)
    blend = np.clip((0.15 + 0.50 * strength) * skin_mask, 0.0, 1.0)
    ycrcb[:, :, 0] = np.clip(l_channel * (1.0 - blend) + boosted * blend, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _apply_acne_removal_bgr(frame: np.ndarray, strength: float) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    ksize = 3 if strength < 0.55 else 5
    smooth = cv2.medianBlur(frame, ksize)
    skin_mask = _build_skin_mask(frame)
    blend = np.clip((0.10 + 0.45 * strength) * skin_mask, 0.0, 1.0)[..., None]
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


def _apply_face_slimming_bgr(
    frame: np.ndarray,
    strength: float,
    faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    if faces is None:
        faces = _detect_faces(frame)
    if not faces:
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


def _apply_bulge_patch(frame: np.ndarray, center_x: int, center_y: int, radius: int, strength: float) -> np.ndarray:
    if np is None or cv2 is None or radius < 4:
        return frame

    h, w = frame.shape[:2]
    x0 = max(0, center_x - radius)
    y0 = max(0, center_y - radius)
    x1 = min(w, center_x + radius)
    y1 = min(h, center_y + radius)
    region = frame[y0:y1, x0:x1]
    rh, rw = region.shape[:2]
    if rh < 8 or rw < 8:
        return frame

    yy, xx = np.indices((rh, rw), dtype=np.float32)
    cx = (center_x - x0)
    cy = (center_y - y0)
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    radius_f = float(max(radius, 1))
    mask = dist < radius_f

    factor = np.ones_like(dist, dtype=np.float32)
    factor[mask] = 1.0 - (0.24 * strength) * ((1.0 - dist[mask] / radius_f) ** 2)

    map_x = np.clip(cx + dx * factor, 0, rw - 1).astype(np.float32)
    map_y = np.clip(cy + dy * factor, 0, rh - 1).astype(np.float32)
    warped = cv2.remap(region, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    out = frame.copy()
    out[y0:y1, x0:x1] = warped
    return out


def _apply_eye_enlarge_bgr(
    frame: np.ndarray,
    strength: float,
    faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    if faces is None:
        faces = _detect_faces(frame)
    if not faces:
        return frame

    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for x, y, w, h in faces:
        face_gray = gray[y:y + h, x:x + w]
        eyes = _detect_eyes_in_face(face_gray)

        eye_centers: List[Tuple[int, int, int]] = []
        if eyes:
            eyes = sorted(eyes, key=lambda it: it[2] * it[3], reverse=True)[:2]
            for ex, ey, ew, eh in eyes:
                eye_centers.append((x + ex + ew // 2, y + ey + eh // 2, max(ew, eh) // 2))
        else:
            eye_centers = [
                (x + int(w * 0.33), y + int(h * 0.42), int(w * 0.12)),
                (x + int(w * 0.67), y + int(h * 0.42), int(w * 0.12)),
            ]

        for cx, cy, rr in eye_centers:
            radius = int(max(8, rr * 1.25))
            out = _apply_bulge_patch(out, cx, cy, radius, strength)

    return out


def _apply_nose_slimming_bgr(
    frame: np.ndarray,
    strength: float,
    faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    if faces is None:
        faces = _detect_faces(frame)
    if not faces:
        return frame

    out = frame.copy()
    for x, y, w, h in faces:
        x0 = max(0, x + int(w * 0.32))
        x1 = min(out.shape[1], x + int(w * 0.68))
        y0 = max(0, y + int(h * 0.30))
        y1 = min(out.shape[0], y + int(h * 0.76))
        region = out[y0:y1, x0:x1]
        rh, rw = region.shape[:2]
        if rh < 16 or rw < 16:
            continue

        yy, xx = np.indices((rh, rw), dtype=np.float32)
        cx = (rw - 1) * 0.5
        nx = (xx - cx) / max(cx, 1.0)
        ny = (yy / max(rh - 1, 1))
        influence = np.exp(-(nx ** 2) * 4.0) * np.exp(-((ny - 0.52) ** 2) * 10.0)
        shift = nx * influence * (strength * rw * 0.18)

        map_x = np.clip(xx + shift, 0, rw - 1).astype(np.float32)
        map_y = yy.astype(np.float32)
        out[y0:y1, x0:x1] = cv2.remap(region, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    return out


def _apply_chin_refine_bgr(
    frame: np.ndarray,
    strength: float,
    faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    if faces is None:
        faces = _detect_faces(frame)
    if not faces:
        return frame

    out = frame.copy()
    for x, y, w, h in faces:
        x0 = max(0, x + int(w * 0.18))
        x1 = min(out.shape[1], x + int(w * 0.82))
        y0 = max(0, y + int(h * 0.56))
        y1 = min(out.shape[0], y + h)
        region = out[y0:y1, x0:x1]
        rh, rw = region.shape[:2]
        if rh < 16 or rw < 16:
            continue

        yy, xx = np.indices((rh, rw), dtype=np.float32)
        cx = (rw - 1) * 0.5
        nx = (xx - cx) / max(cx, 1.0)
        y_norm = yy / max(rh - 1, 1)
        influence = np.exp(-(nx ** 2) * 4.0) * np.clip((y_norm - 0.25) / 0.75, 0.0, 1.0)
        shift = influence * (strength * rh * 0.14)

        map_x = xx.astype(np.float32)
        map_y = np.clip(yy - shift, 0, rh - 1).astype(np.float32)
        out[y0:y1, x0:x1] = cv2.remap(region, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    return out


def _blend_tint(frame: np.ndarray, mask: np.ndarray, tint_bgr: Tuple[float, float, float], strength: float) -> np.ndarray:
    if np is None:
        return frame

    tint = np.zeros_like(frame, dtype=np.float32)
    tint[:, :, 0] = tint_bgr[0]
    tint[:, :, 1] = tint_bgr[1]
    tint[:, :, 2] = tint_bgr[2]

    alpha = np.clip(mask * strength, 0.0, 1.0)[..., None]
    out = frame.astype(np.float32) * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_cheek_blush_bgr(
    frame: np.ndarray,
    strength: float,
    faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    if faces is None:
        faces = _detect_faces(frame)
    if not faces:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for x, y, fw, fh in faces:
        for cx, cy in (
            (x + int(fw * 0.28), y + int(fh * 0.62)),
            (x + int(fw * 0.72), y + int(fh * 0.62)),
        ):
            rr_x = max(8, int(fw * 0.16))
            rr_y = max(8, int(fh * 0.12))
            y0 = max(0, cy - rr_y)
            y1 = min(h, cy + rr_y)
            x0 = max(0, cx - rr_x)
            x1 = min(w, cx + rr_x)

            yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
            nx = (xx - (cx - x0)) / max(rr_x, 1)
            ny = (yy - (cy - y0)) / max(rr_y, 1)
            local = np.exp(-(nx * nx + ny * ny) * 3.0)
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], local)

    return _blend_tint(out, mask, tint_bgr=(185, 126, 226), strength=0.34 * strength)


def _apply_lip_tint_bgr(
    frame: np.ndarray,
    strength: float,
    faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    if faces is None:
        faces = _detect_faces(frame)
    if not faces:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for x, y, fw, fh in faces:
        cx = x + int(fw * 0.5)
        cy = y + int(fh * 0.78)
        rr_x = max(8, int(fw * 0.16))
        rr_y = max(6, int(fh * 0.07))

        y0 = max(0, cy - rr_y)
        y1 = min(h, cy + rr_y)
        x0 = max(0, cx - rr_x)
        x1 = min(w, cx + rr_x)

        yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
        nx = (xx - (cx - x0)) / max(rr_x, 1)
        ny = (yy - (cy - y0)) / max(rr_y, 1)
        local = np.exp(-(nx * nx + ny * ny) * 4.0)
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], local)

    return _blend_tint(out, mask, tint_bgr=(145, 85, 205), strength=0.46 * strength)


def _apply_eye_brighten_bgr(
    frame: np.ndarray,
    strength: float,
    faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    if np is None or cv2 is None or strength <= 0:
        return frame

    strength = _clamp_unit(strength)
    if faces is None:
        faces = _detect_faces(frame)
    if not faces:
        return frame

    out = frame.copy().astype(np.float32)
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for x, y, fw, fh in faces:
        face_gray = gray[y:y + fh, x:x + fw]
        eyes = _detect_eyes_in_face(face_gray)
        if not eyes:
            eyes = [
                (int(fw * 0.22), int(fh * 0.34), int(fw * 0.18), int(fh * 0.12)),
                (int(fw * 0.60), int(fh * 0.34), int(fw * 0.18), int(fh * 0.12)),
            ]

        for ex, ey, ew, eh in eyes[:2]:
            cx = x + ex + ew // 2
            cy = y + ey + eh // 2
            rr_x = max(8, int(ew * 0.9))
            rr_y = max(6, int(eh * 0.9))
            y0 = max(0, cy - rr_y)
            y1 = min(h, cy + rr_y)
            x0 = max(0, cx - rr_x)
            x1 = min(w, cx + rr_x)

            yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
            nx = (xx - (cx - x0)) / max(rr_x, 1)
            ny = (yy - (cy - y0)) / max(rr_y, 1)
            local = np.exp(-(nx * nx + ny * ny) * 3.0)
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], local)

    gain = (16.0 + 30.0 * strength) * mask[..., None]
    out = np.clip(out + gain, 0, 255)
    return out.astype(np.uint8)


def _apply_advanced_retouch_to_bgr(frame: np.ndarray, values: Dict[str, float]) -> np.ndarray:
    if np is None or cv2 is None:
        return frame

    out = frame
    face_related_strengths = [
        values.get("slim_face", 0.0),
        values.get("eye_enlarge", 0.0),
        values.get("nose_slim", 0.0),
        values.get("chin_refine", 0.0),
        values.get("blush", 0.0),
        values.get("lip_tint", 0.0),
        values.get("eye_brighten", 0.0),
    ]
    faces = _detect_faces(out) if any(s > 0 for s in face_related_strengths) else []

    out = _apply_face_slimming_bgr(out, values.get("slim_face", 0.0), faces=faces)
    out = _apply_nose_slimming_bgr(out, values.get("nose_slim", 0.0), faces=faces)
    out = _apply_chin_refine_bgr(out, values.get("chin_refine", 0.0), faces=faces)
    out = _apply_eye_enlarge_bgr(out, values.get("eye_enlarge", 0.0), faces=faces)

    out = _apply_skin_smoothing_bgr(out, values.get("skin_smooth", 0.0))
    out = _apply_acne_removal_bgr(out, values.get("acne_remove", 0.0))
    out = _apply_skin_whitening_bgr(out, values.get("skin_whiten", 0.0))
    out = _apply_cheek_blush_bgr(out, values.get("blush", 0.0), faces=faces)
    out = _apply_lip_tint_bgr(out, values.get("lip_tint", 0.0), faces=faces)
    out = _apply_eye_brighten_bgr(out, values.get("eye_brighten", 0.0), faces=faces)
    return out


def _apply_advanced_retouch_to_pil(image: "Image.Image", values: Dict[str, float]) -> "Image.Image":
    if Image is None or np is None or cv2 is None:
        return image

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _apply_advanced_retouch_to_bgr(bgr, values)
    result_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def process_image_file(
    path: str,
    advisor: LightweightStyleAdvisor,
    requested_style: str,
    skip_type_check: bool = False,
    retouch_controls: Optional[Dict[str, float]] = None,
    solutions: Optional[list] = None,
) -> Tuple[str, AnalysisResult] | Tuple[Dict[str, str], EnhancedAnalysisResult]:
    if Image is None:
        raise RuntimeError("Pillow is required. Install pillow to process images.")

    src = Path(path)
    if not skip_type_check and detect_media_type(path) != "image":
        raise ValueError(f"Unsupported image format: {src.suffix.lower()}")

    try:
        with Image.open(src) as image:
            image = image.convert("RGB")
            analysis = advisor.analyze(image, requested_style=requested_style)
            working_image = image
            auto_geometry = {"rotation": 0.0, "crop_factor": AUTO_CROP_DEFAULT_FACTOR}
            if requested_style == "auto":
                desc = getattr(analysis, "raw_description", None) or getattr(analysis, "description", "")
                auto_geometry = _decide_auto_geometry(_extract_text_from_model_output(desc))
                working_image = _apply_auto_geometry_to_pil(image, auto_geometry)
            setattr(analysis, "auto_geometry_decision", f"rotate={int(auto_geometry['rotation'])}, crop={auto_geometry['crop_factor']:.2f}")
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Unable to decode image file: {src.name}") from exc

    # 如果没有指定多方案，走原有单方案流程
    if solutions is None:
        try:
            styled = apply_style_to_pil(working_image, analysis.selected_style, retouch_controls=retouch_controls)
        except UnidentifiedImageError as exc:
            raise RuntimeError(f"Unable to decode image file: {src.name}") from exc
        
        output_path = _build_output_path(src, analysis.selected_style, ".jpg")
        styled.save(output_path, format="JPEG", quality=95)
        return str(output_path), analysis
    
    # 多方案处理
    output_dict = {}
    for solution in solutions:
        style_vals = solution.style_adjustments.copy()
        
        # 合并精修参数
        retouch = dict(retouch_controls or {})
        retouch.update(solution.retouch_overrides)
        
        # 应用方案
        styled = apply_style_to_pil(working_image, style_name=None, retouch_controls=retouch, style_values=style_vals)
        
        # 保存
        output_path = _build_output_path(src, solution.name, ".jpg")
        styled.save(output_path, format="JPEG", quality=95)
        output_dict[solution.name] = str(output_path)
    
    return output_dict, analysis


def _apply_style_to_frame(
    frame: np.ndarray,
    style_name: str,
    retouch_controls: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for video frame processing.")
    values = _resolve_style_values(style_name, retouch_controls=retouch_controls, for_video=True)
    frame_f = frame.astype(np.float32)
    frame_f = np.clip(frame_f * values["brightness"], 0, 255)

    mean = frame_f.mean(axis=(0, 1), keepdims=True)
    frame_f = np.clip((frame_f - mean) * values["contrast"] + mean, 0, 255)

    frame_u8 = frame_f.astype(np.uint8)
    gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    frame_f = np.clip(gray + (frame_f - gray) * values["color"], 0, 255)

    out = frame_f.astype(np.uint8)
    return _apply_advanced_retouch_to_bgr(out, values)


def process_video_file(
    path: str,
    advisor: LightweightStyleAdvisor,
    requested_style: str,
    skip_type_check: bool = False,
    retouch_controls: Optional[Dict[str, float]] = None,
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

        writer.write(_apply_style_to_frame(sample_frame, analysis.selected_style, retouch_controls=retouch_controls))

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_apply_style_to_frame(frame, analysis.selected_style, retouch_controls=retouch_controls))
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return str(output_path), analysis


def process_media(
    path: str,
    requested_style: str = "auto",
    retouch_controls: Optional[Dict[str, float]] = None,
) -> Tuple[str, str, str]:
    if not path:
        raise ValueError("Please upload an image or video file.")

    advisor = get_advisor()
    media_type = detect_media_type(path)

    if media_type == "image":
        output_path, analysis = process_image_file(
            path,
            advisor,
            requested_style,
            skip_type_check=True,
            retouch_controls=retouch_controls,
        )
    else:
        output_path, analysis = process_video_file(
            path,
            advisor,
            requested_style,
            skip_type_check=True,
            retouch_controls=retouch_controls,
        )

    reason = (
        f"Model: {MODEL_NAME} | Strategy: {analysis.strategy} | "
        f"Description: {analysis.description} | Selected style: {analysis.selected_style} | "
        f"Retouch: {summarize_retouch_controls(retouch_controls)}"
    )
    geometry = getattr(analysis, "auto_geometry_decision", "")
    if geometry:
        reason = f"{reason} | Geometry: {geometry}"
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
        if isinstance(item, dict):
            # Gradio FileData-like payloads expose the resolved temporary path in `path`.
            # `name` is used only as a compatibility fallback when `path` is absent.
            raw_path = item.get("path") if "path" in item else item.get("name")
            path = str(raw_path).strip() if raw_path is not None else ""
        elif hasattr(item, "path"):
            path = str(getattr(item, "path") or "").strip()
        elif hasattr(item, "name"):
            path = str(getattr(item, "name") or "").strip()
        else:
            path = str(item).strip()
        if path:
            paths.append(path)
    return paths


def process_uploaded_files(
    file_paths: Sequence[str],
    requested_style: str = "auto",
    retouch_controls: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], List[str], List[str], str, str, str]:
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
            retouch_controls=retouch_controls,
        )
        reason = _build_analysis_reason(analysis, retouch_controls)
        return [str(output_path)], [], [], analysis.selected_style, reason, double_check_implementation()

    output_paths: List[str] = []
    before_gallery: List[str] = []
    after_gallery: List[str] = []
    style_lines: List[str] = []
    reason_lines: List[str] = []

    for path in image_paths:
        output_path, analysis = process_image_file(
            path,
            advisor,
            requested_style,
            skip_type_check=True,
            retouch_controls=retouch_controls,
        )
        src_name = Path(path).name

        output_path_str = str(output_path)
        output_paths.append(output_path_str)
        before_gallery.append(path)
        after_gallery.append(output_path_str)
        style_lines.append(f"{src_name}: {analysis.selected_style}")
        reason_lines.append(_build_analysis_reason(analysis, retouch_controls, src_name=src_name))

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

    default_profile_name = "自然韩系"
    default_profile_values = get_retouch_profile_values(default_profile_name)

    def _build_manual_retouch_controls(
        enable_manual_retouch,
        skin_smooth,
        skin_whiten,
        acne_remove,
        blush,
        eye_brighten,
        lip_tint,
        slim_face,
        eye_enlarge,
        nose_slim,
        chin_refine,
    ) -> Optional[Dict[str, float]]:
        if not enable_manual_retouch:
            return None

        return normalize_retouch_controls(
            {
                "skin_smooth": skin_smooth,
                "skin_whiten": skin_whiten,
                "acne_remove": acne_remove,
                "blush": blush,
                "eye_brighten": eye_brighten,
                "lip_tint": lip_tint,
                "slim_face": slim_face,
                "eye_enlarge": eye_enlarge,
                "nose_slim": nose_slim,
                "chin_refine": chin_refine,
            }
        )

    def _apply_retouch_profile(profile_name: str):
        values = get_retouch_profile_values(profile_name)
        return tuple(values.get(key, 0.0) for key in RETOUCH_CONTROL_KEYS)

    def _format_solutions_for_display(solutions: List[SolutionVariant]) -> str:
        """将多个方案格式化为易读的文本展示"""
        if not solutions:
            return "未生成任何方案"
        
        display = "📋 AI 推荐的修图方案（共 {} 个）:\n\n".format(len(solutions))
        for i, sol in enumerate(solutions, 1):
            display += f"【方案 {i}】{sol.display_name}\n"
            display += f"  • 描述: {sol.description}\n"
            display += f"  • 理由: {sol.reasoning}\n"
            display += f"  • 强度: {sol.intensity:.0%}\n"
            display += "\n"
        
        return display
    
    def _extract_preferences_from_feedback(feedback_text: str, feedback_sentiment: str) -> Dict[str, str]:
        lowered = (feedback_text or "").lower()
        preferences: Dict[str, str] = {}
        if any(word in lowered for word in PREFERENCE_BRIGHTNESS_LOWER_KEYWORDS):
            preferences["brightness"] = "lower"
        elif any(word in lowered for word in PREFERENCE_BRIGHTNESS_HIGHER_KEYWORDS):
            preferences["brightness"] = "higher"
        if any(word in lowered for word in PREFERENCE_SATURATION_LOWER_KEYWORDS):
            preferences["saturation"] = "lower"
        elif any(word in lowered for word in PREFERENCE_SATURATION_HIGHER_KEYWORDS):
            preferences["saturation"] = "higher"
        if "自然" in lowered:
            preferences["style"] = "natural"
        elif "电影" in lowered:
            preferences["style"] = "cinematic"
        if feedback_sentiment.startswith("不满意") and "style" not in preferences:
            preferences["style"] = "natural"
        return preferences
    
    def _create_version_comparison_html(versions: List) -> str:
        """为多个版本创建对比HTML"""
        if not versions:
            return "<p>暂无版本可对比</p>"
        
        html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;'>\n"
        for i, ver in enumerate(versions, 1):
            html += f"""
            <div style='border: 1px solid #ddd; padding: 15px; border-radius: 8px;'>
                <h3>版本 {i}: {ver.get('display_name', '未命名')}</h3>
                <p><strong>方案:</strong> {ver.get('solution_name', 'N/A')}</p>
                <p><strong>强度:</strong> {ver.get('intensity', 0):.0%}</p>
                <p><strong>生成轮次:</strong> {ver.get('generation_round', 1)}</p>
                <p><strong>理由:</strong><br/>{ver.get('reasoning', '无')}</p>
            </div>
            """
        html += "</div>"
        return html
    
    def _handle_feedback_submission(
        session_id_state: str,
        feedback_text: str,
        target_solution: str,
        feedback_sentiment: str,
        prefer_large_model: bool,
    ) -> Tuple[str, str, str]:
        """处理用户反馈，返回确认信息和迭代建议"""
        if not HAS_SOLUTION_MANAGER:
            return "error", "解决方案管理器不可用", "暂无偏好记忆"
        
        try:
            manager = get_manager()
            if not session_id_state or session_id_state not in manager.sessions:
                return "error", "找不到会话，请重新上传图片", "暂无偏好记忆"
            
            sentiment_map = {"满意 👍": "positive", "一般 🤷": "neutral", "不满意 👎": "negative"}
            sentiment = sentiment_map.get(feedback_sentiment, "neutral")
            
            # 添加反馈
            feedback = manager.add_feedback(
                session_id_state,
                feedback_text=feedback_text,
                target_solution=target_solution,
                sentiment=sentiment,
                prefer_stronger_model=prefer_large_model,
            )
            preferences = _extract_preferences_from_feedback(feedback_text, feedback_sentiment)
            if preferences:
                manager.update_preference_memory(
                    preferences=preferences,
                    source_feedback=feedback_text,
                    updated_at=feedback.created_at,
                )
            
            # 生成迭代建议
            suggestions = manager.get_iteration_suggestions(session_id_state)
            
            confirm_msg = f"✅ 反馈已记录！共 {len(manager.sessions[session_id_state].feedbacks)} 条反馈"
            suggestions_text = "## 迭代建议\n\n"
            for key, value in suggestions.items():
                suggestions_text += f"**{key}**: {value}\n\n"
            if preferences:
                suggestions_text += f"**memory**: 已更新偏好记忆 {preferences}\n\n"
            
            if not suggestions:
                suggestions_text += "继续上传反馈以获得更多优化建议"
            
            return confirm_msg, suggestions_text, manager.get_preference_memory_summary()
            
        except Exception as e:
            return "error", f"处理反馈时出错: {str(e)}", "暂无偏好记忆"

    def _build_memory_aware_solutions(
        analysis: EnhancedAnalysisResult,
        manager,
        session_id: str,
    ) -> List[SolutionVariant]:
        solutions = generate_multiple_solutions(analysis, max_solutions=4)
        latest_feedback = manager.sessions[session_id].feedbacks[-1] if manager.sessions[session_id].feedbacks else None
        adjusted: List[SolutionVariant] = []
        for sol in solutions:
            style_adjustments = manager.apply_memory_preferences(sol.style_adjustments, sol.name)
            intensity = sol.intensity
            if latest_feedback and latest_feedback.sentiment == "negative":
                if sol.name in {"contrast_pop", "cinematic_grade", "vibrance_boost"}:
                    intensity = max(NEGATIVE_STYLE_MIN_INTENSITY, intensity * NEGATIVE_STYLE_INTENSITY_MULTIPLIER)
                    style_adjustments["contrast"] = max(
                        NEGATIVE_STYLE_CONTRAST_MIN,
                        style_adjustments.get("contrast", 1.0) * NEGATIVE_STYLE_CONTRAST_MULTIPLIER,
                    )
                    style_adjustments["color"] = max(
                        NEGATIVE_STYLE_COLOR_MIN,
                        style_adjustments.get("color", 1.0) * NEGATIVE_STYLE_COLOR_MULTIPLIER,
                    )
            adjusted.append(
                SolutionVariant(
                    name=sol.name,
                    display_name=sol.display_name,
                    description=sol.description,
                    reasoning=sol.reasoning,
                    style_adjustments=style_adjustments,
                    retouch_overrides=sol.retouch_overrides.copy(),
                    applicable_scenes=sol.applicable_scenes.copy(),
                    intensity=intensity,
                )
            )
        return adjusted

    def _handle_upload(
        file_obj,
        style_name,
        enable_manual_retouch,
        skin_smooth,
        skin_whiten,
        acne_remove,
        blush,
        eye_brighten,
        lip_tint,
        slim_face,
        eye_enlarge,
        nose_slim,
        chin_refine,
    ):
        file_paths = normalize_uploaded_file_paths(file_obj)
        if not file_paths:
            raise gr.Error("请先上传图片或视频文件")

        retouch_controls = _build_manual_retouch_controls(
            enable_manual_retouch,
            skin_smooth,
            skin_whiten,
            acne_remove,
            blush,
            eye_brighten,
            lip_tint,
            slim_face,
            eye_enlarge,
            nose_slim,
            chin_refine,
        )

        try:
            result = process_uploaded_files(file_paths, style_name, retouch_controls=retouch_controls)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        versions_text = "当前上传结果未生成多版本详情"
        session_id = ""
        solution_preview_paths: List[str] = []
        target_update = gr.update()
        memory_summary = "暂无偏好记忆"

        if HAS_MULTI_SOLUTION and HAS_SOLUTION_MANAGER:
            try:
                first_image_path = next((p for p in file_paths if detect_media_type(p) == "image"), None)
                if first_image_path and Image is not None:
                    with Image.open(first_image_path) as image:
                        image = image.convert("RGB")
                        analysis = get_advisor().analyze(image, requested_style=style_name)

                    if not isinstance(analysis, EnhancedAnalysisResult):
                        analysis = _convert_to_enhanced(analysis, getattr(analysis, "description", ""))

                    solutions = generate_multiple_solutions(analysis, max_solutions=4)
                    versions_text = _format_solutions_for_display(solutions)

                    manager = get_manager()
                    session_id = str(uuid.uuid4())
                    session = manager.create_session(session_id, first_image_path)
                    session.analysis_reasoning = _build_analysis_reason(
                        analysis,
                        retouch_controls,
                        src_name=Path(first_image_path).name,
                    )

                    for sol in solutions:
                        manager.add_version(
                            session_id=session_id,
                            solution_name=sol.name,
                            display_name=sol.display_name,
                            description=sol.description,
                            reasoning=sol.reasoning,
                            intensity=sol.intensity,
                            style_adjustments=sol.style_adjustments,
                            retouch_overrides=sol.retouch_overrides,
                        )
                    try:
                        output_dict, _ = process_image_file(
                            first_image_path,
                            get_advisor(),
                            style_name,
                            skip_type_check=True,
                            retouch_controls=retouch_controls,
                            solutions=solutions,
                        )
                        if isinstance(output_dict, dict):
                            solution_preview_paths = list(output_dict.values())
                            for ver in manager.sessions[session_id].versions:
                                ver.output_path = output_dict.get(ver.solution_name)
                    except Exception:
                        solution_preview_paths = []
                    target_update = gr.update(
                        choices=[sol.name for sol in solutions],
                        value=solutions[0].name if solutions else None,
                    )
                    memory_summary = manager.get_preference_memory_summary()
                elif not first_image_path:
                    versions_text = "当前输入包含视频，暂不支持多版本对比。"
            except Exception as exc:
                versions_text = f"生成多版本详情失败: {exc}"

        return (*result, versions_text, session_id, solution_preview_paths, target_update, memory_summary)

    def _regenerate_from_feedback(
        session_id_state: str,
        style_name: str,
        enable_manual_retouch,
        skin_smooth,
        skin_whiten,
        acne_remove,
        blush,
        eye_brighten,
        lip_tint,
        slim_face,
        eye_enlarge,
        nose_slim,
        chin_refine,
    ):
        if not HAS_MULTI_SOLUTION or not HAS_SOLUTION_MANAGER:
            return "❌ 当前环境不支持迭代重生成", "", [], gr.update(), "暂无偏好记忆"
        manager = get_manager()
        if not session_id_state or session_id_state not in manager.sessions:
            return "❌ 找不到会话，请先上传图片", "", [], gr.update(), manager.get_preference_memory_summary()
        session = manager.sessions[session_id_state]
        source_path = session.input_image_path
        if not source_path:
            return "❌ 会话缺少原始图片", "", [], gr.update(), manager.get_preference_memory_summary()

        retouch_controls = _build_manual_retouch_controls(
            enable_manual_retouch,
            skin_smooth,
            skin_whiten,
            acne_remove,
            blush,
            eye_brighten,
            lip_tint,
            slim_face,
            eye_enlarge,
            nose_slim,
            chin_refine,
        )
        try:
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                analysis = get_advisor().analyze(image, requested_style=style_name)
            if not isinstance(analysis, EnhancedAnalysisResult):
                analysis = _convert_to_enhanced(analysis, getattr(analysis, "description", ""))
            round_no = manager.start_new_round(session_id_state)
            solutions = _build_memory_aware_solutions(analysis, manager, session_id_state)
            output_dict, _ = process_image_file(
                source_path,
                get_advisor(),
                style_name,
                skip_type_check=True,
                retouch_controls=retouch_controls,
                solutions=solutions,
            )
            for sol in solutions:
                manager.add_version(
                    session_id=session_id_state,
                    solution_name=sol.name,
                    display_name=sol.display_name,
                    description=sol.description,
                    reasoning=sol.reasoning,
                    intensity=sol.intensity,
                    style_adjustments=sol.style_adjustments,
                    retouch_overrides=sol.retouch_overrides,
                    output_path=output_dict.get(sol.name) if isinstance(output_dict, dict) else None,
                )
            preview_paths = list(output_dict.values()) if isinstance(output_dict, dict) else []
            msg = f"✅ 已基于原图完成第 {round_no} 轮重生成（未叠加历史滤镜）。"
            return (
                msg,
                _format_solutions_for_display(solutions),
                preview_paths,
                gr.update(choices=[sol.name for sol in solutions], value=solutions[0].name if solutions else None),
                manager.get_preference_memory_summary(),
            )
        except Exception as exc:
            return f"❌ 重新生成失败: {exc}", "", [], gr.update(), manager.get_preference_memory_summary()

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

                with gr.Accordion("🧑 亚洲人像精修调整板块", open=False):
                    with gr.Row():
                        retouch_profile = gr.Dropdown(
                            choices=list(RETOUCH_PROFILE_PRESETS.keys()),
                            value=default_profile_name,
                            label="人像参数预设包",
                            info="一键加载常见人像风格参数：自然韩系 / 清透日系 / 轻欧式",
                        )
                        apply_profile_btn = gr.Button("应用预设到滑杆", size="sm")

                    enable_manual_retouch = gr.Checkbox(
                        label="启用手动精修参数（覆盖风格默认）",
                        value=False,
                    )

                    gr.Markdown("**皮肤美化**")
                    with gr.Row():
                        skin_smooth = gr.Slider(0.0, 1.0, value=default_profile_values["skin_smooth"], step=0.01, label="磨皮")
                        skin_whiten = gr.Slider(0.0, 1.0, value=default_profile_values["skin_whiten"], step=0.01, label="美白")
                        acne_remove = gr.Slider(0.0, 1.0, value=default_profile_values["acne_remove"], step=0.01, label="祛痘")
                    with gr.Row():
                        blush = gr.Slider(0.0, 1.0, value=default_profile_values["blush"], step=0.01, label="红润")
                        eye_brighten = gr.Slider(0.0, 1.0, value=default_profile_values["eye_brighten"], step=0.01, label="亮眼")
                        lip_tint = gr.Slider(0.0, 1.0, value=default_profile_values["lip_tint"], step=0.01, label="唇色")

                    gr.Markdown("**五官塑形**")
                    with gr.Row():
                        slim_face = gr.Slider(0.0, 1.0, value=default_profile_values["slim_face"], step=0.01, label="瘦脸")
                        eye_enlarge = gr.Slider(0.0, 1.0, value=default_profile_values["eye_enlarge"], step=0.01, label="大眼")
                    with gr.Row():
                        nose_slim = gr.Slider(0.0, 1.0, value=default_profile_values["nose_slim"], step=0.01, label="窄鼻")
                        chin_refine = gr.Slider(0.0, 1.0, value=default_profile_values["chin_refine"], step=0.01, label="下巴")

                    gr.Markdown("提示：不勾选“启用手动精修参数”时，将自动使用当前风格内置参数。")

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
                
                # ==================== 新增：多版本对比和用户反馈部分 ====================
                with gr.Accordion("🎨 多版本对比与反馈（新功能）", open=False):
                    gr.Markdown("**双模型协作出方案，支持对比-点评-重生成闭环。**")
                    
                    # 隐藏的会话ID状态
                    session_id_state = gr.State(value="")
                    
                    with gr.Tabs():
                        with gr.Tab("方案与预览"):
                            versions_display = gr.Textbox(
                                label="📋 生成的修图方案详情",
                                lines=10,
                                interactive=False,
                                info="展示双模型协作后的方案与推荐理由"
                            )
                            solution_preview_gallery = gr.Gallery(
                                label="🖼️ 多方案预览（可反复点评）",
                                columns=4,
                                rows=1,
                                height="auto",
                            )
                        with gr.Tab("点评与记忆"):
                            with gr.Row():
                                target_solution = gr.Dropdown(
                                    choices=["color_correction"],
                                    value="color_correction",
                                    label="目标方案",
                                    info="选择你要点评的方案"
                                )
                                feedback_sentiment = gr.Radio(
                                    choices=["满意 👍", "一般 🤷", "不满意 👎"],
                                    value="一般 🤷",
                                    label="你的评价"
                                )
                            feedback_text = gr.Textbox(
                                label="反馈内容",
                                placeholder="例如：亮度太高、希望更自然、饱和度降低一点",
                                lines=3,
                            )
                            prefer_large_model = gr.Checkbox(
                                label="🚀 使用更强的AI模型重新分析 (LLaVA-1.6-13B)",
                                value=False,
                                info="当前模型达不到要求时可勾选"
                            )
                            submit_feedback_btn = gr.Button("提交点评并生成迭代建议 💬", variant="secondary")
                            feedback_response = gr.Textbox(
                                label="AI迭代建议",
                                lines=6,
                                interactive=False,
                            )
                            memory_display = gr.Textbox(
                                label="🧠 用户偏好记忆库",
                                lines=5,
                                interactive=False,
                            )
                        with gr.Tab("重新生成"):
                            with gr.Row():
                                regenerate_btn = gr.Button("基于点评重新生成并刷新预览 🔄", variant="primary")
                                export_report_btn = gr.Button("导出完整报告 📄")
                            regenerate_output = gr.Textbox(
                                label="重新生成结果",
                                lines=4,
                                interactive=False
                            )
                
                # ==================== 新增部分结束 ====================

        run_button.click(
            fn=_handle_upload,
            inputs=[
                media_input,
                style_choice,
                enable_manual_retouch,
                skin_smooth,
                skin_whiten,
                acne_remove,
                blush,
                eye_brighten,
                lip_tint,
                slim_face,
                eye_enlarge,
                nose_slim,
                chin_refine,
            ],
            outputs=[
                output_files,
                before_preview,
                after_preview,
                output_style,
                output_reason,
                output_check,
                versions_display,
                session_id_state,
                solution_preview_gallery,
                target_solution,
                memory_display,
            ],
        )

        retouch_profile.change(
            fn=_apply_retouch_profile,
            inputs=[retouch_profile],
            outputs=[
                skin_smooth,
                skin_whiten,
                acne_remove,
                blush,
                eye_brighten,
                lip_tint,
                slim_face,
                eye_enlarge,
                nose_slim,
                chin_refine,
            ],
        )

        apply_profile_btn.click(
            fn=_apply_retouch_profile,
            inputs=[retouch_profile],
            outputs=[
                skin_smooth,
                skin_whiten,
                acne_remove,
                blush,
                eye_brighten,
                lip_tint,
                slim_face,
                eye_enlarge,
                nose_slim,
                chin_refine,
            ],
        )
        
        # ==================== 新增：多版本反馈回调 ====================
        # 提交反馈回调
        submit_feedback_btn.click(
            fn=_handle_feedback_submission,
            inputs=[session_id_state, feedback_text, target_solution, feedback_sentiment, prefer_large_model],
            outputs=[feedback_response, regenerate_output, memory_display],
        )
        
        regenerate_btn.click(
            fn=_regenerate_from_feedback,
            inputs=[
                session_id_state,
                style_choice,
                enable_manual_retouch,
                skin_smooth,
                skin_whiten,
                acne_remove,
                blush,
                eye_brighten,
                lip_tint,
                slim_face,
                eye_enlarge,
                nose_slim,
                chin_refine,
            ],
            outputs=[regenerate_output, versions_display, solution_preview_gallery, target_solution, memory_display],
        )
        
        # 导出报告回调
        def _export_report(session_id: str) -> str:
            if not HAS_SOLUTION_MANAGER or not session_id:
                return "❌ 无可用会话"
            try:
                manager = get_manager()
                report = manager.export_session_report(session_id)
                return f"✅ 报告已生成:\n\n{report}"
            except Exception as e:
                return f"❌ 导出失败: {str(e)}"
        
        export_report_btn.click(
            fn=_export_report,
            inputs=[session_id_state],
            outputs=[regenerate_output],
        )
        
        # ==================== 新增回调结束 ====================

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
