"""多方案图片修改生成器 - 支持同时生成多个不同方向的修改版本"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class EnhancedAnalysisResult:
    """升级的分析结果 - 包含多维度图像特征"""
    raw_description: str           # 原始模型输出
    scene: str                     # "portrait"/"landscape"/"food"/"night"/"product"/etc
    subjects: List[str]            # ["person", "face"], ["landscape"], etc
    lighting: Dict = field(default_factory=dict)  # {"brightness": 0.7, "direction": "side"}
    color_profile: Dict = field(default_factory=dict)  # {"dominant": "warm", "saturation": "high"}
    recommended_directions: List[str] = field(default_factory=list)  # 推荐的修改方向
    selected_style: str = ""       # 默认选中的风格
    strategy: str = "llm"          # "llm" or "heuristic_fallback"
    analysis_reasoning: str = ""   # 详细分析理由 - 为什么这样判断


@dataclass
class SolutionVariant:
    """单个修改方案"""
    name: str                      # "skin_tone_enhance", "color_correction" 等
    display_name: str              # 用户可见的中文名称
    description: str               # 简要描述
    reasoning: str = ""            # 推荐理由 - 为什么选择这个方案
    style_adjustments: Dict[str, float] = field(default_factory=dict)  # 基础调色参数
    retouch_overrides: Dict[str, float] = field(default_factory=dict)  # 人像精修覆盖
    applicable_scenes: List[str] = field(default_factory=list)  # 适用场景
    intensity: float = 0.7         # 方案强度 (0.3-1.0)


# ============================================================================
# 多方案配置模板
# ============================================================================

SOLUTION_TEMPLATES: Dict[str, Dict] = {
    "color_correction": {
        "display_name": "色温纠正",
        "description": "自动纠正色温，恢复自然色彩",
        "applicable_scenes": ["landscape", "food", "product", "portrait"],
        "style_adjustments": {
            "brightness": 1.0,
            "contrast": 1.05,
            "color": 0.95,
        }
    },
    "skin_tone_enhance": {
        "display_name": "肤色优化",
        "description": "提升肤色气色，增加红润感",
        "applicable_scenes": ["portrait"],
        "style_adjustments": {
            "brightness": 1.03,
            "contrast": 1.08,
            "color": 1.08,
        },
        "retouch_overrides": {
            "skin_smooth": 0.45,
            "skin_whiten": 0.30,
            "blush": 0.35,
            "eye_brighten": 0.25,
        }
    },
    "contrast_pop": {
        "display_name": "对比强化",
        "description": "增强对比度，让画面更有冲击力",
        "applicable_scenes": ["landscape", "food", "night"],
        "style_adjustments": {
            "brightness": 1.0,
            "contrast": 1.25,
            "color": 1.0,
        }
    },
    "vibrance_boost": {
        "display_name": "活力增强",
        "description": "提升饱和度和鲜艳度，色彩更生动",
        "applicable_scenes": ["landscape", "food", "product"],
        "style_adjustments": {
            "brightness": 1.0,
            "contrast": 1.08,
            "color": 1.30,
        }
    },
    "portrait_retouch": {
        "display_name": "人像精修",
        "description": "全面精修人像：磨皮、美白、塑形",
        "applicable_scenes": ["portrait"],
        "style_adjustments": {
            "brightness": 1.05,
            "contrast": 1.08,
            "color": 1.06,
        },
        "retouch_overrides": {
            "skin_smooth": 0.55,
            "skin_whiten": 0.35,
            "acne_remove": 0.40,
            "blush": 0.25,
            "eye_brighten": 0.30,
            "lip_tint": 0.20,
            "slim_face": 0.35,
            "eye_enlarge": 0.25,
            "nose_slim": 0.20,
            "chin_refine": 0.18,
        }
    },
    "cinematic_grade": {
        "display_name": "电影分级",
        "description": "电影风格分级，降低亮度增加对比和饱和度平衡",
        "applicable_scenes": ["landscape", "night", "portrait"],
        "style_adjustments": {
            "brightness": 0.95,
            "contrast": 1.20,
            "color": 0.88,
        }
    },
    "detail_sharpen": {
        "display_name": "细节锐化",
        "description": "增强细节和清晰度",
        "applicable_scenes": ["landscape", "product", "food"],
        "style_adjustments": {
            "brightness": 1.0,
            "contrast": 1.15,
            "color": 1.05,
        }
    },
    # ── Stylized color-grading templates ──────────────────────────────────
    "warm_golden_grade": {
        "display_name": "暖金调色",
        "description": "黄金时刻暖色调，橙红高光，柔和暗部",
        "applicable_scenes": ["portrait", "landscape", "food"],
        "style_adjustments": {
            "brightness": 1.06,
            "contrast": 1.08,
            "color": 1.12,
            "warm_tint": 0.65,
            "shadows_lift": 0.20,
            "highlights_pull": 0.15,
            "vignette": 0.25,
        }
    },
    "teal_orange_grade": {
        "display_name": "青橙分色",
        "description": "电影级青橙色调，暗部偏青，高光偏橙",
        "applicable_scenes": ["landscape", "night", "portrait"],
        "style_adjustments": {
            "brightness": 0.96,
            "contrast": 1.22,
            "color": 0.85,
            "warm_tint": 0.50,
            "teal_shadow_tint": 0.60,
            "highlights_pull": 0.20,
            "vignette": 0.35,
        }
    },
    "vintage_fade_grade": {
        "display_name": "复古胶片",
        "description": "胶片复古效果，暗角、褪色和颗粒感",
        "applicable_scenes": ["portrait", "landscape", "food"],
        "style_adjustments": {
            "brightness": 1.02,
            "contrast": 0.88,
            "color": 0.78,
            "shadows_lift": 0.35,
            "warm_tint": 0.30,
            "vignette": 0.40,
            "grain": 0.50,
        }
    },
    "bright_airy_grade": {
        "display_name": "明亮空气感",
        "description": "高亮低对比，营造清新空气感",
        "applicable_scenes": ["portrait", "food", "product"],
        "style_adjustments": {
            "brightness": 1.15,
            "contrast": 0.90,
            "color": 1.08,
            "shadows_lift": 0.25,
            "highlights_pull": 0.25,
            "warm_tint": 0.25,
            "vignette": 0.05,
        }
    },
    "moody_dark_grade": {
        "display_name": "暗调情绪",
        "description": "低亮高对比，营造戏剧性暗调氛围",
        "applicable_scenes": ["landscape", "night", "portrait"],
        "style_adjustments": {
            "brightness": 0.88,
            "contrast": 1.30,
            "color": 0.82,
            "highlights_pull": 0.30,
            "warm_tint": -0.20,
            "vignette": 0.50,
        }
    },
}


# ============================================================================
# 多方案生成函数
# ============================================================================

def generate_multiple_solutions(
    analysis_result: EnhancedAnalysisResult,
    max_solutions: int = 4,
) -> List[SolutionVariant]:
    """
    根据分析结果生成多个修改方案（含风格化调色选项）
    
    Args:
        analysis_result: 增强的图像分析结果
        max_solutions: 最多生成多少个方案
    
    Returns:
        方案列表（按推荐度排序）
    """
    scene = analysis_result.scene
    recommended_dirs = analysis_result.recommended_directions
    
    solutions: List[SolutionVariant] = []
    
    # 1. 总是加入色温纠正（通用方案）
    solutions.append(_create_solution("color_correction", intensity=0.6))
    
    # 2. 根据场景推荐专化方案（实用 + 风格化各一个）
    if scene == "portrait":
        solutions.append(_create_solution("skin_tone_enhance", intensity=0.75))
        solutions.append(_create_solution("warm_golden_grade", intensity=0.70))
        solutions.append(_create_solution("bright_airy_grade", intensity=0.65))
        solutions.append(_create_solution("portrait_retouch", intensity=0.65))
        solutions.append(_create_solution("vintage_fade_grade", intensity=0.55))
        solutions.append(_create_solution("cinematic_grade", intensity=0.55))
        
    elif scene == "landscape":
        solutions.append(_create_solution("vibrance_boost", intensity=0.8))
        solutions.append(_create_solution("teal_orange_grade", intensity=0.75))
        solutions.append(_create_solution("contrast_pop", intensity=0.7))
        solutions.append(_create_solution("moody_dark_grade", intensity=0.65))
        solutions.append(_create_solution("detail_sharpen", intensity=0.6))
        
    elif scene == "food":
        solutions.append(_create_solution("warm_golden_grade", intensity=0.80))
        solutions.append(_create_solution("vibrance_boost", intensity=0.80))
        solutions.append(_create_solution("bright_airy_grade", intensity=0.70))
        solutions.append(_create_solution("detail_sharpen", intensity=0.65))
        
    elif scene == "night":
        solutions.append(_create_solution("teal_orange_grade", intensity=0.80))
        solutions.append(_create_solution("moody_dark_grade", intensity=0.75))
        solutions.append(_create_solution("contrast_pop", intensity=0.75))
        solutions.append(_create_solution("detail_sharpen", intensity=0.70))

    elif scene in {"warm", "airy"}:
        solutions.append(_create_solution("warm_golden_grade", intensity=0.80))
        solutions.append(_create_solution("bright_airy_grade", intensity=0.75))
        solutions.append(_create_solution("vibrance_boost", intensity=0.65))

    elif scene in {"moody", "vintage"}:
        solutions.append(_create_solution("moody_dark_grade", intensity=0.80))
        solutions.append(_create_solution("teal_orange_grade", intensity=0.75))
        solutions.append(_create_solution("vintage_fade_grade", intensity=0.70))
        solutions.append(_create_solution("contrast_pop", intensity=0.65))
    
    else:
        # 默认：活力 + 暖金 + 对比
        solutions.append(_create_solution("vibrance_boost", intensity=0.7))
        solutions.append(_create_solution("warm_golden_grade", intensity=0.65))
        solutions.append(_create_solution("contrast_pop", intensity=0.65))
        if any(s in analysis_result.subjects for s in ["person", "face"]):
            solutions.append(_create_solution("portrait_retouch", intensity=0.6))
    
    # 3. 去重并限制数量
    seen_names = set()
    unique_solutions = []
    for sol in solutions:
        if sol.name not in seen_names:
            unique_solutions.append(sol)
            seen_names.add(sol.name)
    
    return unique_solutions[:max_solutions]


def _create_solution(
    template_name: str,
    intensity: float = 0.7,
    reasoning: str = "",
) -> SolutionVariant:
    """从模板创建一个方案实例"""
    template = SOLUTION_TEMPLATES.get(template_name)
    if not template:
        raise ValueError(f"Unknown solution template: {template_name}")
    
    style_adjustments = template.get("style_adjustments", {}).copy()
    retouch_overrides = template.get("retouch_overrides", {}).copy()
    
    # 根据强度调整参数
    # 对于亮度和对比度，接近1.0时减弱强度
    for key in style_adjustments:
        base_val = style_adjustments[key]
        if base_val > 1.0:
            # 增强方向：向1.0靠近
            style_adjustments[key] = 1.0 + (base_val - 1.0) * intensity
        elif base_val < 1.0:
            # 削弱方向：向1.0靠近
            style_adjustments[key] = base_val + (1.0 - base_val) * (1.0 - intensity)
    
    # 精修参数直接乘以强度
    for key in retouch_overrides:
        retouch_overrides[key] *= intensity
    
    # 默认理由文本
    default_reasonings = {
        "color_correction": "检测到色温偏差，纠正为自然色彩。这是所有图片的基础优化。",
        "skin_tone_enhance": "检测到人像肤色欠佳，通过增加亮度和红润感来提升气色。",
        "contrast_pop": "检测到对比度不足，增强明暗对比使画面更有冲击力和深度感。",
        "vibrance_boost": "检测到色彩欠缺活力，提升饱和度让色彩更生动亮丽。",
        "portrait_retouch": "这是人像的专业修饰方案，综合应用磨皮、美白、塑形等技术。",
        "cinematic_grade": "采用电影分级风格，创造更具艺术感和专业质感的视觉效果。",
        "detail_sharpen": "增强细节清晰度，让纹理更加分明，提升整体质感。",
        # Stylized grading reasonings
        "warm_golden_grade": "画面呈现暖色调特征，黄金时刻风格可强化橙红高光并柔化暗部，营造温暖氛围。",
        "teal_orange_grade": "采用专业电影青橙分色：暗部偏青蓝、高光偏橙黄，是最受认可的电影调色风格之一。",
        "vintage_fade_grade": "复古胶片风格：提升暗部基调（褪色感）并加入颗粒和暗角，营造胶片年代质感。",
        "bright_airy_grade": "高亮低对比的空气感风格：柔和的白色高光区配合轻暖调，适合清新时尚类题材。",
        "moody_dark_grade": "暗调情绪风格：压低整体亮度并强化对比度与暗角，配合轻冷色调营造戏剧张力。",
    }
    
    if not reasoning:
        reasoning = default_reasonings.get(template_name, "智能推荐的修改方案")
    
    return SolutionVariant(
        name=template_name,
        display_name=template.get("display_name", template_name),
        description=template.get("description", ""),
        reasoning=reasoning,
        style_adjustments=style_adjustments,
        retouch_overrides=retouch_overrides,
        applicable_scenes=template.get("applicable_scenes", []),
        intensity=intensity,
    )


# ============================================================================
# 方案应用函数（由主模块调用）
# ============================================================================

def get_solution_by_name(name: str) -> Optional[SolutionVariant]:
    """获取指定名称的方案"""
    if name not in SOLUTION_TEMPLATES:
        return None
    return _create_solution(name, intensity=0.7)


def solutions_to_ui_format(solutions: List[SolutionVariant]) -> str:
    """将方案列表转为UI展示格式（JSON字符串）"""
    data = []
    for sol in solutions:
        data.append({
            "name": sol.name,
            "display_name": sol.display_name,
            "description": sol.description,
            "reasoning": sol.reasoning,
            "intensity": sol.intensity,
        })
    return json.dumps(data, ensure_ascii=False, indent=2)
