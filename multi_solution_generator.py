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
    ai_geometry: Optional[Dict] = None  # AI判断的几何变换 {"rotation": float, "crop_factor": float}


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
}


# ============================================================================
# 多方案生成函数
# ============================================================================

def generate_multiple_solutions(
    analysis_result: EnhancedAnalysisResult,
    max_solutions: int = 4,
) -> List[SolutionVariant]:
    """
    根据分析结果生成多个修改方案
    
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
    
    # 2. 根据场景推荐专化方案
    if scene == "portrait":
        # 人像优先推荐肤色优化和精修
        solutions.append(_create_solution("skin_tone_enhance", intensity=0.75))
        solutions.append(_create_solution("portrait_retouch", intensity=0.65))
        solutions.append(_create_solution("cinematic_grade", intensity=0.55))
        
    elif scene == "landscape":
        # 风景推荐对比和活力
        solutions.append(_create_solution("vibrance_boost", intensity=0.8))
        solutions.append(_create_solution("contrast_pop", intensity=0.7))
        solutions.append(_create_solution("detail_sharpen", intensity=0.6))
        
    elif scene == "food":
        # 美食推荐活力和对比
        solutions.append(_create_solution("vibrance_boost", intensity=0.85))
        solutions.append(_create_solution("detail_sharpen", intensity=0.7))
        solutions.append(_create_solution("contrast_pop", intensity=0.65))
        
    elif scene == "night":
        # 夜景推荐对比和清晰
        solutions.append(_create_solution("contrast_pop", intensity=0.8))
        solutions.append(_create_solution("detail_sharpen", intensity=0.75))
        solutions.append(_create_solution("cinematic_grade", intensity=0.7))
    
    else:
        # 默认：活力 + 对比 + 精修
        solutions.append(_create_solution("vibrance_boost", intensity=0.7))
        solutions.append(_create_solution("contrast_pop", intensity=0.65))
        if "portrait" in recommended_dirs or any(s in analysis_result.subjects for s in ["person", "face"]):
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
