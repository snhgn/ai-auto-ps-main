"""
多版本解决方案管理器 - 支持多版本对比、用户反馈、迭代优化
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

BRIGHTNESS_LOWER_MIN = 0.85
BRIGHTNESS_LOWER_MULTIPLIER = 0.92
BRIGHTNESS_HIGHER_MAX = 1.20
BRIGHTNESS_HIGHER_MULTIPLIER = 1.08

SATURATION_LOWER_MIN = 0.80
SATURATION_LOWER_MULTIPLIER = 0.90
SATURATION_HIGHER_MAX = 1.35
SATURATION_HIGHER_MULTIPLIER = 1.10

NATURAL_STYLE_CONTRAST_MIN = 1.0
NATURAL_STYLE_CONTRAST_MULTIPLIER = 0.90
NATURAL_STYLE_COLOR_MIN = 0.90
NATURAL_STYLE_COLOR_MULTIPLIER = 0.92


@dataclass
class SolutionVersion:
    """单个方案版本"""
    solution_name: str             # "color_correction" 等
    display_name: str              # "色温纠正"
    description: str               # 方案描述
    reasoning: str                 # AI 推荐理由
    intensity: float               # 强度 0.3-1.0
    style_adjustments: Dict[str, float]  # 调整参数
    retouch_overrides: Dict[str, float]  # 精修参数
    output_path: Optional[str] = None    # 生成的输出路径
    generation_round: int = 1           # 第几轮生成（迭代轮数）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserFeedback:
    """用户反馈"""
    feedback_text: str                   # 用户的评论/反馈
    target_solution: str                 # 反馈针对的方案名称
    sentiment: str = "neutral"           # "positive"/"negative"/"neutral"
    requested_adjustments: Dict[str, str] = field(default_factory=dict)  # 请求的调整 {"亮度": "偏暗", "饱和度": "太高"}
    prefer_stronger_model: bool = False  # 是否请求使用更强的 AI 模型
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SolutionSession:
    """一个完整的修图会话"""
    session_id: str                      # 会话唯一ID
    input_image_path: str                # 输入图片路径
    analysis_result: Optional[str] = None    # 图像分析结果（JSON）
    analysis_reasoning: str = ""         # 详细分析理由
    
    # 版本管理
    versions: List[SolutionVersion] = field(default_factory=list)  # 所有版本
    current_round: int = 1               # 当前轮数
    
    # 反馈与迭代
    feedbacks: List[UserFeedback] = field(default_factory=list)  # 所有反馈
    model_history: List[str] = field(default_factory=list)       # 使用过的模型列表
    
    # 会话元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PreferenceMemoryEntry:
    """用户偏好记忆条目（按更新时间覆盖冲突偏好）"""
    key: str
    value: str
    source_feedback: str
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SolutionManager:
    """管理多版本方案、反馈和迭代"""
    
    def __init__(self):
        self.sessions: Dict[str, SolutionSession] = {}
        self.current_session_id: Optional[str] = None
        self.preference_memory: Dict[str, PreferenceMemoryEntry] = {}
    
    def create_session(self, session_id: str, input_image_path: str) -> SolutionSession:
        """创建新的修图会话"""
        session = SolutionSession(
            session_id=session_id,
            input_image_path=input_image_path,
        )
        self.sessions[session_id] = session
        self.current_session_id = session_id
        return session
    
    def add_version(
        self,
        session_id: str,
        solution_name: str,
        display_name: str,
        description: str,
        reasoning: str,
        intensity: float,
        style_adjustments: Dict[str, float],
        retouch_overrides: Dict[str, float],
        output_path: Optional[str] = None,
    ) -> SolutionVersion:
        """向会话中添加一个新版本"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        version = SolutionVersion(
            solution_name=solution_name,
            display_name=display_name,
            description=description,
            reasoning=reasoning,
            intensity=intensity,
            style_adjustments=style_adjustments,
            retouch_overrides=retouch_overrides,
            output_path=output_path,
            generation_round=session.current_round,
        )
        
        session.versions.append(version)
        session.last_modified = datetime.now().isoformat()
        return version
    
    def add_feedback(
        self,
        session_id: str,
        feedback_text: str,
        target_solution: str,
        sentiment: str = "neutral",
        requested_adjustments: Optional[Dict[str, str]] = None,
        prefer_stronger_model: bool = False,
    ) -> UserFeedback:
        """添加用户反馈"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        feedback = UserFeedback(
            feedback_text=feedback_text,
            target_solution=target_solution,
            sentiment=sentiment,
            requested_adjustments=requested_adjustments or {},
            prefer_stronger_model=prefer_stronger_model,
        )
        
        session.feedbacks.append(feedback)
        session.last_modified = datetime.now().isoformat()
        
        # 如果反馈要求使用更强模型，记录下来
        if prefer_stronger_model and "large_model" not in session.model_history:
            session.model_history.append("large_model")
        
        return feedback
    
    def start_new_round(self, session_id: str) -> int:
        """开始新一轮生成（迭代）"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.current_round += 1
        session.last_modified = datetime.now().isoformat()
        return session.current_round
    
    def get_versions_for_comparison(self, session_id: str) -> List[SolutionVersion]:
        """获取当前会话的所有版本（用于对比）"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.sessions[session_id].versions
    
    def get_feedback_summary(self, session_id: str) -> str:
        """获取反馈总结"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        if not session.feedbacks:
            return "暂无反馈"
        
        summary = f"共收到 {len(session.feedbacks)} 条反馈：\n"
        for i, fb in enumerate(session.feedbacks, 1):
            summary += f"\n[反馈 {i}] {fb.created_at}\n"
            summary += f"  目标方案: {fb.target_solution}\n"
            summary += f"  评价: {fb.sentiment}\n"
            summary += f"  内容: {fb.feedback_text}\n"
            if fb.requested_adjustments:
                summary += f"  请求调整: {json.dumps(fb.requested_adjustments, ensure_ascii=False)}\n"
            if fb.prefer_stronger_model:
                summary += f"  ✓ 请求使用更强 AI 模型\n"
        
        return summary
    
    def get_iteration_suggestions(self, session_id: str) -> Dict[str, str]:
        """根据反馈生成迭代建议"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        suggestions = {}
        
        if not session.feedbacks:
            return suggestions
        
        # 分析最近的反馈
        latest_feedback = session.feedbacks[-1]
        
        if latest_feedback.prefer_stronger_model:
            suggestions["model"] = "使用更强的AI模型（LLaVA-1.6-13B）以获得更精准的分析和更好的效果"
        
        if latest_feedback.requested_adjustments:
            adjustments = latest_feedback.requested_adjustments
            adj_text = "建议的参数调整：\n"
            for param, change in adjustments.items():
                adj_text += f"  • {param}: {change}\n"
            suggestions["adjustments"] = adj_text
        
        if latest_feedback.sentiment == "negative":
            suggestions["approach"] = "用户对当前方案不满意，建议尝试完全不同的修图风格（如从艺术风改为自然风）"
        elif latest_feedback.sentiment == "positive":
            suggestions["approach"] = "用户满意，建议在此基础上进行微调以进一步优化"
        
        return suggestions
    
    def export_session_report(self, session_id: str) -> str:
        """导出会话的完整报告（Markdown 格式）"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        report = f"# 修图会话报告\n\n"
        report += f"**会话ID**: {session.session_id}\n"
        report += f"**创建时间**: {session.created_at}\n"
        report += f"**最后修改**: {session.last_modified}\n"
        report += f"**迭代轮数**: {session.current_round}\n\n"
        
        if session.analysis_reasoning:
            report += f"## 图像分析\n\n{session.analysis_reasoning}\n\n"
        
        if session.versions:
            report += f"## 生成的版本 ({len(session.versions)} 个)\n\n"
            for i, ver in enumerate(session.versions, 1):
                report += f"### 版本 {i}: {ver.display_name}\n"
                report += f"- 方案: {ver.solution_name}\n"
                report += f"- 轮数: {ver.generation_round}\n"
                report += f"- 强度: {ver.intensity:.1%}\n"
                report += f"- 理由: {ver.reasoning}\n"
                if ver.output_path:
                    report += f"- 输出: {ver.output_path}\n"
                report += "\n"
        
        if session.feedbacks:
            report += f"## 用户反馈 ({len(session.feedbacks)} 条)\n\n"
            for i, fb in enumerate(session.feedbacks, 1):
                report += f"### 反馈 {i}\n"
                report += f"- 时间: {fb.created_at}\n"
                report += f"- 目标: {fb.target_solution}\n"
                report += f"- 评价: {fb.sentiment}\n"
                report += f"- 内容: {fb.feedback_text}\n"
                if fb.requested_adjustments:
                    report += f"- 请求调整: {json.dumps(fb.requested_adjustments, ensure_ascii=False)}\n"
                report += "\n"
        
        if session.model_history:
            report += f"## 使用的模型\n\n{', '.join(session.model_history)}\n"
        
        if self.preference_memory:
            report += "\n## 用户偏好记忆库\n\n"
            for key in sorted(self.preference_memory.keys()):
                entry = self.preference_memory[key]
                report += f"- {entry.key}: {entry.value} (更新时间: {entry.updated_at})\n"
        
        return report
    
    def update_preference_memory(
        self,
        preferences: Dict[str, str],
        source_feedback: str,
        updated_at: Optional[str] = None,
    ) -> Dict[str, PreferenceMemoryEntry]:
        """更新用户偏好记忆，冲突项按最新时间戳覆盖并清理旧值"""
        timestamp = updated_at or datetime.now().isoformat()
        for key, value in (preferences or {}).items():
            if value is None:
                continue
            normalized_value = str(value).strip()
            if not normalized_value:
                continue
            self.preference_memory[key] = PreferenceMemoryEntry(
                key=key,
                value=normalized_value,
                source_feedback=source_feedback,
                updated_at=timestamp,
            )
        return dict(self.preference_memory)
    
    def get_preference_memory_summary(self) -> str:
        """返回偏好记忆可读摘要"""
        if not self.preference_memory:
            return "暂无偏好记忆"
        lines = ["当前偏好记忆："]
        for key in sorted(self.preference_memory.keys()):
            entry = self.preference_memory[key]
            lines.append(f"- {entry.key}={entry.value}（{entry.updated_at}）")
        return "\n".join(lines)
    
    def apply_memory_preferences(
        self,
        style_adjustments: Dict[str, float],
        solution_name: str,
    ) -> Dict[str, float]:
        """应用偏好记忆到调色参数（后写入优先，避免冲突叠加）"""
        adjusted = dict(style_adjustments or {})
        brightness_pref = self.preference_memory.get("brightness")
        saturation_pref = self.preference_memory.get("saturation")
        style_pref = self.preference_memory.get("style")
        
        if brightness_pref:
            if brightness_pref.value == "lower":
                adjusted["brightness"] = max(BRIGHTNESS_LOWER_MIN, adjusted.get("brightness", 1.0) * BRIGHTNESS_LOWER_MULTIPLIER)
            elif brightness_pref.value == "higher":
                adjusted["brightness"] = min(BRIGHTNESS_HIGHER_MAX, adjusted.get("brightness", 1.0) * BRIGHTNESS_HIGHER_MULTIPLIER)
        
        if saturation_pref:
            if saturation_pref.value == "lower":
                adjusted["color"] = max(SATURATION_LOWER_MIN, adjusted.get("color", 1.0) * SATURATION_LOWER_MULTIPLIER)
            elif saturation_pref.value == "higher":
                adjusted["color"] = min(SATURATION_HIGHER_MAX, adjusted.get("color", 1.0) * SATURATION_HIGHER_MULTIPLIER)
        
        if style_pref and style_pref.value == "natural" and solution_name in {"cinematic_grade", "contrast_pop"}:
            adjusted["contrast"] = max(NATURAL_STYLE_CONTRAST_MIN, adjusted.get("contrast", 1.0) * NATURAL_STYLE_CONTRAST_MULTIPLIER)
            adjusted["color"] = max(NATURAL_STYLE_COLOR_MIN, adjusted.get("color", 1.0) * NATURAL_STYLE_COLOR_MULTIPLIER)
        
        return adjusted
    
    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session_id == session_id:
                self.current_session_id = None


# 全局单例
_manager = SolutionManager()

def get_manager() -> SolutionManager:
    """获取全局的 SolutionManager 实例"""
    return _manager
