import unittest

from ai_auto_ps import Image, LightweightStyleAdvisor, _merge_collaborative_style
from solution_manager import SolutionManager


class SolutionMemoryTests(unittest.TestCase):
    def test_preference_memory_overwrites_existing_key_with_latest_timestamp(self):
        manager = SolutionManager()
        manager.update_preference_memory(
            {"brightness": "higher"},
            source_feedback="有点暗",
            updated_at="2026-01-01T00:00:00",
        )
        manager.update_preference_memory(
            {"brightness": "lower"},
            source_feedback="太亮了",
            updated_at="2026-01-02T00:00:00",
        )
        self.assertEqual(manager.preference_memory["brightness"].value, "lower")
        self.assertEqual(len(manager.preference_memory), 1)

    def test_apply_memory_preferences_adjusts_style_values(self):
        manager = SolutionManager()
        manager.update_preference_memory(
            {"brightness": "lower", "saturation": "lower", "style": "natural"},
            source_feedback="偏好自然",
        )
        adjusted = manager.apply_memory_preferences(
            {"brightness": 1.1, "contrast": 1.2, "color": 1.2},
            solution_name="cinematic_grade",
        )
        self.assertLess(adjusted["brightness"], 1.1)
        self.assertLess(adjusted["color"], 1.2)
        self.assertLess(adjusted["contrast"], 1.2)

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_merge_collaborative_style_prefers_secondary_for_bright_landscape(self):
        image = Image.new("RGB", (12, 12), (245, 245, 245))
        merged = _merge_collaborative_style("cinematic", "landscape_vivid", image)
        self.assertEqual(merged, "landscape_vivid")

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_advisor_analyze_uses_dual_model_strategy(self):
        advisor = LightweightStyleAdvisor()
        advisor._captioner = lambda image, **kw: [{"generated_text": "city night street"}]
        advisor._heuristic_description = lambda image: "high-color landscape photo"
        image = Image.new("RGB", (8, 8), (250, 250, 250))
        analysis = advisor.analyze(image, requested_style="auto")
        self.assertEqual(analysis.strategy, "dual_model_collaboration")
        self.assertEqual(analysis.selected_style, "landscape_vivid")

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_advisor_analyze_attaches_ai_geometry_when_captioner_returns_json(self):
        geometry_json = '{"rotation": 90, "crop_factor": 0.88, "reason": "tilted"}'

        def _mock_captioner(image, **kw):
            # Return geometry JSON when geometry prompt is given, else plain description
            import ai_auto_ps as _m
            text = kw.get("text", "")
            if "rotation" in text or "crop_factor" in text:
                return [{"generated_text": geometry_json}]
            return [{"generated_text": "portrait person photo"}]

        advisor = LightweightStyleAdvisor()
        advisor._captioner = _mock_captioner
        advisor._heuristic_description = lambda image: "portrait person photo"
        image = Image.new("RGB", (8, 8), (100, 80, 90))
        analysis = advisor.analyze(image, requested_style="auto")
        ai_geom = getattr(analysis, "ai_geometry", None)
        self.assertIsNotNone(ai_geom)
        self.assertEqual(ai_geom["rotation"], 90.0)
        self.assertAlmostEqual(ai_geom["crop_factor"], 0.88)

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_advisor_analyze_falls_back_to_no_geometry_when_captioner_unavailable(self):
        advisor = LightweightStyleAdvisor()
        advisor._captioner = None
        advisor._heuristic_description = lambda image: "portrait person photo"
        image = Image.new("RGB", (8, 8), (100, 80, 90))
        analysis = advisor.analyze(image, requested_style="auto")
        ai_geom = getattr(analysis, "ai_geometry", None)
        self.assertIsNone(ai_geom)


if __name__ == "__main__":
    unittest.main()
