import unittest

from ai_auto_ps import (
    Image,
    STYLE_PRESETS,
    STYLE_HINTS,
    STYLIZED_GRADE_KEYS,
    LightweightStyleAdvisor,
    _apply_stylized_grading_pil,
    _resolve_style_values,
    apply_style_to_pil,
    choose_style_from_description,
    _convert_to_enhanced,
    AnalysisResult,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from multi_solution_generator import (
    SOLUTION_TEMPLATES,
    EnhancedAnalysisResult,
    generate_multiple_solutions,
)


class StylizedPresetsTests(unittest.TestCase):
    def test_new_presets_in_style_presets(self):
        for name in ("warm_golden", "teal_orange", "vintage_fade", "bright_airy", "moody_dark"):
            self.assertIn(name, STYLE_PRESETS, f"Missing preset: {name}")

    def test_stylized_presets_have_required_keys(self):
        for name in ("warm_golden", "teal_orange", "vintage_fade", "bright_airy", "moody_dark"):
            preset = STYLE_PRESETS[name]
            for key in ("brightness", "contrast", "color"):
                self.assertIn(key, preset, f"{name} missing {key}")

    def test_stylized_grade_keys_constant(self):
        for key in (
            "shadows_lift", "highlights_pull", "warm_tint",
            "teal_shadow_tint", "vignette", "clarity", "grain",
            "r_gain", "g_gain", "b_gain",
        ):
            self.assertIn(key, STYLIZED_GRADE_KEYS)

    def test_resolve_style_values_copies_grade_keys(self):
        vals = _resolve_style_values("warm_golden")
        self.assertIn("warm_tint", vals)
        self.assertAlmostEqual(vals["warm_tint"], STYLE_PRESETS["warm_golden"]["warm_tint"])

    def test_resolve_style_values_for_teal_orange(self):
        vals = _resolve_style_values("teal_orange")
        self.assertIn("teal_shadow_tint", vals)
        self.assertIn("vignette", vals)

    def test_resolve_style_values_for_vintage_fade_has_grain(self):
        vals = _resolve_style_values("vintage_fade")
        self.assertIn("grain", vals)
        self.assertGreater(vals["grain"], 0.0)


class StyleHintsTests(unittest.TestCase):
    def test_golden_routes_to_warm_golden(self):
        self.assertEqual(choose_style_from_description("golden sunset photo"), "warm_golden")

    def test_warm_routes_to_warm_golden(self):
        self.assertEqual(choose_style_from_description("warm outdoor scene"), "warm_golden")

    def test_vintage_routes_to_vintage_fade(self):
        self.assertEqual(choose_style_from_description("vintage retro style"), "vintage_fade")

    def test_moody_routes_to_moody_dark(self):
        self.assertEqual(choose_style_from_description("moody dramatic portrait"), "moody_dark")

    def test_airy_routes_to_bright_airy(self):
        self.assertEqual(choose_style_from_description("bright airy light photo"), "bright_airy")

    def test_teal_routes_to_teal_orange(self):
        self.assertEqual(choose_style_from_description("teal blue scene"), "teal_orange")


class ConvertToEnhancedTests(unittest.TestCase):
    def _make_basic(self, style="clean_natural"):
        return AnalysisResult(description="test", selected_style=style, strategy="manual")

    def test_warm_golden_scene_detected(self):
        result = _convert_to_enhanced(self._make_basic(), "warm golden sunset outdoor scene")
        self.assertEqual(result.scene, "warm")
        self.assertIn("warm_golden_grade", result.recommended_directions)

    def test_moody_scene_detected(self):
        result = _convert_to_enhanced(self._make_basic(), "dramatic moody desaturated scene")
        self.assertEqual(result.scene, "moody")
        self.assertIn("moody_dark_grade", result.recommended_directions)

    def test_airy_scene_detected(self):
        result = _convert_to_enhanced(self._make_basic(), "bright airy minimal scene")
        self.assertEqual(result.scene, "airy")
        self.assertIn("bright_airy_grade", result.recommended_directions)

    def test_portrait_scene_includes_stylized_options(self):
        result = _convert_to_enhanced(self._make_basic(), "portrait person photo")
        self.assertIn("warm_golden_grade", result.recommended_directions)
        self.assertIn("bright_airy_grade", result.recommended_directions)

    def test_landscape_scene_includes_teal_orange(self):
        result = _convert_to_enhanced(self._make_basic(), "high-color landscape photo")
        self.assertIn("teal_orange_grade", result.recommended_directions)


@unittest.skipIf(np is None or Image is None, "numpy/pillow not installed")
class StylizedGradingFunctionsTests(unittest.TestCase):
    def _make_image(self, r=128, g=128, b=128, size=(16, 16)):
        return Image.new("RGB", size, (r, g, b))

    def test_warm_tint_raises_red_lowers_blue(self):
        img = self._make_image(128, 128, 128)
        out = _apply_stylized_grading_pil(img, {"warm_tint": 0.5})
        arr = np.asarray(out)
        # Red channel should increase, blue should decrease
        self.assertGreater(int(arr[:, :, 0].mean()), 128)
        self.assertLess(int(arr[:, :, 2].mean()), 128)

    def test_cool_tint_lowers_red_raises_blue(self):
        img = self._make_image(128, 128, 128)
        out = _apply_stylized_grading_pil(img, {"warm_tint": -0.5})
        arr = np.asarray(out)
        self.assertLess(int(arr[:, :, 0].mean()), 128)
        self.assertGreater(int(arr[:, :, 2].mean()), 128)

    def test_shadows_lift_raises_dark_pixels(self):
        img = self._make_image(50, 50, 50)  # dark image
        out = _apply_stylized_grading_pil(img, {"shadows_lift": 0.5})
        arr = np.asarray(out)
        self.assertGreater(arr.mean(), 50)

    def test_highlights_pull_lowers_bright_pixels(self):
        img = self._make_image(230, 230, 230)  # bright image
        out = _apply_stylized_grading_pil(img, {"highlights_pull": 0.8})
        arr = np.asarray(out)
        self.assertLess(arr.mean(), 230)

    def test_vignette_darkens_edges(self):
        img = self._make_image(200, 200, 200, size=(64, 64))
        out = _apply_stylized_grading_pil(img, {"vignette": 0.8})
        arr = np.asarray(out, dtype=np.float32)
        # Corner pixels should be darker than center
        center_mean = arr[28:36, 28:36].mean()
        corner_mean = (
            arr[:8, :8].mean() + arr[:8, -8:].mean() +
            arr[-8:, :8].mean() + arr[-8:, -8:].mean()
        ) / 4.0
        self.assertGreater(center_mean, corner_mean)

    def test_teal_shadow_tint_shifts_dark_pixels(self):
        img = self._make_image(60, 60, 60)  # dark image – should receive teal
        out = _apply_stylized_grading_pil(img, {"teal_shadow_tint": 0.8})
        arr = np.asarray(out)
        # G and B channels should increase, R should decrease
        self.assertGreater(int(arr[:, :, 1].mean()), 60)
        self.assertGreater(int(arr[:, :, 2].mean()), 60)
        self.assertLess(int(arr[:, :, 0].mean()), 60)

    def test_output_size_unchanged(self):
        img = self._make_image(size=(32, 48))
        out = _apply_stylized_grading_pil(img, {"warm_tint": 0.5, "vignette": 0.4, "shadows_lift": 0.3})
        self.assertEqual(out.size, img.size)

    def test_apply_style_to_pil_warm_golden_keeps_size(self):
        img = self._make_image(120, 100, 80, size=(24, 24))
        out = apply_style_to_pil(img, "warm_golden")
        self.assertEqual(out.size, img.size)

    def test_apply_style_to_pil_teal_orange_keeps_size(self):
        img = self._make_image(100, 110, 130, size=(24, 24))
        out = apply_style_to_pil(img, "teal_orange")
        self.assertEqual(out.size, img.size)

    def test_apply_style_to_pil_moody_dark_keeps_size(self):
        img = self._make_image(160, 140, 120, size=(24, 24))
        out = apply_style_to_pil(img, "moody_dark")
        self.assertEqual(out.size, img.size)


class SolutionTemplatesTests(unittest.TestCase):
    def test_new_templates_in_solution_templates(self):
        for name in (
            "warm_golden_grade", "teal_orange_grade",
            "vintage_fade_grade", "bright_airy_grade", "moody_dark_grade",
        ):
            self.assertIn(name, SOLUTION_TEMPLATES, f"Missing template: {name}")

    def test_new_templates_have_stylized_keys(self):
        warm = SOLUTION_TEMPLATES["warm_golden_grade"]["style_adjustments"]
        self.assertIn("warm_tint", warm)
        self.assertIn("vignette", warm)

        teal = SOLUTION_TEMPLATES["teal_orange_grade"]["style_adjustments"]
        self.assertIn("teal_shadow_tint", teal)

        vintage = SOLUTION_TEMPLATES["vintage_fade_grade"]["style_adjustments"]
        self.assertIn("grain", vintage)

    def test_portrait_scene_solutions_include_stylized(self):
        analysis = EnhancedAnalysisResult(
            raw_description="portrait person photo",
            scene="portrait",
            subjects=["person", "face"],
            lighting={},
            color_profile={},
            recommended_directions=["warm_golden_grade", "bright_airy_grade"],
            selected_style="portrait_soft",
            strategy="heuristic_fallback",
        )
        solutions = generate_multiple_solutions(analysis, max_solutions=4)
        names = [s.name for s in solutions]
        stylized = [n for n in names if n.endswith("_grade")]
        self.assertGreater(len(stylized), 0, "Expected at least one stylized grade solution")

    def test_landscape_scene_solutions_include_teal_orange(self):
        analysis = EnhancedAnalysisResult(
            raw_description="landscape photo",
            scene="landscape",
            subjects=["landscape"],
            lighting={},
            color_profile={},
            recommended_directions=["teal_orange_grade", "moody_dark_grade"],
            selected_style="landscape_vivid",
            strategy="heuristic_fallback",
        )
        solutions = generate_multiple_solutions(analysis, max_solutions=4)
        names = [s.name for s in solutions]
        self.assertIn("teal_orange_grade", names)

    def test_night_scene_solutions_include_moody_or_teal(self):
        analysis = EnhancedAnalysisResult(
            raw_description="night scene",
            scene="night",
            subjects=[],
            lighting={},
            color_profile={},
            recommended_directions=["teal_orange_grade", "moody_dark_grade"],
            selected_style="night_clarity",
            strategy="heuristic_fallback",
        )
        solutions = generate_multiple_solutions(analysis, max_solutions=4)
        names = [s.name for s in solutions]
        self.assertTrue(
            "teal_orange_grade" in names or "moody_dark_grade" in names,
            f"Expected teal/moody grade in night solutions, got {names}",
        )

    def test_generate_respects_max_solutions(self):
        analysis = EnhancedAnalysisResult(
            raw_description="portrait",
            scene="portrait",
            subjects=["person"],
            lighting={},
            color_profile={},
            recommended_directions=[],
            selected_style="portrait_soft",
            strategy="heuristic_fallback",
        )
        for n in (1, 2, 3, 4):
            solutions = generate_multiple_solutions(analysis, max_solutions=n)
            self.assertLessEqual(len(solutions), n)


class HeuristicDescriptionTests(unittest.TestCase):
    @unittest.skipIf(Image is None or np is None, "pillow/numpy not installed")
    def test_warm_image_produces_golden_description(self):
        # Image with high R, low B → warm golden
        img = Image.new("RGB", (32, 32), (210, 150, 90))
        advisor = LightweightStyleAdvisor()
        desc = advisor._heuristic_description(img)
        self.assertIn("warm", desc.lower())

    @unittest.skipIf(Image is None or np is None, "pillow/numpy not installed")
    def test_dark_image_produces_night_description(self):
        img = Image.new("RGB", (32, 32), (40, 40, 40))
        advisor = LightweightStyleAdvisor()
        desc = advisor._heuristic_description(img)
        self.assertIn("night", desc.lower())

    @unittest.skipIf(Image is None or np is None, "pillow/numpy not installed")
    def test_bright_low_saturation_produces_airy_description(self):
        img = Image.new("RGB", (32, 32), (240, 238, 235))
        advisor = LightweightStyleAdvisor()
        desc = advisor._heuristic_description(img)
        self.assertIn("airy", desc.lower())


if __name__ == "__main__":
    unittest.main()
