import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from ai_auto_ps import (
    RETOUCH_CONTROL_KEYS,
    RETOUCH_PROFILE_PRESETS,
    _apply_auto_geometry_to_pil,
    _build_analysis_reason,
    _decide_auto_geometry,
    _extract_text_from_model_output,
    _apply_style_to_frame,
    _parse_ai_geometry_response,
    apply_style_to_pil,
    get_retouch_profile_values,
    Image,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    STYLE_PRESETS,
    choose_style_from_description,
    detect_media_type,
    double_check_implementation,
    get_advisor,
    normalize_uploaded_file_paths,
    normalize_retouch_controls,
    process_uploaded_files,
    summarize_retouch_controls,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class StyleRoutingTests(unittest.TestCase):
    def test_extract_text_from_nested_model_output(self):
        payload = [{"generated_text": [{"role": "assistant", "content": "清晰的人像照片"}]}]
        self.assertIn("清晰的人像照片", _extract_text_from_model_output(payload))

    def test_build_analysis_reason_sanitizes_control_chars(self):
        fake_analysis = SimpleNamespace(
            selected_style="portrait_soft",
            strategy="llm",
            raw_description="portrait\x00\nperson",
        )
        reason = _build_analysis_reason(fake_analysis, {"skin_smooth": 0.6}, src_name="demo.jpg")
        self.assertIn("demo.jpg | strategy=llm", reason)
        self.assertIn("description=portrait person", reason)

    def test_build_analysis_reason_contains_geometry(self):
        fake_analysis = SimpleNamespace(
            selected_style="portrait_soft",
            strategy="llm",
            raw_description="portrait person",
            auto_geometry_decision="rotate=90, crop=0.90",
        )
        reason = _build_analysis_reason(fake_analysis, {"skin_smooth": 0.6}, src_name="demo.jpg")
        self.assertIn("geometry=rotate=90, crop=0.90", reason)

    def test_decide_auto_geometry_detects_rotate_and_crop_keywords(self):
        decision = _decide_auto_geometry("建议向左旋转并裁切空白背景")
        self.assertEqual(decision["rotation"], 90.0)
        self.assertLess(decision["crop_factor"], 1.0)

    def test_parse_ai_geometry_response_valid_json(self):
        text = '{"rotation": 90, "crop_factor": 0.85, "reason": "tilted"}'
        result = _parse_ai_geometry_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["rotation"], 90.0)
        self.assertAlmostEqual(result["crop_factor"], 0.85)

    def test_parse_ai_geometry_response_embedded_in_text(self):
        text = 'The image needs adjustment. {"rotation": 180, "crop_factor": 0.9, "reason": "upside down"} Done.'
        result = _parse_ai_geometry_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["rotation"], 180.0)

    def test_parse_ai_geometry_response_clamps_crop_factor(self):
        text = '{"rotation": 0, "crop_factor": 0.3, "reason": "extreme"}'
        result = _parse_ai_geometry_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["crop_factor"], 0.5)

    def test_parse_ai_geometry_response_rejects_invalid_rotation(self):
        text = '{"rotation": 45, "crop_factor": 0.9, "reason": "odd angle"}'
        result = _parse_ai_geometry_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["rotation"], 0.0)

    def test_parse_ai_geometry_response_returns_none_for_empty(self):
        self.assertIsNone(_parse_ai_geometry_response(""))
        self.assertIsNone(_parse_ai_geometry_response("no json here"))

    def test_landscape_keywords_route_to_vivid_style(self):
        self.assertEqual(
            choose_style_from_description("A mountain landscape with blue sky"),
            "landscape_vivid",
        )

    def test_unknown_description_routes_to_clean_natural(self):
        self.assertEqual(
            choose_style_from_description("abstract composition"),
            "clean_natural",
        )

    def test_detect_media_type_accepts_expanded_image_suffix(self):
        self.assertEqual(detect_media_type("demo.jfif"), "image")

    def test_detect_media_type_accepts_expanded_video_suffix(self):
        self.assertEqual(detect_media_type("demo.mts"), "video")

    def test_supported_formats_include_expanded_entries(self):
        self.assertIn(".jfif", SUPPORTED_IMAGE_FORMATS)
        self.assertIn(".mts", SUPPORTED_VIDEO_FORMATS)

    def test_get_advisor_is_singleton(self):
        self.assertIs(get_advisor(), get_advisor())

    def test_portrait_style_contains_advanced_controls(self):
        portrait = STYLE_PRESETS["portrait_soft"]
        for key in RETOUCH_CONTROL_KEYS:
            self.assertIn(key, portrait)
        self.assertGreater(portrait["skin_smooth"], 0)

    def test_normalize_retouch_controls_clamps_values(self):
        normalized = normalize_retouch_controls({"skin_smooth": 1.8, "eye_enlarge": -0.3, "x": 0.5})
        self.assertEqual(normalized["skin_smooth"], 1.0)
        self.assertEqual(normalized["eye_enlarge"], 0.0)
        self.assertNotIn("x", normalized)

    def test_summarize_retouch_controls_contains_labels(self):
        summary = summarize_retouch_controls({"skin_smooth": 0.6, "slim_face": 0.3})
        self.assertIn("磨皮=0.60", summary)
        self.assertIn("瘦脸=0.30", summary)

    def test_retouch_profiles_cover_all_controls(self):
        for _, values in RETOUCH_PROFILE_PRESETS.items():
            for key in RETOUCH_CONTROL_KEYS:
                self.assertIn(key, values)

    def test_get_retouch_profile_values_returns_profile_values(self):
        values = get_retouch_profile_values("自然韩系")
        self.assertAlmostEqual(values["skin_smooth"], RETOUCH_PROFILE_PRESETS["自然韩系"]["skin_smooth"], places=2)
        self.assertAlmostEqual(values["nose_slim"], RETOUCH_PROFILE_PRESETS["自然韩系"]["nose_slim"], places=2)

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_apply_style_to_pil_keeps_size_for_advanced_style(self):
        sample = Image.new("RGB", (24, 24), (130, 90, 160))
        styled = apply_style_to_pil(sample, "portrait_soft")
        self.assertEqual(styled.size, sample.size)

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_apply_auto_geometry_crop_keeps_output_size(self):
        sample = Image.new("RGB", (24, 18), (130, 90, 160))
        transformed = _apply_auto_geometry_to_pil(sample, {"rotation": 0.0, "crop_factor": 0.82})
        self.assertEqual(transformed.size, sample.size)

    def test_normalize_uploaded_file_paths_from_mixed_input(self):
        class _MockFile:
            def __init__(self, name):
                self.name = name

        files = [_MockFile("a.jpg"), "b.png", None]
        self.assertEqual(normalize_uploaded_file_paths(files), ["a.jpg", "b.png"])

    def test_normalize_uploaded_file_paths_supports_path_and_dict_items(self):
        class _MockPathFile:
            def __init__(self, path):
                self.path = path

        files = [_MockPathFile("c.webp"), {"path": "d.jpg"}, {"name": "e.png"}, {"path": "f.bmp", "name": "g.png"}]
        normalized = normalize_uploaded_file_paths(files)
        self.assertEqual(normalized, ["c.webp", "d.jpg", "e.png", "f.bmp"])
        self.assertNotIn("g.png", normalized)

    def test_process_uploaded_files_empty_raises(self):
        with self.assertRaises(ValueError):
            process_uploaded_files([])

    def test_process_uploaded_files_stringifies_path_outputs(self):
        fake_output = Path("out.jpg")
        fake_analysis = SimpleNamespace(
            selected_style="portrait_soft",
            strategy="auto",
            raw_description="mock",
        )
        with (
            patch("ai_auto_ps.get_advisor", return_value=object()),
            patch("ai_auto_ps.detect_media_type", return_value="image"),
            patch("ai_auto_ps.process_image_file", return_value=(fake_output, fake_analysis)),
            patch("ai_auto_ps.double_check_implementation", return_value="ok"),
        ):
            outputs, before, after, styles, reason, check = process_uploaded_files(["demo.jpg"], "auto")

        self.assertEqual(outputs, [str(fake_output)])
        self.assertEqual(after, [str(fake_output)])
        self.assertEqual(before, ["demo.jpg"])
        self.assertIn("demo.jpg: portrait_soft", styles)
        self.assertIn("strategy=auto", reason)
        self.assertEqual(check, "ok")

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_process_uploaded_files_supports_multiple_images(self):
        with TemporaryDirectory() as tmp_dir:
            p1 = Path(tmp_dir) / "demo1.jpg"
            p2 = Path(tmp_dir) / "demo2.png"

            Image.new("RGB", (16, 16), (60, 80, 120)).save(p1)
            Image.new("RGB", (16, 16), (90, 70, 150)).save(p2)

            outputs, before, after, styles, reason, check = process_uploaded_files([str(p1), str(p2)], "auto")

            self.assertEqual(len(outputs), 2)
            self.assertEqual(len(before), 2)
            self.assertEqual(len(after), 2)
            self.assertEqual(before[0], str(p1))
            self.assertTrue(after[0].endswith(".jpg"))
            self.assertIn("demo1", styles)
            self.assertIn("strategy=", reason)
            self.assertTrue(check.startswith("两轮检查"))

    @unittest.skipIf(Image is None, "pillow not installed")
    def test_process_uploaded_files_accepts_manual_retouch_controls(self):
        with TemporaryDirectory() as tmp_dir:
            p1 = Path(tmp_dir) / "portrait.jpg"
            Image.new("RGB", (18, 18), (100, 90, 130)).save(p1)

            _, _, _, _, reason, _ = process_uploaded_files(
                [str(p1)],
                "portrait_soft",
                retouch_controls={"skin_smooth": 0.9, "eye_enlarge": 0.55},
            )

            self.assertIn("retouch=", reason)
            self.assertIn("磨皮=0.90", reason)

    def test_double_check_reports_success(self):
        result = double_check_implementation()
        if Image is None:
            self.assertIn("两轮检查未通过", result)
        else:
            self.assertIn("两轮检查通过", result)

    @unittest.skipIf(np is None, "numpy not installed")
    def test_apply_style_to_frame_keeps_shape_and_dtype(self):
        frame = np.full((8, 8, 3), 64, dtype=np.uint8)
        styled = _apply_style_to_frame(frame, "landscape_vivid")
        self.assertEqual(styled.shape, frame.shape)
        self.assertEqual(styled.dtype, frame.dtype)


if __name__ == "__main__":
    unittest.main()
