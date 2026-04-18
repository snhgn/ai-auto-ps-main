import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_auto_ps import (
    _apply_style_to_frame,
    Image,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    choose_style_from_description,
    detect_media_type,
    double_check_implementation,
    get_advisor,
    normalize_uploaded_file_paths,
    process_uploaded_files,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class StyleRoutingTests(unittest.TestCase):
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

    def test_normalize_uploaded_file_paths_from_mixed_input(self):
        class _MockFile:
            def __init__(self, name):
                self.name = name

        files = [_MockFile("a.jpg"), "b.png", None]
        self.assertEqual(normalize_uploaded_file_paths(files), ["a.jpg", "b.png"])

    def test_process_uploaded_files_empty_raises(self):
        with self.assertRaises(ValueError):
            process_uploaded_files([])

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
            self.assertIn("demo1", styles)
            self.assertIn("strategy=", reason)
            self.assertTrue(check.startswith("两轮检查"))

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
