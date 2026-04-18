import unittest

from ai_auto_ps import (
    _apply_style_to_frame,
    Image,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    choose_style_from_description,
    detect_media_type,
    double_check_implementation,
    get_advisor,
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
