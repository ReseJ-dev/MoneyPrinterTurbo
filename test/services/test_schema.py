import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import MaterialMatchingMode, VideoAspect, VideoParams


class TestVideoAspect(unittest.TestCase):
    def test_to_resolution_known_aspects(self):
        self.assertEqual(VideoAspect.landscape.to_resolution(), (1920, 1080))
        self.assertEqual(VideoAspect.portrait.to_resolution(), (1080, 1920))
        self.assertEqual(VideoAspect.square.to_resolution(), (1080, 1080))

    def test_to_resolution_rejects_unsupported_value(self):
        with self.assertRaises(ValueError):
            VideoAspect.to_resolution("4:5")


class TestVideoParams(unittest.TestCase):
    def test_material_matching_mode_defaults_to_fast_and_serializes(self):
        params = VideoParams(video_subject="Coffee")

        self.assertEqual(
            params.resolved_material_matching_mode,
            MaterialMatchingMode.fast,
        )
        self.assertEqual(
            params.model_dump(mode="json")["material_matching_mode"],
            "fast",
        )

    def test_rejects_unknown_material_matching_mode(self):
        with self.assertRaises(ValidationError):
            VideoParams(video_subject="Coffee", material_matching_mode="bestest")

    def test_legacy_script_matching_flag_remains_ordered_fast_mode(self):
        params = VideoParams(
            video_subject="Coffee",
            match_materials_to_script=True,
        )

        self.assertEqual(params.material_matching_mode, MaterialMatchingMode.fast)
        self.assertTrue(params.uses_legacy_script_matching)
        self.assertFalse(params.uses_visual_scene_matching)

    def test_legacy_scene_planning_maps_to_better_mode(self):
        params = VideoParams(
            video_subject="Coffee",
            visual_scene_planning=True,
        )

        self.assertEqual(params.material_matching_mode, MaterialMatchingMode.better)
        self.assertTrue(params.uses_visual_scene_matching)

    def test_explicit_new_mode_takes_precedence_over_legacy_flags(self):
        params = VideoParams(
            video_subject="Coffee",
            material_matching_mode="strict",
            match_materials_to_script=True,
            visual_scene_planning=False,
        )

        self.assertEqual(params.material_matching_mode, MaterialMatchingMode.strict)
        self.assertTrue(params.uses_strict_visual_qa)
        self.assertFalse(params.uses_legacy_script_matching)

    def test_visual_scene_planning_is_opt_in(self):
        params = VideoParams(video_subject="Coffee")

        self.assertFalse(params.visual_scene_planning)

    def test_rejects_non_positive_generation_counts(self):
        for field_name in ("video_clip_duration", "video_count"):
            for value in (0, -1, None):
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaises(ValidationError):
                        VideoParams(video_subject="Coffee", **{field_name: value})

    def test_accepts_positive_generation_counts(self):
        params = VideoParams(
            video_subject="Coffee", video_clip_duration=1, video_count=1
        )

        self.assertEqual(params.video_clip_duration, 1)
        self.assertEqual(params.video_count, 1)


if __name__ == "__main__":
    unittest.main()
