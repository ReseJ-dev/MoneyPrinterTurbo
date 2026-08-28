import json
import unittest
from unittest.mock import Mock, patch

from app.models.schema import MaterialInfo, VideoParams
from app.models.visual_scene import VisualScene
from app.services import material, scene_materials, task, visual_qa, visual_ranking


def _scene() -> VisualScene:
    return VisualScene(
        id=2,
        narration="An owl turns its head on a branch.",
        visual_description="owl turning its head while perched on a branch",
        search_queries=["owl turning head", "perched owl", "owl close up"],
    )


def _candidate(
    asset_id: str,
    *,
    query_priority: int = 0,
    thumbnail_score: float = 0.5,
) -> MaterialInfo:
    scene = _scene()
    return MaterialInfo(
        provider="pexels",
        url=f"https://videos.example/{asset_id}.mp4",
        duration=5,
        source_info={
            "provider": "pexels",
            "asset_id": asset_id,
            "scene_id": scene.id,
            "visual_description": scene.visual_description,
            "search_query": scene.search_queries[query_priority],
            "query_priority": query_priority,
            "provider_result_order": 0,
            "visual_match_score": thumbnail_score,
        },
    )


def _extract(_path, _frame_count):
    return ["frame-1", "frame-2", "frame-3"]


class SequencedScorer:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = 0

    def score_images(self, _description, _images):
        self.calls += 1
        score = next(self.scores)
        if isinstance(score, Exception):
            raise score
        return [score, score, score]


class TestVisualQARetry(unittest.TestCase):
    def test_frame_sampling_uses_three_interior_timestamps(self):
        clip = Mock(duration=10.0)
        clip.get_frame.side_effect = ["raw-1", "raw-2", "raw-3"]
        converted = Mock()
        converted.convert.side_effect = ["image-1", "image-2", "image-3"]

        with (
            patch(
                "moviepy.video.io.VideoFileClip.VideoFileClip",
                return_value=clip,
            ),
            patch("PIL.Image.fromarray", return_value=converted),
        ):
            frames = visual_qa.sample_video_frames("candidate.mp4", 3)

        self.assertEqual(frames, ["image-1", "image-2", "image-3"])
        self.assertEqual(
            [call.args[0] for call in clip.get_frame.call_args_list],
            [1.0, 5.0, 9.0],
        )
        clip.close.assert_called_once()

    def test_frame_score_aggregate_uses_top_two_mean(self):
        self.assertAlmostEqual(
            visual_qa.aggregate_frame_scores([0.1, 0.8, 0.6]),
            0.7,
        )

    def test_first_candidate_passes(self):
        download = Mock(side_effect=lambda item: f"/tmp/{item.url.rsplit('/', 1)[-1]}")

        result = visual_qa.qa_scene_candidates(
            _scene(),
            [_candidate("first"), _candidate("second")],
            scorer=SequencedScorer([0.45]),
            download=download,
            extract_frames=_extract,
            threshold=0.3,
        )

        self.assertEqual(result.material.source_info["asset_id"], "first")
        self.assertEqual(result.report.video_qa_score, 0.45)
        self.assertEqual(result.report.attempts, 1)
        self.assertEqual(result.report.selection_reason, "threshold_passed")
        download.assert_called_once()

    def test_first_candidate_fails_and_second_passes(self):
        scorer = SequencedScorer([0.1, 0.4])

        result = visual_qa.qa_scene_candidates(
            _scene(),
            [_candidate("first"), _candidate("second")],
            scorer=scorer,
            download=lambda item: f"/tmp/{item.source_info['asset_id']}.mp4",
            extract_frames=_extract,
            threshold=0.3,
        )

        self.assertEqual(result.material.source_info["asset_id"], "second")
        self.assertEqual(result.report.attempts, 2)
        self.assertEqual(result.report.rejections[0].reason, "below_threshold")

    def test_all_candidates_fail_accepts_best_available_by_default(self):
        result = visual_qa.qa_scene_candidates(
            _scene(),
            [_candidate("first"), _candidate("second")],
            scorer=SequencedScorer([0.24, 0.12]),
            download=lambda item: f"/tmp/{item.source_info['asset_id']}.mp4",
            extract_frames=_extract,
            threshold=0.3,
        )

        self.assertEqual(result.material.source_info["asset_id"], "first")
        self.assertEqual(result.report.video_qa_score, 0.24)
        self.assertEqual(
            result.report.selection_reason,
            "best_available_below_threshold",
        )
        self.assertEqual(len(result.report.rejections), 2)

    def test_all_candidates_fail_returns_none_in_fail_on_mismatch_mode(self):
        result = visual_qa.qa_scene_candidates(
            _scene(),
            [_candidate("first"), _candidate("second")],
            scorer=SequencedScorer([0.1, 0.2]),
            download=lambda item: f"/tmp/{item.source_info['asset_id']}.mp4",
            extract_frames=_extract,
            threshold=0.3,
            fail_on_mismatch=True,
        )

        self.assertIsNone(result.material)
        self.assertIsNone(result.local_path)
        self.assertEqual(result.report.selection_reason, "no_candidate_passed")

    def test_candidate_retries_are_bounded(self):
        scorer = SequencedScorer([0.1] * 5)
        candidates = [_candidate(str(index)) for index in range(5)]

        result = visual_qa.qa_scene_candidates(
            _scene(),
            candidates,
            scorer=scorer,
            download=lambda item: f"/tmp/{item.source_info['asset_id']}.mp4",
            extract_frames=_extract,
            threshold=0.9,
            max_attempts=2,
        )

        self.assertEqual(result.report.attempts, 2)
        self.assertEqual(scorer.calls, 2)

    def test_fallback_query_is_used_after_primary_candidate_fails(self):
        result = visual_qa.qa_scene_candidates(
            _scene(),
            [
                _candidate("primary", query_priority=0),
                _candidate("fallback", query_priority=1),
            ],
            scorer=SequencedScorer([0.1, 0.5]),
            download=lambda item: f"/tmp/{item.source_info['asset_id']}.mp4",
            extract_frames=_extract,
            threshold=0.3,
        )

        self.assertEqual(result.material.source_info["asset_id"], "fallback")
        self.assertTrue(result.report.fallback_used)

    def test_recent_asset_is_deferred_when_an_alternative_exists(self):
        repeated = _candidate("repeated")
        alternative = _candidate("alternative")
        selected_paths = []

        result = visual_qa.qa_scene_candidates(
            _scene(),
            [repeated, alternative],
            scorer=SequencedScorer([0.5]),
            download=lambda item: (
                selected_paths.append(item.source_info["asset_id"])
                or f"/tmp/{item.source_info['asset_id']}.mp4"
            ),
            extract_frames=_extract,
            threshold=0.3,
            recent_identities=[scene_materials.candidate_identity(repeated)],
        )

        self.assertEqual(selected_paths, ["alternative"])
        self.assertEqual(result.material.source_info["asset_id"], "alternative")

    def test_missing_ml_dependency_degrades_even_when_fail_strict(self):
        unavailable = visual_ranking.VisualRankingUnavailable("missing optional ML")

        result = visual_qa.qa_scene_candidates(
            _scene(),
            [_candidate("first")],
            scorer=SequencedScorer([unavailable]),
            download=lambda _item: "/tmp/first.mp4",
            extract_frames=_extract,
            fail_on_mismatch=True,
        )

        self.assertEqual(result.material.source_info["asset_id"], "first")
        self.assertIsNone(result.report.video_qa_score)
        self.assertEqual(
            result.report.selection_reason,
            "visual_ai_unavailable_degraded",
        )

    def test_report_serializes_to_plain_json(self):
        result = visual_qa.qa_scene_candidates(
            _scene(),
            [_candidate("first")],
            scorer=SequencedScorer([0.5]),
            download=lambda _item: "/tmp/first.mp4",
            extract_frames=_extract,
        )

        payload = json.loads(json.dumps(result.report.to_dict()))

        self.assertEqual(payload["scene_id"], 2)
        self.assertEqual(payload["selected_asset_id"], "first")
        self.assertEqual(payload["thumbnail_score"], 0.5)
        self.assertEqual(payload["video_qa_score"], 0.5)


class TestVisualQAIntegration(unittest.TestCase):
    def test_better_mode_disables_legacy_visual_qa_config(self):
        pool = scene_materials.SceneCandidatePool(
            scene=_scene(),
            candidates=(_candidate("first"),),
            attempted_queries=tuple(_scene().search_queries),
        )
        with (
            patch.object(
                material,
                "search_video_candidates_by_scene",
                return_value=[pool],
            ),
            patch.object(material.config, "app", {"visual_qa_enabled": True}),
            patch.object(
                material,
                "download_selected_scene_videos",
                return_value=["first.mp4"],
            ),
            patch.object(material.visual_qa, "qa_scene_candidates") as qa,
        ):
            result = material.download_videos_for_scenes(
                "task",
                [_scene()],
                material_matching_mode="better",
            )

        self.assertEqual(result, ["first.mp4"])
        qa.assert_not_called()

    def test_disabled_qa_uses_existing_selection_and_download_path(self):
        pool = scene_materials.SceneCandidatePool(
            scene=_scene(),
            candidates=(_candidate("first"),),
            attempted_queries=tuple(_scene().search_queries),
        )
        with (
            patch.object(
                material,
                "search_video_candidates_by_scene",
                return_value=[pool],
            ),
            patch.object(material.config, "app", {"visual_qa_enabled": False}),
            patch.object(
                material,
                "select_video_candidates_by_scene",
                return_value=[
                    scene_materials.SceneMaterialSelection(
                        scene=_scene(),
                        material=_candidate("first"),
                    )
                ],
            ) as select,
            patch.object(
                material,
                "download_selected_scene_videos",
                return_value=["first.mp4"],
            ) as download,
            patch.object(material.visual_qa, "qa_scene_candidates") as qa,
        ):
            result = material.download_videos_for_scenes("task", [_scene()])

        self.assertEqual(result, ["first.mp4"])
        select.assert_called_once()
        download.assert_called_once()
        qa.assert_not_called()

    def test_visual_matching_report_is_persisted_as_task_artifact(self):
        report = visual_qa.VisualQAReport(
            scene_id=2,
            visual_description="owl on branch",
            queries=("owl", "perched owl", "owl close up"),
            candidates_considered=7,
            selected_asset_id="123",
            thumbnail_score=0.81,
            video_qa_score=0.76,
            attempts=2,
            fallback_used=False,
            selection_reason="threshold_passed",
            rejections=(),
        )

        with patch.object(
            material.task_artifacts,
            "patch_script_data",
            return_value=True,
        ) as persist:
            material._persist_visual_matching_report("task", [report])

        payload = persist.call_args.kwargs["visual_matching_report"]
        self.assertEqual(payload[0]["selected_asset_id"], "123")
        self.assertEqual(payload[0]["video_qa_score"], 0.76)

    def test_explicit_fail_on_mismatch_does_not_use_legacy_fallback(self):
        params = VideoParams(
            video_subject="owl",
            visual_scene_planning=True,
        )
        mismatch = visual_qa.StrictVisualQAMismatch("scene 2 failed")

        with (
            patch.object(
                material,
                "download_videos_for_scenes",
                side_effect=mismatch,
            ),
            patch.object(material, "download_videos") as legacy,
            patch.object(task, "_mark_task_failed", return_value=None) as fail,
        ):
            result = task.get_video_materials(
                "task",
                params,
                ["owl"],
                5,
                visual_scene_plan=[_scene()],
            )

        self.assertIsNone(result)
        legacy.assert_not_called()
        fail.assert_called_once_with("task", "materials", "scene 2 failed")


if __name__ == "__main__":
    unittest.main()
