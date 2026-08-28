import sys
import unittest
from unittest.mock import patch

from app.models.schema import MaterialInfo
from app.models.visual_scene import VisualScene
from app.services import scene_materials, visual_ranking


def _scene() -> VisualScene:
    return VisualScene(
        id=1,
        narration="An owl turns its head while perched on a branch.",
        visual_description="owl turning its head while perched on a branch",
        search_queries=["owl turning head", "perched owl", "owl close up"],
    )


def _candidate(
    asset_id: str,
    *,
    preview: bool = True,
    query_priority: int = 0,
    result_order: int = 0,
    width: int = 1080,
    height: int = 1920,
    duration: int = 5,
) -> MaterialInfo:
    source_info = {
        "provider": "pexels",
        "asset_id": asset_id,
        "query_priority": query_priority,
        "provider_result_order": result_order,
        "rendition": {"width": width, "height": height},
    }
    if preview:
        source_info["preview_url"] = f"https://images.example/{asset_id}.jpg"
    return MaterialInfo(
        provider="pexels",
        url=f"https://videos.example/{asset_id}.mp4",
        duration=duration,
        source_info=source_info,
    )


class TestLocalVisualRanker(unittest.TestCase):
    def setUp(self):
        visual_ranking._MODEL_CACHE.clear()
        visual_ranking._get_local_ranker.cache_clear()

    def tearDown(self):
        visual_ranking._MODEL_CACHE.clear()
        visual_ranking._get_local_ranker.cache_clear()

    def test_semantic_scores_control_ranking_order_and_metadata(self):
        candidates = [
            _candidate("owl-still"),
            _candidate("eagle"),
            _candidate("owl-turning"),
            _candidate("forest"),
        ]
        ranker = visual_ranking.LocalVisualRanker("test-model", "test-weights")

        with patch.object(
            ranker,
            "_score_thumbnails",
            return_value=[0.55, 0.12, 0.91, 0.20],
        ):
            ranked = ranker.rank(_scene(), candidates)

        self.assertEqual(
            [item.source_info["asset_id"] for item in ranked],
            ["owl-turning", "owl-still", "forest", "eagle"],
        )
        self.assertEqual(ranked[0].source_info["visual_match_score"], 0.91)
        self.assertIsInstance(ranked[0].source_info["visual_match_score"], float)
        self.assertNotIn("visual_match_score", candidates[2].source_info)

    def test_missing_thumbnails_are_preserved_after_scored_candidates(self):
        candidates = [
            _candidate("missing", preview=False),
            _candidate("scored"),
        ]
        ranker = visual_ranking.LocalVisualRanker("test-model", "test-weights")

        with patch.object(
            ranker,
            "_score_thumbnails",
            return_value=[-0.2],
        ) as score:
            ranked = ranker.rank(_scene(), candidates)

        score.assert_called_once_with(
            _scene().visual_description,
            ["https://images.example/scored.jpg"],
        )
        self.assertEqual(
            [item.source_info["asset_id"] for item in ranked],
            ["scored", "missing"],
        )
        self.assertNotIn("visual_match_score", ranked[1].source_info)

    def test_no_thumbnails_skips_model_inference(self):
        candidates = [
            _candidate("one", preview=False),
            _candidate("two", preview=False),
        ]
        ranker = visual_ranking.LocalVisualRanker("test-model", "test-weights")

        with patch.object(ranker, "_score_thumbnails") as score:
            ranked = ranker.rank(_scene(), candidates)

        score.assert_not_called()
        self.assertEqual(ranked, candidates)

    def test_equal_scores_use_query_then_provider_order_ties(self):
        candidates = [
            _candidate("later-query", query_priority=1, result_order=0),
            _candidate("second-result", query_priority=0, result_order=1),
            _candidate("first-result", query_priority=0, result_order=0),
        ]
        ranker = visual_ranking.LocalVisualRanker("test-model", "test-weights")

        with patch.object(
            ranker,
            "_score_thumbnails",
            return_value=[0.5, 0.5, 0.5],
        ):
            ranked = ranker.rank(_scene(), candidates)

        self.assertEqual(
            [item.source_info["asset_id"] for item in ranked],
            ["first-result", "second-result", "later-query"],
        )

    def test_model_bundle_is_loaded_once_per_process_configuration(self):
        bundle = object()
        with patch.object(
            visual_ranking,
            "_load_model_bundle",
            return_value=bundle,
        ) as load:
            first = visual_ranking._get_model_bundle("model", "weights")
            second = visual_ranking._get_model_bundle("model", "weights")

        self.assertIs(first, bundle)
        self.assertIs(second, bundle)
        load.assert_called_once_with("model", "weights")

    def test_missing_optional_dependency_has_clear_error(self):
        with patch.dict(sys.modules, {"open_clip": None}):
            with self.assertRaisesRegex(
                visual_ranking.VisualRankingUnavailable,
                "visual-ranking",
            ):
                visual_ranking._load_model_bundle("model", "weights")


class TestVisualRankingConfigurationAndFallback(unittest.TestCase):
    def tearDown(self):
        visual_ranking._get_local_ranker.cache_clear()

    def test_disabled_ranking_does_not_create_local_ranker(self):
        with patch.object(visual_ranking, "_get_local_ranker") as get_ranker:
            ranker = visual_ranking.configured_ranker({"visual_ranking_enabled": False})

        self.assertIsNone(ranker)
        get_ranker.assert_not_called()

    def test_enabled_local_configuration_is_cached(self):
        config = {
            "visual_ranking_enabled": True,
            "visual_ranking_provider": "local",
            "visual_ranking_model": "model",
            "visual_ranking_pretrained": "weights",
        }

        first = visual_ranking.configured_ranker(config)
        second = visual_ranking.configured_ranker(config)

        self.assertIs(first, second)

    def test_ranker_failure_returns_original_deterministic_pools(self):
        scene = _scene()
        pool = scene_materials.SceneCandidatePool(
            scene=scene,
            candidates=(_candidate("first"), _candidate("second")),
            attempted_queries=tuple(scene.search_queries),
        )

        class FailingRanker:
            def rank(self, scene, candidates):
                raise visual_ranking.VisualRankingUnavailable("missing model")

        with patch.object(visual_ranking.logger, "warning") as warning:
            ranked = visual_ranking.rank_candidate_pools([pool], FailingRanker())

        self.assertEqual(ranked, [pool])
        self.assertTrue(warning.called)


if __name__ == "__main__":
    unittest.main()
