import unittest
from unittest.mock import patch

from app.models.schema import MaterialInfo, VideoParams
from app.models.visual_scene import VisualScene
from app.services import material
from app.services import scene_materials
from app.services import task as task_service


def _scene(scene_id: int) -> VisualScene:
    return VisualScene(
        id=scene_id,
        narration=f"Narration {scene_id}",
        visual_description=f"Visible scene {scene_id}",
        search_queries=[
            f"query {scene_id}a",
            f"query {scene_id}b",
            f"query {scene_id}c",
        ],
    )


def _material(asset_id: str) -> MaterialInfo:
    return MaterialInfo(
        provider="pexels",
        url=f"https://videos.example/{asset_id}.mp4",
        duration=5,
        source_info={
            "provider": "pexels",
            "asset_id": asset_id,
            "source_page": f"https://www.pexels.com/video/{asset_id}/",
            "creator": {"name": "Creator"},
            "rendition": {"id": "hd", "width": 1080, "height": 1920},
        },
    )


class TestSceneMaterialRetrieval(unittest.TestCase):
    def test_search_builds_enriched_pools_without_downloading(self):
        scene = _scene(1)

        with (
            patch.object(
                material,
                "_search_videos_with_cache",
                side_effect=lambda **kwargs: [_material(kwargs["search_term"])],
            ) as search,
            patch.object(material, "save_video") as save,
        ):
            pools = material.search_video_candidates_by_scene([scene])

        self.assertEqual(search.call_count, 3)
        save.assert_not_called()
        candidate = pools[0].candidates[0]
        self.assertEqual(candidate.source_info["scene_id"], 1)
        self.assertEqual(candidate.source_info["visual_description"], "Visible scene 1")
        self.assertEqual(candidate.source_info["search_query"], "query 1a")
        self.assertEqual(candidate.source_info["creator"], {"name": "Creator"})
        self.assertEqual(candidate.source_info["rendition"]["width"], 1080)

    def test_download_only_receives_selected_assets_in_scene_order(self):
        scenes = [_scene(1), _scene(2)]
        pools = scene_materials.search_candidates_by_scene(
            scenes,
            lambda query: [_material(query.replace(" ", "-"))],
        )
        selections = material.select_video_candidates_by_scene(pools)

        with (
            patch.object(
                material,
                "save_video",
                side_effect=lambda video_url, save_dir="": (
                    f"/tmp/{video_url.rsplit('/', 1)[-1]}"
                ),
            ) as save,
            patch.object(material, "_persist_material_sources") as persist,
        ):
            paths = material.download_selected_scene_videos("task-scenes", selections)

        self.assertEqual(save.call_count, 2)
        self.assertEqual(paths, ["/tmp/query-1a.mp4", "/tmp/query-2a.mp4"])
        records = persist.call_args.args[1]
        self.assertEqual([record["scene_id"] for record in records], [1, 2])
        self.assertEqual(
            [record["search_query"] for record in records],
            ["query 1a", "query 2a"],
        )

    def test_material_source_record_keeps_numeric_visual_score(self):
        item = _material("ranked")
        item.source_info["visual_match_score"] = 0.82

        record = material._material_source_record(item, "/tmp/ranked.mp4")

        self.assertEqual(record["visual_match_score"], 0.82)

    def test_scene_selection_uses_configured_visual_ranker(self):
        scene = _scene(1)
        first = _material("first")
        second = _material("second")
        pool = scene_materials.SceneCandidatePool(
            scene=scene,
            candidates=(first, second),
            attempted_queries=tuple(scene.search_queries),
        )

        class ReverseRanker:
            def rank(self, ranked_scene, candidates):
                self.scene = ranked_scene
                return list(reversed(candidates))

        ranker = ReverseRanker()
        with patch.object(
            material.visual_ranking,
            "configured_ranker",
            return_value=ranker,
        ):
            selections = material.select_video_candidates_by_scene([pool])

        self.assertEqual(ranker.scene, scene)
        self.assertEqual(selections[0].material.url, second.url)


class TestSceneMaterialFallback(unittest.TestCase):
    def test_complete_scene_failure_falls_back_to_legacy_ordered_download(self):
        params = VideoParams(
            video_subject="black beach",
            visual_scene_planning=True,
        )
        scenes = [_scene(1)]

        with (
            patch.object(material, "download_videos_for_scenes", return_value=[]),
            patch.object(
                material,
                "download_videos",
                return_value=["legacy.mp4"],
            ) as legacy,
            patch.object(task_service.logger, "warning") as warning,
        ):
            result = task_service.get_video_materials(
                "scene-fallback",
                params,
                ["query 1a", "query 1b", "query 1c"],
                5,
                visual_scene_plan=scenes,
            )

        self.assertEqual(result, ["legacy.mp4"])
        self.assertTrue(legacy.call_args.kwargs["match_script_order"])
        self.assertEqual(
            legacy.call_args.kwargs["video_concat_mode"].value,
            "sequential",
        )
        self.assertTrue(
            any("fall back" in str(call) for call in warning.call_args_list)
        )

    def test_legacy_mode_does_not_call_scene_retrieval(self):
        params = VideoParams(video_subject="black beach")

        with (
            patch.object(material, "download_videos_for_scenes") as scene_download,
            patch.object(
                material,
                "download_videos",
                return_value=["legacy.mp4"],
            ) as legacy,
        ):
            result = task_service.get_video_materials(
                "legacy",
                params,
                ["black beach"],
                5,
            )

        self.assertEqual(result, ["legacy.mp4"])
        scene_download.assert_not_called()
        self.assertFalse(legacy.call_args.kwargs["match_script_order"])


if __name__ == "__main__":
    unittest.main()
