import unittest

from app.models.schema import MaterialInfo
from app.models.visual_scene import VisualScene
from app.services import scene_materials


def _scene(scene_id: int, *queries: str) -> VisualScene:
    return VisualScene(
        id=scene_id,
        narration=f"Narration {scene_id}",
        visual_description=f"Visible scene {scene_id}",
        search_queries=list(queries),
    )


def _material(asset_id: str, *, url: str | None = None) -> MaterialInfo:
    return MaterialInfo(
        provider="pexels",
        url=url or f"https://videos.example/{asset_id}.mp4",
        duration=5,
        source_info={
            "provider": "pexels",
            "asset_id": asset_id,
            "creator": {"name": "Creator"},
        },
    )


class TestSceneCandidateSearch(unittest.TestCase):
    def test_executes_three_queries_per_scene_in_priority_order(self):
        scene = _scene(1, "black sand beach", "volcanic beach", "dark ocean")
        calls = []

        pools = scene_materials.search_candidates_by_scene(
            [scene],
            lambda query: calls.append(query) or [_material(query.replace(" ", "-"))],
        )

        self.assertEqual(calls, scene.search_queries)
        self.assertEqual(pools[0].attempted_queries, tuple(scene.search_queries))
        self.assertEqual(len(pools[0].candidates), 3)

    def test_empty_or_failed_query_falls_back_to_later_queries(self):
        scene = _scene(1, "empty query", "failed query", "working query")

        def search(query):
            if query == "empty query":
                return []
            if query == "failed query":
                raise RuntimeError("provider timeout")
            return [_material("working")]

        pools = scene_materials.search_candidates_by_scene([scene], search)

        self.assertEqual(pools[0].attempted_queries, tuple(scene.search_queries))
        self.assertEqual(
            [item.source_info["asset_id"] for item in pools[0].candidates],
            ["working"],
        )
        self.assertEqual(
            pools[0].candidates[0].source_info["search_query"],
            "working query",
        )

    def test_deduplicates_pexels_candidates_by_asset_id(self):
        scene = _scene(1, "first query", "second query", "third query")
        results = {
            "first query": [_material("42", url="https://cdn.example/one.mp4")],
            "second query": [_material("42", url="https://cdn.example/two.mp4")],
            "third query": [_material("43")],
        }

        pools = scene_materials.search_candidates_by_scene(
            [scene], lambda query: results[query]
        )

        self.assertEqual(
            [item.source_info["asset_id"] for item in pools[0].candidates],
            ["42", "43"],
        )
        self.assertEqual(
            pools[0].candidates[0].source_info["search_query"],
            "first query",
        )

    def test_url_fallback_identity_ignores_query_parameters(self):
        first = MaterialInfo(
            provider="coverr",
            url="https://cdn.example/clip.mp4?token=one",
            duration=5,
        )
        second = MaterialInfo(
            provider="coverr",
            url="https://cdn.example/clip.mp4?token=two",
            duration=5,
        )

        self.assertEqual(
            scene_materials.candidate_identity(first),
            scene_materials.candidate_identity(second),
        )


class TestSceneCandidateSelection(unittest.TestCase):
    def test_selection_is_chronological_and_balanced_across_scenes(self):
        scenes = [
            _scene(1, "a", "b", "c"),
            _scene(2, "d", "e", "f"),
            _scene(3, "g", "h", "i"),
        ]
        pools = [
            scene_materials.SceneCandidatePool(
                scene=scenes[0],
                candidates=(_material("1a"), _material("1b"), _material("1c")),
                attempted_queries=tuple(scenes[0].search_queries),
            ),
            scene_materials.SceneCandidatePool(
                scene=scenes[1],
                candidates=(_material("2a"),),
                attempted_queries=tuple(scenes[1].search_queries),
            ),
            scene_materials.SceneCandidatePool(
                scene=scenes[2],
                candidates=(_material("3a"),),
                attempted_queries=tuple(scenes[2].search_queries),
            ),
        ]

        selections = scene_materials.select_candidates_by_scene(pools)

        self.assertEqual([selection.scene.id for selection in selections], [1, 2, 3])
        self.assertEqual(
            [selection.material.source_info["asset_id"] for selection in selections],
            ["1a", "2a", "3a"],
        )

    def test_missing_scene_results_do_not_block_later_scenes(self):
        scenes = [_scene(1, "a", "b", "c"), _scene(2, "d", "e", "f")]
        pools = [
            scene_materials.SceneCandidatePool(
                scene=scenes[0],
                candidates=(),
                attempted_queries=tuple(scenes[0].search_queries),
            ),
            scene_materials.SceneCandidatePool(
                scene=scenes[1],
                candidates=(_material("2a"),),
                attempted_queries=tuple(scenes[1].search_queries),
            ),
        ]

        selections = scene_materials.select_candidates_by_scene(pools)

        self.assertEqual([selection.scene.id for selection in selections], [2])


if __name__ == "__main__":
    unittest.main()
