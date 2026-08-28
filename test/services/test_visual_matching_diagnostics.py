import json
from unittest.mock import patch

from app.models.schema import MaterialInfo
from app.models.visual_scene import VisualScene
from app.services import (
    material,
    scene_materials,
    visual_matching_diagnostics,
    visual_qa,
)


def _scene(scene_id: int, narration: str = "An owl turns its head.") -> VisualScene:
    return VisualScene(
        id=scene_id,
        narration=narration,
        visual_description="owl turning its head on a branch",
        search_queries=["owl turning head", "perched owl", "owl close up"],
    )


def _candidate(asset_id: str, query_priority: int, score: float) -> MaterialInfo:
    scene = _scene(1)
    return MaterialInfo(
        provider="pexels",
        url=f"https://signed.example/{asset_id}.mp4?api_key=private",
        duration=5,
        source_info={
            "provider": "pexels",
            "asset_id": asset_id,
            "scene_id": scene.id,
            "visual_description": scene.visual_description,
            "search_query": scene.search_queries[query_priority],
            "query_priority": query_priority,
            "provider_result_order": 0,
            "visual_match_score": score,
            "source_page": (
                f"https://www.pexels.com/video/{asset_id}?signature=private"
            ),
            "preview_url": "https://preview.example/image.jpg?token=private",
        },
    )


def test_report_contains_candidate_decisions_metrics_and_safe_source_pages():
    first = _candidate("100", 0, 0.81)
    selected = _candidate("200", 1, 0.83)
    empty_scene = _scene(2, "The forest is quiet.")
    pools = [
        scene_materials.SceneCandidatePool(
            scene=_scene(1),
            candidates=(first, selected),
            attempted_queries=tuple(_scene(1).search_queries),
        ),
        scene_materials.SceneCandidatePool(
            scene=empty_scene,
            candidates=(),
            attempted_queries=tuple(empty_scene.search_queries),
        ),
    ]
    qa_report = visual_qa.VisualQAReport(
        scene_id=1,
        visual_description=_scene(1).visual_description,
        queries=tuple(_scene(1).search_queries),
        candidates_considered=2,
        selected_asset_id="200",
        thumbnail_score=0.83,
        video_qa_score=0.79,
        attempts=2,
        fallback_used=True,
        selection_reason="threshold_passed",
        rejections=(
            visual_qa.VisualQARejection(
                asset_id="100",
                query="owl turning head",
                score=0.12,
                reason="below_threshold",
            ),
        ),
    )

    report = visual_matching_diagnostics.build_visual_matching_report(
        "strict",
        pools,
        {1: selected},
        {1: qa_report},
    )
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["mode"] == "strict"
    assert payload["scenes"][0]["narration"] == "An owl turns its head."
    assert payload["scenes"][0]["selected"]["asset_id"] == "200"
    assert payload["scenes"][0]["selected"]["video_qa_score"] == 0.79
    assert payload["scenes"][0]["candidates"][0]["status"] == "rejected"
    assert payload["scenes"][0]["candidates"][0]["rejection_reason"] == (
        "below_threshold"
    )
    assert payload["scenes"][0]["selected"]["source_page"] == (
        "https://www.pexels.com/video/200"
    )
    assert payload["metrics"] == {
        "scene_coverage_percentage": 50.0,
        "average_semantic_score": 0.83,
        "retries": 1,
        "fallback_scenes": 1,
        "duplicate_asset_count": 0,
        "scenes_without_candidates": 1,
    }
    serialized = json.dumps(payload)
    assert "api_key" not in serialized
    assert "signature" not in serialized
    assert "private" not in serialized
    assert "preview.example" not in serialized
    assert "signed.example" not in serialized


def test_report_counts_duplicate_selected_assets():
    shared = _candidate("same", 0, 0.5)
    second_scene = _scene(2)
    second_shared = MaterialInfo(
        provider=shared.provider,
        url=shared.url,
        duration=shared.duration,
        source_info={**shared.source_info, "scene_id": 2},
    )
    pools = [
        scene_materials.SceneCandidatePool(_scene(1), (shared,), ("owl",)),
        scene_materials.SceneCandidatePool(second_scene, (second_shared,), ("owl",)),
    ]

    report = visual_matching_diagnostics.build_visual_matching_report(
        "better",
        pools,
        {1: shared, 2: second_shared},
    )

    assert report.metrics.scene_coverage_percentage == 100.0
    assert report.metrics.duplicate_asset_count == 1


def test_credential_bearing_source_page_is_omitted():
    assert (
        visual_matching_diagnostics.safe_public_page(
            "https://api-key:secret@example.com/video/1"
        )
        is None
    )


def test_unified_report_is_persisted_as_task_artifact():
    report = visual_matching_diagnostics.build_visual_matching_report(
        "better",
        [
            scene_materials.SceneCandidatePool(
                _scene(1),
                (),
                tuple(_scene(1).search_queries),
            )
        ],
        {},
    )

    with patch.object(
        material.task_artifacts,
        "patch_script_data",
        return_value=True,
    ) as persist:
        material._persist_visual_matching_report("task", report)

    payload = persist.call_args.kwargs["visual_matching_report"]
    assert payload["mode"] == "better"
    assert payload["scenes"][0]["scene_id"] == 1
    assert payload["metrics"]["scene_coverage_percentage"] == 0.0
