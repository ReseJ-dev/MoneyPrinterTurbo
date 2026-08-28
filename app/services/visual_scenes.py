from __future__ import annotations

import json
import re
from typing import Any

from app.models.visual_scene import VisualScene


DEFAULT_QUERIES_PER_SCENE = 3
DEFAULT_CLIP_DURATION_SECONDS = 4.0


def build_visual_scene_prompt(
    *,
    video_subject: str,
    video_script: str,
    target_scene_count: int | None = None,
    clip_duration: float = DEFAULT_CLIP_DURATION_SECONDS,
    queries_per_scene: int = DEFAULT_QUERIES_PER_SCENE,
) -> str:
    """Build stock-footage-specific instructions for chronological scene planning."""
    if target_scene_count is not None and target_scene_count < 1:
        raise ValueError("target_scene_count must be positive")
    if clip_duration <= 0:
        raise ValueError("clip_duration must be positive")
    if queries_per_scene < 1:
        raise ValueError("queries_per_scene must be positive")

    scene_count_guidance = (
        f"Aim for approximately {target_scene_count} scenes."
        if target_scene_count is not None
        else "Short scripts commonly need 4-7 scenes, but use fewer when the same "
        "footage can cover consecutive phrases."
    )
    query_example = json.dumps(
        [f"visible query {index}" for index in range(1, queries_per_scene + 1)]
    )

    return f"""
# Role: Stock Footage Visual Scene Planner

## Goal
Convert the narration into a chronological visual plan for stock footage search.

## Rules
1. Keep scenes in narration order and number ids consecutively from 1.
2. Plan a new visual roughly every 3-5 seconds (about {clip_duration:g} seconds), but do not split every sentence blindly.
3. Do not create a new scene when consecutive narration can use the same footage.
4. Every scene must represent contiguous narration text from the script.
5. visual_description must be a concise English description of something visibly filmable.
6. Generate exactly {queries_per_scene} distinct English search_queries per scene.
7. Prefer simple 1-4 word queries made from visible nouns and actions.
8. Avoid abstract ideas, complete sentences, explanations, opinions, and duplicate queries.
9. Optimize for geography, nature, animals, large machines, transport, and engineering stock footage.
10. Do not invent talking heads, medical animation, maps, historical footage, or news footage.
11. Return only a JSON array. Do not include markdown or commentary.

{scene_count_guidance}

## Output schema
[
  {{
    "id": 1,
    "narration": "Exact narration represented by this scene.",
    "visual_description": "concise visible English description",
    "search_queries": {query_example}
  }}
]

## Query examples
Good: "owl flying", "owl close up", "forest owl"
Bad: "owl amazing ability", "how owls rotate their heads", "interesting nature fact"

## Video subject
{video_subject}

## Full narration
{video_script}
""".strip()


def _strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _json_candidates(text: str):
    """Yield complete JSON values, including values embedded in surrounding text."""
    stripped = _strip_code_fence(text)
    if stripped:
        try:
            yield json.loads(stripped)
            return
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        yield value


def _scene_items(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("visual_scenes"), list):
        return value["visual_scenes"]
    return None


def parse_visual_scenes(
    response: str,
    *,
    queries_per_scene: int = DEFAULT_QUERIES_PER_SCENE,
) -> list[VisualScene]:
    """Recover and validate a typed visual scene plan from an LLM response."""
    if queries_per_scene < 1:
        raise ValueError("queries_per_scene must be positive")
    if not (response or "").strip():
        raise ValueError("visual scene response is empty")

    validation_errors: list[Exception] = []
    for candidate in _json_candidates(response):
        items = _scene_items(candidate)
        if not items:
            continue
        try:
            scenes = [VisualScene.model_validate(item) for item in items]
            expected_ids = list(range(1, len(scenes) + 1))
            if [scene.id for scene in scenes] != expected_ids:
                raise ValueError(
                    "visual scene ids must be consecutive and chronological"
                )
            if any(len(scene.search_queries) != queries_per_scene for scene in scenes):
                raise ValueError(
                    f"each visual scene must have exactly {queries_per_scene} queries"
                )
            return scenes
        except (TypeError, ValueError) as exc:
            validation_errors.append(exc)

    if validation_errors:
        raise ValueError(f"invalid visual scene plan: {validation_errors[-1]}")
    raise ValueError("response does not contain a visual scene JSON array")


def serialize_visual_scenes(scenes: list[VisualScene]) -> list[dict[str, Any]]:
    """Return JSON-ready scene data for task artifacts."""
    return [scene.model_dump(mode="json") for scene in scenes]
