from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from app.models.schema import MaterialInfo
from app.models.visual_scene import VisualScene


SceneSearch = Callable[[str], Sequence[MaterialInfo]]


@dataclass(frozen=True)
class SceneCandidatePool:
    scene: VisualScene
    candidates: tuple[MaterialInfo, ...]
    attempted_queries: tuple[str, ...]


@dataclass(frozen=True)
class SceneMaterialSelection:
    scene: VisualScene
    material: MaterialInfo


def _normalized_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
    except (AttributeError, ValueError):
        return str(value).strip()
    if not parsed.scheme or not parsed.netloc:
        return str(value).strip()
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            "",
            "",
        )
    )


def candidate_identity(item: MaterialInfo) -> str:
    """Return the strongest provider-scoped stable identity available."""
    source = item.source_info if isinstance(item.source_info, dict) else {}
    provider = str(item.provider or source.get("provider") or "unknown").casefold()
    asset_id = source.get("asset_id")
    if asset_id not in (None, ""):
        return f"{provider}:asset:{asset_id}"

    source_page = source.get("source_page")
    if isinstance(source_page, str) and source_page.strip():
        return f"{provider}:page:{_normalized_url(source_page)}"
    return f"{provider}:url:{_normalized_url(item.url)}"


def _associate_candidate(
    item: MaterialInfo,
    *,
    scene: VisualScene,
    query: str,
) -> MaterialInfo:
    source = dict(item.source_info) if isinstance(item.source_info, dict) else {}
    source.update(
        {
            "provider": str(item.provider or source.get("provider") or ""),
            "search_term": query,
            "scene_id": scene.id,
            "visual_description": scene.visual_description,
            "search_query": query,
        }
    )
    return MaterialInfo(
        provider=item.provider,
        url=item.url,
        duration=item.duration,
        source_info=source,
    )


def search_candidates_by_scene(
    scenes: Sequence[VisualScene],
    search: SceneSearch,
) -> list[SceneCandidatePool]:
    """Search all scene queries in priority order without downloading assets."""
    pools: list[SceneCandidatePool] = []
    for scene in scenes:
        candidates: list[MaterialInfo] = []
        seen: set[str] = set()
        attempted_queries: list[str] = []
        for query in scene.search_queries:
            attempted_queries.append(query)
            try:
                results = search(query)
            except Exception as exc:
                logger.warning(
                    "visual scene query failed, continue with remaining queries: "
                    f"scene_id={scene.id}, query={query!r}, "
                    f"error={type(exc).__name__}, detail={exc}"
                )
                continue

            for item in results:
                identity = candidate_identity(item)
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append(_associate_candidate(item, scene=scene, query=query))

        pools.append(
            SceneCandidatePool(
                scene=scene,
                candidates=tuple(candidates),
                attempted_queries=tuple(attempted_queries),
            )
        )
    return pools


def select_candidates_by_scene(
    pools: Sequence[SceneCandidatePool],
) -> list[SceneMaterialSelection]:
    """Select at most one deterministic candidate per scene in scene order."""
    selections: list[SceneMaterialSelection] = []
    selected_identities: set[str] = set()
    for pool in pools:
        selected = next(
            (
                item
                for item in pool.candidates
                if candidate_identity(item) not in selected_identities
            ),
            None,
        )
        if selected is None and pool.candidates:
            selected = pool.candidates[0]
        if selected is None:
            continue
        selected_identities.add(candidate_identity(selected))
        selections.append(SceneMaterialSelection(scene=pool.scene, material=selected))
    return selections
