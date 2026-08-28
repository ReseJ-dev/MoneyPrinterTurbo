from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from app.models.schema import MaterialInfo
from app.services import scene_materials, visual_qa


@dataclass(frozen=True)
class VisualCandidateDiagnostic:
    provider: str
    asset_id: str
    query: str
    status: str
    source_page: str | None = None
    thumbnail_score: float | None = None
    video_qa_score: float | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class VisualSceneDiagnostic:
    scene_id: int
    narration: str
    visual_description: str
    queries: tuple[str, ...]
    candidates_found: int
    selected: VisualCandidateDiagnostic | None
    candidates: tuple[VisualCandidateDiagnostic, ...]
    retries: int = 0
    fallback_used: bool = False


@dataclass(frozen=True)
class VisualMatchingMetrics:
    scene_coverage_percentage: float
    average_semantic_score: float | None
    retries: int
    fallback_scenes: int
    duplicate_asset_count: int
    scenes_without_candidates: int


@dataclass(frozen=True)
class VisualMatchingReport:
    mode: str
    metrics: VisualMatchingMetrics
    scenes: tuple[VisualSceneDiagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_public_page(value: Any) -> str | None:
    """Return a credential-free public page URL, never a signed download URL."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _source(item: MaterialInfo) -> Mapping[str, Any]:
    return item.source_info if isinstance(item.source_info, dict) else {}


def _short_text(value: Any, *, maximum: int = 300) -> str:
    return str(value or "").strip()[:maximum]


def _asset_id(item: MaterialInfo) -> str:
    value = _source(item).get("asset_id")
    if value not in (None, ""):
        text = _short_text(value, maximum=200)
        if "://" not in text:
            return text
    return _short_text(scene_materials.candidate_identity(item), maximum=300)


def _score(item: MaterialInfo, key: str) -> float | None:
    value = _source(item).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return None


def _candidate(
    item: MaterialInfo,
    *,
    status: str,
    video_qa_score: float | None = None,
    rejection_reason: str | None = None,
) -> VisualCandidateDiagnostic:
    source = _source(item)
    return VisualCandidateDiagnostic(
        provider=_short_text(item.provider or source.get("provider"), maximum=50),
        asset_id=_asset_id(item),
        query=_short_text(source.get("search_query"), maximum=120),
        status=status,
        source_page=safe_public_page(source.get("source_page")),
        thumbnail_score=_score(item, "visual_match_score"),
        video_qa_score=(
            round(float(video_qa_score), 6) if video_qa_score is not None else None
        ),
        rejection_reason=(
            _short_text(rejection_reason, maximum=120) if rejection_reason else None
        ),
    )


def build_visual_matching_report(
    mode: str,
    pools: Sequence[scene_materials.SceneCandidatePool],
    selected_by_scene: Mapping[int, MaterialInfo],
    qa_reports: Mapping[int, visual_qa.VisualQAReport] | None = None,
    *,
    legacy_fallback_used: bool = False,
) -> VisualMatchingReport:
    """Build a compact, JSON-safe Better/Strict matching diagnostic report."""
    qa_reports = qa_reports or {}
    scenes: list[VisualSceneDiagnostic] = []

    for pool in pools:
        selected = selected_by_scene.get(pool.scene.id)
        selected_identity = (
            scene_materials.candidate_identity(selected) if selected else None
        )
        qa_report = qa_reports.get(pool.scene.id)
        rejection_by_asset = (
            {rejection.asset_id: rejection for rejection in qa_report.rejections}
            if qa_report
            else {}
        )
        candidates: list[VisualCandidateDiagnostic] = []

        for item in pool.candidates:
            identity = scene_materials.candidate_identity(item)
            asset_id = _asset_id(item)
            rejection = rejection_by_asset.get(asset_id)
            is_selected = selected_identity == identity
            status = (
                "selected" if is_selected else "rejected" if rejection else "candidate"
            )
            candidates.append(
                _candidate(
                    item,
                    status=status,
                    video_qa_score=(
                        qa_report.video_qa_score
                        if is_selected and qa_report
                        else rejection.score
                        if rejection
                        else None
                    ),
                    rejection_reason=rejection.reason if rejection else None,
                )
            )

        selected_record = next(
            (candidate for candidate in candidates if candidate.status == "selected"),
            _candidate(
                selected,
                status="selected",
                video_qa_score=qa_report.video_qa_score if qa_report else None,
            )
            if selected
            else None,
        )
        retries = max((qa_report.attempts if qa_report else 1) - 1, 0)
        fallback_used = legacy_fallback_used or bool(
            qa_report.fallback_used
            if qa_report
            else selected and int(_score(selected, "query_priority") or 0) > 0
        )
        scenes.append(
            VisualSceneDiagnostic(
                scene_id=pool.scene.id,
                narration=_short_text(pool.scene.narration, maximum=1000),
                visual_description=_short_text(
                    pool.scene.visual_description,
                    maximum=500,
                ),
                queries=tuple(
                    _short_text(query, maximum=120) for query in pool.attempted_queries
                ),
                candidates_found=len(pool.candidates),
                selected=selected_record,
                candidates=tuple(candidates),
                retries=retries,
                fallback_used=fallback_used,
            )
        )

    selected = [scene.selected for scene in scenes if scene.selected]
    semantic_scores = [
        item.thumbnail_score for item in selected if item.thumbnail_score is not None
    ]
    selected_identities = [
        f"{item.provider.casefold()}:{item.asset_id}" for item in selected
    ]
    duplicates = len(selected_identities) - len(set(selected_identities))
    metrics = VisualMatchingMetrics(
        scene_coverage_percentage=round(
            (len(selected) / len(scenes) * 100.0) if scenes else 0.0,
            2,
        ),
        average_semantic_score=(
            round(sum(semantic_scores) / len(semantic_scores), 6)
            if semantic_scores
            else None
        ),
        retries=sum(scene.retries for scene in scenes),
        fallback_scenes=sum(1 for scene in scenes if scene.fallback_used),
        duplicate_asset_count=max(duplicates, 0),
        scenes_without_candidates=sum(
            1 for scene in scenes if scene.candidates_found == 0
        ),
    )
    return VisualMatchingReport(
        mode=_short_text(mode, maximum=20), metrics=metrics, scenes=tuple(scenes)
    )
