from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from loguru import logger

from app.models.schema import MaterialInfo
from app.models.visual_scene import VisualScene
from app.services import scene_materials


DEFAULT_THRESHOLD = 0.20
DEFAULT_FRAME_COUNT = 3
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RECENT_ASSET_WINDOW = 1
MAX_FRAME_COUNT = 5
MAX_CANDIDATE_ATTEMPTS = 10


class StrictVisualQAMismatch(RuntimeError):
    pass


class FrameScorer(Protocol):
    def score_images(
        self,
        description: str,
        images: Sequence[Any],
    ) -> list[float]: ...


FrameExtractor = Callable[[str, int], Sequence[Any]]
CandidateDownloader = Callable[[MaterialInfo], str]


@dataclass(frozen=True)
class VisualQAConfig:
    enabled: bool = False
    threshold: float = DEFAULT_THRESHOLD
    frame_count: int = DEFAULT_FRAME_COUNT
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    fail_on_mismatch: bool = False
    recent_asset_window: int = DEFAULT_RECENT_ASSET_WINDOW

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> VisualQAConfig:
        try:
            threshold = float(values.get("visual_qa_threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            threshold = DEFAULT_THRESHOLD
        try:
            frame_count = int(values.get("visual_qa_frame_count", DEFAULT_FRAME_COUNT))
        except (TypeError, ValueError):
            frame_count = DEFAULT_FRAME_COUNT
        try:
            max_attempts = int(
                values.get("visual_qa_max_attempts", DEFAULT_MAX_ATTEMPTS)
            )
        except (TypeError, ValueError):
            max_attempts = DEFAULT_MAX_ATTEMPTS
        try:
            recent_window = int(
                values.get(
                    "visual_qa_recent_asset_window",
                    DEFAULT_RECENT_ASSET_WINDOW,
                )
            )
        except (TypeError, ValueError):
            recent_window = DEFAULT_RECENT_ASSET_WINDOW
        return cls(
            enabled=bool(values.get("visual_qa_enabled", False)),
            threshold=max(-1.0, min(threshold, 1.0)),
            frame_count=max(1, min(frame_count, MAX_FRAME_COUNT)),
            max_attempts=max(1, min(max_attempts, MAX_CANDIDATE_ATTEMPTS)),
            fail_on_mismatch=bool(values.get("visual_qa_fail_on_mismatch", False)),
            recent_asset_window=max(0, min(recent_window, 10)),
        )


@dataclass(frozen=True)
class VisualQARejection:
    asset_id: str
    query: str
    score: float | None
    reason: str


@dataclass(frozen=True)
class VisualQAReport:
    scene_id: int
    visual_description: str
    queries: tuple[str, ...]
    candidates_considered: int
    selected_asset_id: str | None
    thumbnail_score: float | None
    video_qa_score: float | None
    attempts: int
    fallback_used: bool
    selection_reason: str
    rejections: tuple[VisualQARejection, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualQASelection:
    scene: VisualScene
    material: MaterialInfo | None
    local_path: str | None
    report: VisualQAReport


def _source(item: MaterialInfo) -> dict[str, Any]:
    return item.source_info if isinstance(item.source_info, dict) else {}


def _asset_id(item: MaterialInfo) -> str:
    source = _source(item)
    value = source.get("asset_id")
    if value not in (None, ""):
        return str(value)
    return scene_materials.candidate_identity(item)


def _query(item: MaterialInfo) -> str:
    value = _source(item).get("search_query")
    return str(value) if value not in (None, "") else ""


def _numeric_metadata(item: MaterialInfo, key: str) -> float | None:
    value = _source(item).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def aggregate_frame_scores(scores: Sequence[float]) -> float:
    """Average the best two sampled frames, or the only frame when singular."""
    if not scores:
        raise ValueError("visual QA produced no frame scores")
    top_scores = sorted((float(score) for score in scores), reverse=True)[:2]
    return sum(top_scores) / len(top_scores)


def sample_video_frames(video_path: str, frame_count: int) -> list[Any]:
    """Decode bounded frames at interior, evenly spaced video positions."""
    from PIL import Image
    from moviepy.video.io.VideoFileClip import VideoFileClip

    bounded_count = max(1, min(int(frame_count), MAX_FRAME_COUNT))
    clip = VideoFileClip(video_path, audio=False)
    try:
        duration = float(clip.duration or 0)
        if duration <= 0:
            raise ValueError("video has no usable duration")
        # Sample from 10% through 90% to avoid fade-prone exact endpoints while
        # retaining beginning, middle, and end context.
        positions = (
            [0.5]
            if bounded_count == 1
            else [
                0.1 + index * 0.8 / (bounded_count - 1)
                for index in range(bounded_count)
            ]
        )
        timestamps = [duration * position for position in positions]
        return [
            Image.fromarray(clip.get_frame(timestamp)).convert("RGB")
            for timestamp in timestamps
        ]
    finally:
        clip.close()


def prioritize_non_recent(
    candidates: Sequence[MaterialInfo],
    recent_identities: Sequence[str],
) -> list[MaterialInfo]:
    """Defer recently selected assets while preserving all deterministic order."""
    recent = set(recent_identities)
    available = [
        item
        for item in candidates
        if scene_materials.candidate_identity(item) not in recent
    ]
    repeated = [
        item
        for item in candidates
        if scene_materials.candidate_identity(item) in recent
    ]
    return available + repeated


def prioritize_query_coverage(
    candidates: Sequence[MaterialInfo],
) -> list[MaterialInfo]:
    """Round-robin ranked results so bounded retries reach fallback queries."""
    groups: dict[int, list[MaterialInfo]] = {}
    for item in candidates:
        priority = int(_numeric_metadata(item, "query_priority") or 0)
        groups.setdefault(priority, []).append(item)

    ordered: list[MaterialInfo] = []
    while any(groups.values()):
        for priority in sorted(groups):
            if groups[priority]:
                ordered.append(groups[priority].pop(0))
    return ordered


def _report(
    *,
    scene: VisualScene,
    candidates: Sequence[MaterialInfo],
    selected: MaterialInfo | None,
    score: float | None,
    attempts: int,
    fallback_used: bool,
    selection_reason: str,
    rejections: Sequence[VisualQARejection],
) -> VisualQAReport:
    return VisualQAReport(
        scene_id=scene.id,
        visual_description=scene.visual_description,
        queries=tuple(scene.search_queries),
        candidates_considered=len(candidates),
        selected_asset_id=_asset_id(selected) if selected else None,
        thumbnail_score=(
            _numeric_metadata(selected, "visual_match_score") if selected else None
        ),
        video_qa_score=score,
        attempts=attempts,
        fallback_used=fallback_used,
        selection_reason=selection_reason,
        rejections=tuple(rejections),
    )


def qa_scene_candidates(
    scene: VisualScene,
    candidates: Sequence[MaterialInfo],
    *,
    scorer: FrameScorer,
    download: CandidateDownloader,
    extract_frames: FrameExtractor = sample_video_frames,
    threshold: float = DEFAULT_THRESHOLD,
    frame_count: int = DEFAULT_FRAME_COUNT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    fail_on_mismatch: bool = False,
    recent_identities: Sequence[str] = (),
) -> VisualQASelection:
    query_balanced = prioritize_query_coverage(candidates)
    ordered = prioritize_non_recent(query_balanced, recent_identities)
    bounded_attempts = max(1, min(int(max_attempts), MAX_CANDIDATE_ATTEMPTS))
    bounded_frames = max(1, min(int(frame_count), MAX_FRAME_COUNT))
    rejections: list[VisualQARejection] = []
    best: tuple[float, MaterialInfo, str] | None = None
    attempts = 0
    used_fallback_query = False

    for item in ordered[:bounded_attempts]:
        attempts += 1
        query_priority = int(_numeric_metadata(item, "query_priority") or 0)
        used_fallback_query = used_fallback_query or query_priority > 0
        try:
            local_path = download(item)
        except Exception as exc:
            rejections.append(
                VisualQARejection(
                    asset_id=_asset_id(item),
                    query=_query(item),
                    score=None,
                    reason=f"download_failed:{type(exc).__name__}",
                )
            )
            continue
        if not local_path:
            rejections.append(
                VisualQARejection(
                    asset_id=_asset_id(item),
                    query=_query(item),
                    score=None,
                    reason="download_failed",
                )
            )
            continue

        try:
            frames = extract_frames(local_path, bounded_frames)
            score = aggregate_frame_scores(
                scorer.score_images(scene.visual_description, frames)
            )
        except Exception as exc:
            reason = f"qa_unavailable:{type(exc).__name__}"
            logger.warning(
                "strict visual QA unavailable for candidate: "
                f"scene_id={scene.id}, asset_id={_asset_id(item)}, reason={reason}"
            )
            if not fail_on_mismatch:
                return VisualQASelection(
                    scene=scene,
                    material=item,
                    local_path=local_path,
                    report=_report(
                        scene=scene,
                        candidates=ordered,
                        selected=item,
                        score=None,
                        attempts=attempts,
                        fallback_used=used_fallback_query,
                        selection_reason="qa_unavailable_fallback",
                        rejections=rejections,
                    ),
                )
            rejections.append(
                VisualQARejection(
                    asset_id=_asset_id(item),
                    query=_query(item),
                    score=None,
                    reason=reason,
                )
            )
            break

        if score >= threshold:
            return VisualQASelection(
                scene=scene,
                material=item,
                local_path=local_path,
                report=_report(
                    scene=scene,
                    candidates=ordered,
                    selected=item,
                    score=score,
                    attempts=attempts,
                    fallback_used=used_fallback_query,
                    selection_reason="threshold_passed",
                    rejections=rejections,
                ),
            )

        rejections.append(
            VisualQARejection(
                asset_id=_asset_id(item),
                query=_query(item),
                score=score,
                reason="below_threshold",
            )
        )
        if best is None or score > best[0]:
            best = (score, item, local_path)

    if best is not None and not fail_on_mismatch:
        score, item, local_path = best
        logger.warning(
            "no scene candidate passed strict visual QA; accepting best available: "
            f"scene_id={scene.id}, asset_id={_asset_id(item)}, score={score:.4f}, "
            f"threshold={threshold:.4f}"
        )
        return VisualQASelection(
            scene=scene,
            material=item,
            local_path=local_path,
            report=_report(
                scene=scene,
                candidates=ordered,
                selected=item,
                score=score,
                attempts=attempts,
                fallback_used=used_fallback_query,
                selection_reason="best_available_below_threshold",
                rejections=rejections,
            ),
        )

    return VisualQASelection(
        scene=scene,
        material=None,
        local_path=None,
        report=_report(
            scene=scene,
            candidates=ordered,
            selected=None,
            score=None,
            attempts=attempts,
            fallback_used=any(
                int(_numeric_metadata(item, "query_priority") or 0) > 0
                for item in ordered[:attempts]
            ),
            selection_reason="no_candidate_passed",
            rejections=rejections,
        ),
    )
