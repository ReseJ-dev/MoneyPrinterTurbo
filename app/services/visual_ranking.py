from __future__ import annotations

import io
import importlib.util
import threading
from dataclasses import replace
from functools import lru_cache
from typing import Any, Mapping, Protocol, Sequence

import requests
from loguru import logger

from app.models.schema import MaterialInfo
from app.models.visual_scene import VisualScene
from app.services.scene_materials import SceneCandidatePool


DEFAULT_PROVIDER = "local"
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
MAX_PREVIEW_BYTES = 10 * 1024 * 1024
MAX_TEXT_EMBEDDINGS = 128


def local_visual_ai_dependencies_available() -> bool:
    """Check optional packages without importing or initializing the model."""
    try:
        return all(
            importlib.util.find_spec(package) is not None
            for package in ("open_clip", "torch")
        )
    except (ImportError, ValueError):
        return False


class VisualRankingUnavailable(RuntimeError):
    pass


class UnavailableVisualScorer:
    def __init__(self, reason: str):
        self.reason = reason

    def score_images(
        self,
        description: str,
        images: Sequence[Any],
    ) -> list[float]:
        raise VisualRankingUnavailable(self.reason)


class VisualRanker(Protocol):
    def rank(
        self,
        scene: VisualScene,
        candidates: Sequence[MaterialInfo],
    ) -> list[MaterialInfo]: ...


def _clone_with_score(item: MaterialInfo, score: float) -> MaterialInfo:
    source = dict(item.source_info) if isinstance(item.source_info, dict) else {}
    source["visual_match_score"] = float(score)
    return MaterialInfo(
        provider=item.provider,
        url=item.url,
        duration=item.duration,
        source_info=source,
    )


def _metadata_int(item: MaterialInfo, field: str, default: int) -> int:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    try:
        return int(source.get(field, default))
    except (TypeError, ValueError, OverflowError):
        return default


def _resolution_pixels(item: MaterialInfo) -> int:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    rendition = source.get("rendition")
    if not isinstance(rendition, dict):
        return 0
    try:
        return int(rendition.get("width") or 0) * int(rendition.get("height") or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _rank_key(
    item: MaterialInfo,
    *,
    original_index: int,
) -> tuple[int, float, int, int, int, int, int]:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    score = source.get("visual_match_score")
    has_score = isinstance(score, (int, float)) and not isinstance(score, bool)
    semantic_order = -float(score) if has_score else 0.0
    return (
        0 if has_score else 1,
        semantic_order,
        _metadata_int(item, "query_priority", original_index),
        _metadata_int(item, "provider_result_order", original_index),
        -_resolution_pixels(item),
        -int(item.duration),
        original_index,
    )


_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model_bundle(model_name: str, pretrained: str):
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise VisualRankingUnavailable(
            "local visual ranking dependencies are unavailable; install with "
            "`uv sync --extra visual-ranking`"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    logger.info(
        "loaded local visual ranking model: "
        f"model={model_name}, pretrained={pretrained}, device={device}"
    )
    return torch, model, preprocess, tokenizer, device


def _get_model_bundle(model_name: str, pretrained: str):
    key = (model_name, pretrained)
    bundle = _MODEL_CACHE.get(key)
    if bundle is not None:
        return bundle
    with _MODEL_CACHE_LOCK:
        bundle = _MODEL_CACHE.get(key)
        if bundle is None:
            bundle = _load_model_bundle(model_name, pretrained)
            _MODEL_CACHE[key] = bundle
    return bundle


def _download_preview_image(url: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise VisualRankingUnavailable(
            "Pillow is required for local visual ranking"
        ) from exc

    with requests.get(url, stream=True, timeout=(10, 30)) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_PREVIEW_BYTES:
            raise ValueError("candidate preview image exceeds the size limit")
        payload = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            payload.extend(chunk)
            if len(payload) > MAX_PREVIEW_BYTES:
                raise ValueError("candidate preview image exceeds the size limit")
    return Image.open(io.BytesIO(payload)).convert("RGB")


class LocalVisualRanker:
    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained
        self._text_embeddings: dict[str, Any] = {}
        self._text_cache_lock = threading.Lock()

    def _text_embedding(self, description: str, bundle):
        cached = self._text_embeddings.get(description)
        if cached is not None:
            return cached
        torch, model, _, tokenizer, device = bundle
        with self._text_cache_lock:
            cached = self._text_embeddings.get(description)
            if cached is None:
                tokens = tokenizer([description]).to(device)
                with torch.no_grad():
                    cached = model.encode_text(tokens)
                    cached = cached / cached.norm(dim=-1, keepdim=True)
                if len(self._text_embeddings) >= MAX_TEXT_EMBEDDINGS:
                    oldest = next(iter(self._text_embeddings))
                    self._text_embeddings.pop(oldest, None)
                self._text_embeddings[description] = cached
        return cached

    def _score_thumbnails(
        self,
        description: str,
        preview_urls: Sequence[str],
    ) -> list[float]:
        images = [_download_preview_image(url) for url in preview_urls]
        return self.score_images(description, images)

    def score_images(
        self,
        description: str,
        images: Sequence[Any],
    ) -> list[float]:
        """Compare in-memory images with text using the cached local model."""
        if not images:
            return []
        bundle = _get_model_bundle(self.model_name, self.pretrained)
        torch, model, preprocess, _, device = bundle
        text_embedding = self._text_embedding(description, bundle)
        image_batch = torch.stack([preprocess(image) for image in images]).to(device)
        with torch.no_grad():
            image_embeddings = model.encode_image(image_batch)
            image_embeddings = image_embeddings / image_embeddings.norm(
                dim=-1,
                keepdim=True,
            )
            scores = image_embeddings @ text_embedding.T
        return [float(score) for score in scores.reshape(-1).detach().cpu().tolist()]

    def rank(
        self,
        scene: VisualScene,
        candidates: Sequence[MaterialInfo],
    ) -> list[MaterialInfo]:
        ranked_candidates = list(candidates)
        score_indexes: list[int] = []
        preview_urls: list[str] = []
        for index, item in enumerate(ranked_candidates):
            source = item.source_info if isinstance(item.source_info, dict) else {}
            preview_url = source.get("preview_url")
            if isinstance(preview_url, str) and preview_url.strip():
                score_indexes.append(index)
                preview_urls.append(preview_url.strip())

        if not preview_urls:
            return ranked_candidates

        scores = self._score_thumbnails(scene.visual_description, preview_urls)
        if len(scores) != len(score_indexes):
            raise ValueError("visual ranker returned an unexpected score count")
        for index, score in zip(score_indexes, scores, strict=True):
            ranked_candidates[index] = _clone_with_score(
                ranked_candidates[index],
                score,
            )

        indexed = list(enumerate(ranked_candidates))
        indexed.sort(key=lambda pair: _rank_key(pair[1], original_index=pair[0]))
        return [item for _, item in indexed]


@lru_cache(maxsize=4)
def _get_local_ranker(model_name: str, pretrained: str) -> LocalVisualRanker:
    return LocalVisualRanker(model_name=model_name, pretrained=pretrained)


def configured_ranker(app_config: Mapping[str, Any]) -> VisualRanker | None:
    if not app_config.get("visual_ranking_enabled", False):
        return None
    return configured_local_scorer(app_config)


def configured_local_scorer(app_config: Mapping[str, Any]) -> LocalVisualRanker | None:
    """Create the shared local embedding service regardless of ranking mode."""
    provider = (
        str(app_config.get("visual_ranking_provider", DEFAULT_PROVIDER)).strip().lower()
    )
    if provider != DEFAULT_PROVIDER:
        logger.warning(
            f"unsupported visual ranking provider {provider!r}; use deterministic ranking"
        )
        return None
    model_name = str(app_config.get("visual_ranking_model", DEFAULT_MODEL)).strip()
    pretrained = str(
        app_config.get("visual_ranking_pretrained", DEFAULT_PRETRAINED)
    ).strip()
    return _get_local_ranker(
        model_name or DEFAULT_MODEL,
        pretrained or DEFAULT_PRETRAINED,
    )


def rank_candidate_pools(
    pools: Sequence[SceneCandidatePool],
    ranker: VisualRanker | None,
) -> list[SceneCandidatePool]:
    if ranker is None:
        return list(pools)

    ranked_pools: list[SceneCandidatePool] = []
    try:
        for pool in pools:
            ranked_pools.append(
                replace(
                    pool,
                    candidates=tuple(ranker.rank(pool.scene, pool.candidates)),
                )
            )
    except Exception as exc:
        logger.warning(
            "visual semantic ranking unavailable; use deterministic candidate "
            f"order: error={type(exc).__name__}, detail={exc}"
        )
        return list(pools)
    return ranked_pools
