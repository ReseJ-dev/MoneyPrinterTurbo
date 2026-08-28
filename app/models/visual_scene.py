from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class VisualScene(BaseModel):
    """A chronological narration segment and its stock-footage search intent."""

    id: int = Field(ge=1)
    narration: str
    visual_description: str
    search_queries: list[str] = Field(min_length=1)

    @field_validator("narration", "visual_description")
    @classmethod
    def _require_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("scene text must not be empty")
        return normalized

    @field_validator("search_queries")
    @classmethod
    def _normalize_queries(cls, value: list[str]) -> list[str]:
        normalized = [query.strip() for query in value]
        if any(not query for query in normalized):
            raise ValueError("scene search queries must not be empty")
        return normalized

    @model_validator(mode="after")
    def _require_distinct_queries(self) -> "VisualScene":
        normalized = [query.casefold() for query in self.search_queries]
        if len(normalized) != len(set(normalized)):
            raise ValueError("scene search queries must be distinct")
        return self
