"""Unit tests for voxlib.models."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from voxlib.models import (
    AnalysisWindow,
    Category,
    CategoryOrigin,
    ChunkSummary,
    ClassificationResult,
    EmbedChunk,
    Entity,
    ProposedCategory,
    SearchHit,
    Sentence,
    Summary,
    Video,
    VideoCategory,
    VideoStatus,
)


def test_video_defaults_to_pending():
    """A newly constructed Video starts in PENDING."""
    video = Video(youtube_id="abc", url="https://youtu.be/abc")
    assert video.status is VideoStatus.PENDING
    assert video.id is None


def test_video_status_rejects_unknown_value():
    """VideoStatus is a closed enum."""
    with pytest.raises(ValidationError):
        Video(
            youtube_id="abc",
            url="https://youtu.be/abc",
            status="not-a-status",
        )


def test_video_status_accepts_string_value():
    """A raw string matching an enum value is coerced."""
    video = Video(
        youtube_id="abc",
        url="https://youtu.be/abc",
        status="fetched",
    )
    assert video.status is VideoStatus.FETCHED


def test_sentence_is_frozen():
    """Sentences cannot be mutated after construction."""
    sentence = Sentence(
        index=0,
        text="Hello.",
        char_start=0,
        char_end=6,
        token_estimate=2,
    )
    with pytest.raises(ValidationError):
        sentence.text = "Changed."


def test_sentence_requires_all_fields():
    """A sentence without an index is invalid."""
    with pytest.raises(ValidationError):
        Sentence(text="Hello.", char_start=0, char_end=6, token_estimate=2)


def test_embed_chunk_analysis_window_id_defaults_to_none():
    """Chunks are not attached to a window until grouping runs."""
    chunk = EmbedChunk(
        index=0,
        text="Some text.",
        sentence_start=0,
        sentence_end=2,
        token_count=5,
    )
    assert chunk.analysis_window_id is None


def test_analysis_window_boundary_score_optional():
    """Windows with size-driven boundaries have no boundary score."""
    window = AnalysisWindow(
        index=0,
        start_chunk_index=0,
        end_chunk_index=3,
        text="Some text.",
        token_count=900,
    )
    assert window.boundary_score is None


def test_summary_key_points_default_empty():
    """Group summaries carry no key points; default is an empty list."""
    summary = Summary(
        level=1,
        index=0,
        tldr="tldr",
        summary_text="body",
        model="llama3.1:8b",
    )
    assert summary.key_points == []
    assert summary.is_root is False


def test_summary_root_flag():
    """The root summary can be marked explicitly."""
    summary = Summary(
        level=3,
        index=0,
        is_root=True,
        tldr="tldr",
        summary_text="body",
        model="llama3.1:8b",
    )
    assert summary.is_root is True


def test_entity_kind_is_a_plain_string():
    """Entity.kind accepts any string, by design."""
    entity = Entity(
        video_id=1,
        name="Spaced Repetition",
        normalized_name="spaced repetition",
        kind="technique",
    )
    assert entity.kind == "technique"
    entity_unknown = Entity(
        video_id=1,
        name="X",
        normalized_name="x",
        kind="something-the-llm-invented",
    )
    assert entity_unknown.kind == "something-the-llm-invented"


def test_entity_canonical_id_is_none_in_v1():
    """canonical_id is reserved for v3 and unset in v1."""
    entity = Entity(
        video_id=1,
        name="React",
        normalized_name="react",
        kind="tool",
    )
    assert entity.canonical_id is None


def test_category_origin_is_enum():
    """Category.origin is a closed enum."""
    category = Category(
        name="Programming",
        normalized_name="programming",
        description="Software.",
        origin="seeded",
    )
    assert category.origin is CategoryOrigin.SEEDED
    with pytest.raises(ValidationError):
        Category(
            name="X",
            normalized_name="x",
            description="",
            origin="invented",
        )


def test_video_category_defaults_to_secondary():
    """A VideoCategory is secondary unless marked primary."""
    link = VideoCategory(video_id=1, category_id=2)
    assert link.is_primary is False


def test_search_hit_carries_chunk_and_score():
    """A search hit bundles a chunk with its score and window."""
    chunk = EmbedChunk(
        id=1,
        video_id=1,
        index=0,
        text="text",
        sentence_start=0,
        sentence_end=1,
        token_count=5,
    )
    window = AnalysisWindow(
        id=1,
        video_id=1,
        index=0,
        start_chunk_index=0,
        end_chunk_index=0,
        text="text",
        token_count=5,
    )
    hit = SearchHit(
        chunk=chunk,
        score=0.87,
        video_id=1,
        analysis_window=window,
    )
    assert hit.chunk.id == 1
    assert hit.analysis_window is not None
    assert hit.analysis_window.id == 1


def test_search_hit_analysis_window_optional():
    """The analysis window is optional in a search hit."""
    chunk = EmbedChunk(
        id=1,
        video_id=1,
        index=0,
        text="text",
        sentence_start=0,
        sentence_end=1,
        token_count=5,
    )
    hit = SearchHit(chunk=chunk, score=0.5, video_id=1)
    assert hit.analysis_window is None


def test_chunk_summary_requires_tldr_and_text():
    """ChunkSummary requires both summary fields."""
    with pytest.raises(ValidationError):
        ChunkSummary(tldr="only tldr")


def test_classification_result_defaults_are_empty():
    """Secondary and proposed default to empty lists."""
    result = ClassificationResult(primary="Programming")
    assert result.secondary == []
    assert result.proposed == []


def test_classification_result_accepts_proposals():
    """Proposed categories parse into ProposedCategory instances."""
    result = ClassificationResult(
        primary="Programming",
        proposed=[
            {"name": "Rust", "description": "The Rust language."},
        ],
    )
    assert isinstance(result.proposed[0], ProposedCategory)
    assert result.proposed[0].name == "Rust"


def test_created_at_accepts_datetime():
    """Summary.created_at accepts a datetime."""
    now = datetime(2026, 1, 1, 12, 0, 0)
    summary = Summary(
        level=0,
        index=0,
        tldr="t",
        summary_text="s",
        model="m",
        created_at=now,
    )
    assert summary.created_at == now
