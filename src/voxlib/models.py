"""Pydantic domain models for voxlib.

These models are the contract between pipeline stages: each stage
consumes and produces instances of these types, and the repository
translates between them and the SQLAlchemy ORM models. They are
deliberately separate from the ORM (NFR-6) so the pipeline does not
depend on the storage layer.

``Sentence`` is transient and never stored. Every other model here
has a corresponding ORM model.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums for structural values
# ---------------------------------------------------------------------------


class VideoStatus(str, Enum):
    """Lifecycle state of a video in the ingestion pipeline.

    The pipeline advances a video through these states in order. Any
    state may transition to :attr:`FAILED`, and a failed video may be
    retried by re-running the pipeline from its last successful stage.
    """

    PENDING = "pending"
    FETCHED = "fetched"
    CLEANED = "cleaned"
    CHUNKED = "chunked"
    ANALYZED = "analyzed"
    EMBEDDED = "embedded"
    COMPLETED = "completed"
    FAILED = "failed"


class CategoryOrigin(str, Enum):
    """How a category came to exist in the database."""

    SEEDED = "seeded"
    PROPOSED = "proposed"


class EntityKind(str, Enum):
    """Placeholder for the enum that will not be used.

    Entity kinds are configured per-user in YAML and therefore stored
    on :class:`Entity` as a plain string. This enum exists only to
    document the reserved names and to keep the module's shape
    consistent. Do not use it as a type annotation.
    """

    CONCEPT = "concept"
    TECHNIQUE = "technique"
    TERM = "term"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Ingestion-stage models
# ---------------------------------------------------------------------------


class Video(BaseModel):
    """A YouTube video and its ingestion state.

    The ``transcript_cleaned`` field is populated at the end of the
    cleaning stage and is the canonical input to sentence splitting.
    ``transcript_hash`` is a stable digest of the cleaned transcript,
    used to detect drift if a video is refetched.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    youtube_id: str
    title: str | None = None
    url: str
    language: str | None = None
    fetched_at: datetime | None = None
    transcript_cleaned: str | None = None
    transcript_hash: str | None = None
    status: VideoStatus = VideoStatus.PENDING
    last_error: str | None = None


class Sentence(BaseModel):
    """A single sentence within a cleaned transcript.

    Transient: produced by sentence splitting, consumed by semantic
    chunking, never stored. Indices are 0-based and refer to position
    within the video's sentence sequence.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    text: str
    char_start: int
    char_end: int
    token_estimate: int


class EmbedChunk(BaseModel):
    """A small, semantically coherent chunk of the transcript.

    Embed chunks are the atoms of retrieval: each one receives a
    vector and is the unit returned by search. The
    ``analysis_window_id`` field is populated after analysis-window
    grouping and is ``None`` until then.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    video_id: int | None = None
    index: int
    text: str
    sentence_start: int
    sentence_end: int
    token_count: int
    analysis_window_id: int | None = None


class AnalysisWindow(BaseModel):
    """A content-aware group of consecutive embed chunks.

    Analysis windows are the unit of LLM analysis (summarization and
    entity extraction). Boundaries are chosen by topic-shift detection
    subject to token bounds, not by fixed arithmetic. The
    ``boundary_score`` records the similarity at the cut when the
    boundary was content-driven, and is ``None`` for windows whose
    boundary was dictated by size.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    video_id: int | None = None
    index: int
    start_chunk_index: int
    end_chunk_index: int
    text: str
    token_count: int
    boundary_score: float | None = None


# ---------------------------------------------------------------------------
# Analysis-stage models
# ---------------------------------------------------------------------------


class Summary(BaseModel):
    """A node in a video's hierarchical summary tree.

    Level 0 nodes are leaves and correspond one-to-one with analysis
    windows. Higher levels are group summaries. Exactly one node per
    video has ``is_root=True``. ``key_points`` is populated for leaves
    and empty for group summaries.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    video_id: int | None = None
    level: int
    index: int
    parent_id: int | None = None
    analysis_window_id: int | None = None
    is_root: bool = False
    tldr: str
    summary_text: str
    key_points: list[str] = Field(default_factory=list)
    model: str
    created_at: datetime | None = None


class Entity(BaseModel):
    """A concept, technique, term, or other notable item extracted
    from a video.

    Entities are video-scoped: the same real-world thing extracted
    from two videos produces two rows. ``canonical_id`` is reserved
    for v3's entity-resolution step and is always ``None`` in v1.

    ``kind`` is a free string rather than an enum because the
    vocabulary is configured per-user in YAML.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    video_id: int | None = None
    canonical_id: int | None = None
    name: str
    normalized_name: str
    kind: str
    description: str | None = None
    salience: float | None = None


class Category(BaseModel):
    """A topical category that videos can be classified into."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    name: str
    normalized_name: str
    description: str
    origin: CategoryOrigin


class VideoCategory(BaseModel):
    """Association between a video and a category.

    ``is_primary`` is ``True`` for exactly one category per fully
    analyzed video.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    video_id: int
    category_id: int
    is_primary: bool = False


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """A single result from a retrieval query.

    Carries the matched chunk, its similarity score, the id of the
    video it belongs to, and its parent analysis window when known.
    The full :class:`Video` is not embedded here: callers that need it
    should fetch it explicitly by ``video_id``.
    """

    model_config = ConfigDict(from_attributes=True)

    chunk: EmbedChunk
    score: float
    video_id: int
    analysis_window: AnalysisWindow | None = None


# ---------------------------------------------------------------------------
# LLM response schemas (used only by analyze/*)
# ---------------------------------------------------------------------------


class ChunkSummary(BaseModel):
    """LLM output for one analysis window (a level-0 summary)."""

    tldr: str
    summary_text: str
    key_points: list[str] = Field(default_factory=list)


class GroupSummary(BaseModel):
    """LLM output for a group of lower-level summaries."""

    tldr: str
    summary_text: str


class ExtractedEntity(BaseModel):
    """LLM output for one entity extracted from an analysis window."""

    name: str
    kind: str
    description: str | None = None
    salience: float | None = None


class EntityExtractionResult(BaseModel):
    """LLM output for entity extraction over one analysis window."""

    entities: list[ExtractedEntity] = Field(default_factory=list)


class ProposedCategory(BaseModel):
    """LLM output for a newly proposed category."""

    name: str
    description: str


class ClassificationResult(BaseModel):
    """LLM output for video-level classification."""

    primary: str
    secondary: list[str] = Field(default_factory=list)
    proposed: list[ProposedCategory] = Field(default_factory=list)
