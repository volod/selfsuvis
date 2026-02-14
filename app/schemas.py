from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    video_id: str
    job_id: str


class JobStatus(BaseModel):
    status: str
    progress: dict
    started_at: Optional[float]
    finished_at: Optional[float]
    error: Optional[str]


class TextQuery(BaseModel):
    text: str = Field(min_length=1)


class Match(BaseModel):
    score: float
    type: Literal["tile", "frame"]
    video_id: str
    segment_id: int
    t_sec: float
    thumbnail_path: str
    frame_path: Optional[str]
    tile_path: Optional[str]
    bbox: Optional[dict]


class QueryResponse(BaseModel):
    results: List[Match]


# Query parameter constraints
SearchType = Literal["both", "frame", "tile"]
VectorSpace = Literal["clip", "dino"]

# Reusable query params for validation
TOP_K_MIN = 1
TOP_K_MAX = 100


class QueryParams(BaseModel):
    """Common query parameters for search endpoints."""

    top_k: int = Field(default=20, ge=TOP_K_MIN, le=TOP_K_MAX)
    search_type: SearchType = "both"
    enable_rerank: bool = True


class ImageQueryParams(QueryParams):
    """Query parameters for image search."""

    vector_space: VectorSpace = "clip"


class ErrorResponse(BaseModel):
    """Standard error response shape."""

    error: str
    detail: Optional[str] = None
