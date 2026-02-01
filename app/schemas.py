from pydantic import BaseModel
from typing import Optional, List, Literal


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
    text: str


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
