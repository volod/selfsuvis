"""LangGraph state schema for the 24-step local learning pipeline.

``PipelineState`` is a TypedDict consumed by every graph node.  Each node
returns a *partial* dict containing only the keys it writes; LangGraph merges
them with last-writer-wins (the default reducer for TypedDict fields).

Design notes
------------
* ``knowledge``  — VideoKnowledge instance stored as a Python object reference.
  MemorySaver keeps it alive in-process.  SqliteSaver reconstruction uses
  ``_reconstruct_knowledge`` (at resume time only, not per-step).
* ``models``     — Large PyTorch objects; never serialised.  On resume they are
  re-injected from the caller's namespace via ``run_graph_pipeline``.
* Parallel fan-out nodes (steps 4–8) each write *distinct* keys so there is
  no reducer conflict.  ``node_p2_merge_parallel`` reads all five and commits
  to ``knowledge`` and ``video_context``.
"""

from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Runtime config (injected once at graph entry, never mutated) ─────────
    args: Any                             # parsed CLI args namespace
    video_path: str                       # absolute path
    video_name: str
    video_id: str
    video_dir: str                        # absolute path of per-video output dir
    output_dir: str
    device: str
    models: Dict[str, Any]               # {clip, dino, uses_api_embedder, …}
    store: Any                            # Qdrant client or InMemoryStore
    is_qdrant: bool

    # ── Phase 1 outputs ───────────────────────────────────────────────────────
    frame_list: List[Tuple[str, float]]  # [(frame_path, t_sec), …]
    frames_meta: Dict[str, Any]
    knowledge: Any                        # VideoKnowledge instance
    clip_dino_on_gpu: bool

    # ── Phase 2 outputs — one key per step ───────────────────────────────────
    gemma_result: Dict[str, Any]
    caption_results: List[Dict[str, Any]]
    asr_result: Dict[str, Any]
    platform_fusion_result: Dict[str, Any]
    ocr_result: Dict[str, Any]
    depth_result: Dict[str, Any]
    det_result: Dict[str, Any]
    yolo_sam_result: Dict[str, Any]
    gemma_tracking_result: Dict[str, Any]
    world_result: Dict[str, Any]
    qwen_result: Dict[str, Any]
    unidrive_result: Dict[str, Any]
    scenetok_result: Dict[str, Any]
    base_results: List[Dict[str, Any]]
    query_frame: str
    query_t_sec: float
    map_result: Dict[str, Any]
    semantic_graph_result: Dict[str, Any]
    full_fusion_result: Dict[str, Any]

    # ── Phase 3 SSL outputs ───────────────────────────────────────────────────
    ssl_result: Dict[str, Any]
    ssl_gate_passed: bool
    checkpoint_path: str
    student_backbone: Any
    student_dim: int
    distill_result: Dict[str, Any]
    export_result: Dict[str, Any]
    ft_results: List[Dict[str, Any]]
    compare_result: Dict[str, Any]

    # ── Phase 4 outputs ───────────────────────────────────────────────────────
    multi_model_result: Dict[str, Any]
    synthesis_result: Dict[str, Any]
    audit_result: Dict[str, Any]

    # ── Cross-cutting accumulation ────────────────────────────────────────────
    stats: Dict[str, Any]               # timing dict T + numeric summaries
    video_context: Dict[str, Any]       # rich context fed to LLM synthesis/audit
    agentic_trace: List[Dict[str, Any]]

    # ── Resume support ───────────────────────────────────────────────────────
    completed_phases: List[str]
    error: Optional[str]
