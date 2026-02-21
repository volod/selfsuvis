from fastapi import APIRouter, Depends

from app.api_utils import ERROR_RESPONSES, error_response
from app.deps import rate_limit, require_api_key
from pipeline.job_db import fetch_job

router = APIRouter(tags=["jobs"], dependencies=[Depends(require_api_key), Depends(rate_limit)])


@router.get("/jobs/{job_id}", summary="Get job status by id", responses={404: ERROR_RESPONSES[404]})
def job_status(job_id: str):
    """Return status, progress, started_at, finished_at, and error for a job."""
    job = fetch_job(job_id)
    if not job:
        return error_response("job not found", status_code=404)
    return {
        "status": job["status"],
        "progress": job["progress"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
    }
