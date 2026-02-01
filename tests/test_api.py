import os
import time
import logging
import requests
import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")
ASSETS_DIR = os.getenv("ASSETS_DIR", os.path.join(os.path.dirname(__file__), "assets"))
INDEX_DIR_PATH = os.getenv("INDEX_DIR_PATH")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def _wait_job(job_id: str, timeout_sec: int = 120):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        resp = requests.get(f"{API_URL}/jobs/{job_id}")
        resp.raise_for_status()
        status = resp.json().get("status")
        if status in {"finished", "error"}:
            return resp.json()
        time.sleep(2)
    raise TimeoutError(f"job timeout: {job_id}")


def _wait_api(timeout_sec: int = 120) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(f"{API_URL}/docs")
            if resp.status_code in {200, 404}:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    if not _wait_api():
        pytest.skip("API not reachable")


def test_index_video_and_query_text():
    green_video = os.path.join(ASSETS_DIR, "vid_green.mp4")
    assert os.path.exists(green_video)

    with open(green_video, "rb") as f:
        resp = requests.post(
            f"{API_URL}/index/video",
            files={"file": ("vid_green.mp4", f, "video/mp4")},
            data={"enable_tiles": "false"},
        )
    resp.raise_for_status()
    job = resp.json()
    job_status = _wait_job(job["job_id"])
    assert job_status["status"] == "finished"

    resp = requests.post(
        f"{API_URL}/query/text",
        json={"text": "green field"},
        params={"top_k": 5, "search_type": "frame", "enable_rerank": False},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    assert isinstance(results, list)


def test_index_video_and_query_image():
    test_video = os.path.join(ASSETS_DIR, "vid_testsrc.mp4")
    query_img = os.path.join(ASSETS_DIR, "green.png")
    assert os.path.exists(test_video)
    assert os.path.exists(query_img)

    with open(test_video, "rb") as f:
        resp = requests.post(
            f"{API_URL}/index/video",
            files={"file": ("vid_testsrc.mp4", f, "video/mp4")},
            data={"enable_tiles": "true"},
        )
    resp.raise_for_status()
    job = resp.json()
    job_status = _wait_job(job["job_id"])
    assert job_status["status"] == "finished"

    with open(query_img, "rb") as f:
        resp = requests.post(
            f"{API_URL}/query/image",
            files={"file": ("green.png", f, "image/png")},
            data={"top_k": "5", "search_type": "both", "vector_space": "clip", "enable_rerank": "false"},
        )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    assert len(results) > 0


def test_precheck_dir_optional_enqueue():
    if not INDEX_DIR_PATH:
        pytest.skip("INDEX_DIR_PATH not set")
    resp = requests.post(
        f"{API_URL}/index/precheck_dir",
        data={"path": INDEX_DIR_PATH, "enqueue": "false"},
    )
    resp.raise_for_status()
    data = resp.json()
    assert "results" in data


def test_index_dir_optional():
    if not INDEX_DIR_PATH:
        pytest.skip("INDEX_DIR_PATH not set")
    resp = requests.post(
        f"{API_URL}/index/dir",
        data={"path": INDEX_DIR_PATH, "enable_tiles": "false"},
    )
    resp.raise_for_status()
    data = resp.json()
    assert "jobs" in data
