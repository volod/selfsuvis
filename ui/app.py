import io
import os
import time
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

# If API_KEY is configured, send it with every request.
_API_KEY = os.getenv("API_KEY", "")
_HEADERS = {"X-API-Key": _API_KEY} if _API_KEY else {}

st.set_page_config(page_title="Video Semantic Search", layout="wide")

st.title("Video Semantic Search (POC)")


def _render_results(results):
    st.subheader("Results")
    cols = st.columns(4)
    for i, r in enumerate(results):
        col = cols[i % 4]
        with col:
            thumb = r.get("thumbnail_path")
            if thumb and os.path.exists(thumb):
                st.image(thumb, use_container_width=True)
            st.write(f"Score: {r['score']:.4f}")
            st.write(f"Video: {r['video_id']}")
            st.write(f"t={r['t_sec']:.2f}s")
            if r.get("frame_path"):
                st.caption(r.get("frame_path"))
            if r.get("tile_path"):
                st.caption(r.get("tile_path"))
            if r.get("video_id"):
                st.code(f"mpv \"./data/videos/{r['video_id']}.mp4\" --start={r['t_sec']:.2f}")


tab_index, tab_image, tab_text, tab_admin = st.tabs(
    ["Index Video", "Image Query", "Text Query", "Admin"]
)

with tab_index:
    st.header("Index Video")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv"])
    url_input = st.text_input("Or URL (HTTP/S)")
    path_input = st.text_input("Or directory path (inside container, for batch indexing)")
    enable_tiles = st.checkbox("Enable tile indexing", value=True)
    if st.button("Start Indexing"):
        if uploaded:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"enable_tiles": str(enable_tiles).lower()}
            resp = requests.post(f"{API_URL}/index/video", files=files, data=data, headers=_HEADERS)
        elif url_input:
            data = {"url": url_input, "enable_tiles": str(enable_tiles).lower()}
            resp = requests.post(f"{API_URL}/index/url", data=data, headers=_HEADERS)
        elif path_input:
            data = {"path": path_input, "enable_tiles": str(enable_tiles).lower()}
            resp = requests.post(f"{API_URL}/index/dir", data=data, headers=_HEADERS)
        else:
            st.error("Upload a video or provide URL/path.")
            resp = None

        if resp is not None:
            if resp.ok:
                job = resp.json()
                if "job_id" in job:
                    st.session_state["job_id"] = job["job_id"]
                    st.success(f"Job created: {job['job_id']}")
                else:
                    st.json(job)
            else:
                st.error(resp.text)

    job_id = st.text_input("Job ID", value=st.session_state.get("job_id", ""))
    if st.button("Refresh Status") and job_id:
        resp = requests.get(f"{API_URL}/jobs/{job_id}", headers=_HEADERS)
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)

with tab_image:
    st.header("Image Query")
    img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img")
    search_type = st.selectbox("Search type", ["both", "frame", "tile"], index=0)
    vector_space = st.selectbox("Vector space", ["clip", "dino"], index=0)
    enable_rerank = st.checkbox("Enable rerank", value=True, key="img_rerank")
    top_k = st.slider("Top-K", min_value=5, max_value=50, value=20)
    if st.button("Search", key="img_search"):
        if not img:
            st.error("Please upload an image.")
        else:
            files = {"file": (img.name, img.getvalue(), img.type)}
            data = {
                "top_k": str(top_k),
                "search_type": search_type,
                "vector_space": vector_space,
                "enable_rerank": str(enable_rerank).lower(),
            }
            resp = requests.post(f"{API_URL}/query/image", files=files, data=data, headers=_HEADERS)
            if resp.ok:
                results = resp.json().get("results", [])
                _render_results(results)
            else:
                st.error(resp.text)

with tab_admin:
    st.header("Admin")
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh", key="admin_refresh"):
            st.rerun()

    try:
        resp = requests.get(f"{API_URL}/admin/stats", headers=_HEADERS, timeout=5)
        if resp.ok:
            stats = resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
            stats = None
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
        stats = None

    if stats:
        jobs = stats.get("jobs", {})
        al_tags = stats.get("al_tags", {})
        worker_active = stats.get("worker_active", False)

        # Worker status badge
        if worker_active:
            st.success("Worker: ACTIVE")
        else:
            st.info("Worker: idle")

        # Queue depth metric
        st.metric("Queue depth (pending)", jobs.get("pending", 0))

        # Job status breakdown
        st.subheader("Job Status")
        col_p, col_r, col_d, col_e = st.columns(4)
        col_p.metric("Pending", jobs.get("pending", 0))
        col_r.metric("Running", jobs.get("running", 0))
        col_d.metric("Done", jobs.get("done", 0))
        col_e.metric("Error", jobs.get("error", 0))

        # AL tag distribution bar chart
        st.subheader("AL Tag Distribution")
        na = al_tags.get("needs_annotation", 0)
        novel = al_tags.get("novel", 0)
        none_count = al_tags.get("none", 0)
        total = na + novel + none_count

        if total == 0:
            st.caption("No indexed frames yet.")
        else:
            # Use st.bar_chart with a simple dict
            import pandas as pd
            chart_data = pd.DataFrame(
                {"count": [na, novel, none_count]},
                index=["needs_annotation", "novel", "none"],
            )
            st.bar_chart(chart_data)
            st.caption(
                f"Total frames: {total} — "
                f"ANNOTATE: {na} ({100*na//total}%) · "
                f"NOVEL: {novel} ({100*novel//total}%) · "
                f"none: {none_count} ({100*none_count//total}%)"
            )

    # ── 3DGS Scene Viewer ────────────────────────────────────────────────────
    st.subheader("3DGS Scene Viewer")
    supersplat_url = os.getenv("SUPERSPLAT_SERVER_URL", "http://localhost:8090")
    static_url = os.getenv("STATIC_SERVER_URL", "http://localhost:8080")

    try:
        missions_resp = requests.get(
            f"{API_URL}/admin/missions", headers=_HEADERS, timeout=5
        )
        missions_list = missions_resp.json() if missions_resp.ok else []
    except requests.exceptions.RequestException:
        missions_list = []

    done_missions = [m for m in missions_list if m.get("splat_paths")]
    if not done_missions:
        st.caption("No missions with 3DGS maps yet.")
    else:
        mission_options = {
            f"{m['id']} ({m.get('scene_count', 1)} scene(s))": m
            for m in done_missions
        }
        selected_label = st.selectbox(
            "Mission", list(mission_options.keys()), key="viewer_mission"
        )
        selected_mission = mission_options[selected_label]
        splat_paths = selected_mission.get("splat_paths", [])

        if len(splat_paths) > 1:
            scene_labels = [os.path.basename(os.path.dirname(p)) for p in splat_paths]
            chosen_label = st.selectbox(
                "Scene", scene_labels, key="viewer_scene"
            )
            scene_idx = scene_labels.index(chosen_label)
            chosen_splat = splat_paths[scene_idx]
        else:
            chosen_splat = splat_paths[0]

        # Build static URL for the splat.ply (served by nginx at /static/maps/)
        mission_id = selected_mission["id"]
        rel_path = os.path.relpath(
            chosen_splat,
            os.environ.get("MAPS_DIR", "data/maps"),
        )
        splat_static_url = f"{static_url}/static/maps/{rel_path}"
        viewer_url = f"{supersplat_url}/?load={splat_static_url}"

        st.caption(f"splat.ply: `{chosen_splat}`")
        st.components.v1.iframe(viewer_url, height=600, scrolling=False)

with tab_text:
    st.header("Text Query")
    text = st.text_input("Query", value="green field", max_chars=1000)
    search_type_text = st.selectbox("Search type", ["both", "frame", "tile"], index=0, key="text_type")
    enable_rerank_text = st.checkbox("Enable rerank", value=True, key="text_rerank")
    top_k_text = st.slider("Top-K", min_value=5, max_value=50, value=20, key="text_topk")
    if st.button("Search", key="text_search"):
        payload = {"text": text}
        params = {
            "top_k": top_k_text,
            "search_type": search_type_text,
            "enable_rerank": enable_rerank_text,
        }
        resp = requests.post(f"{API_URL}/query/text", json=payload, params=params, headers=_HEADERS)
        if resp.ok:
            results = resp.json().get("results", [])
            _render_results(results)
        else:
            st.error(resp.text)
