# UI

## Index form (API)
Simple HTML form at `http://localhost:8000/index/form` to upload a local video file or submit a URL for indexing. Optional API key field when `API_KEY` is set.

## Index Video (Streamlit)
- Upload a video, or provide a URL or local path
- Watch job status updates

## Query
- Text: enter query (e.g., "green field")
- Image: upload a reference image

Each result includes an `mpv` command you can run locally:
```bash
mpv "./data/videos/<video_id>.mp4" --start=<t_sec>
```

---
[← API](api.md) | [Helpers →](helpers.md)
