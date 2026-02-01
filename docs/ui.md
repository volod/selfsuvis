# UI

## Index Video
- Upload a video, or provide a URL or local path
- Watch job status updates

## Query
- Text: enter query (e.g., "green field")
- Image: upload a reference image

Each result includes an `mpv` command you can run locally:
```bash
mpv "./data/videos/<video_id>.mp4" --start=<t_sec>
```
