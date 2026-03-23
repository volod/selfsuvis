# Design System

Streamlit-based App UI. Calm surface hierarchy, utility language, minimal chrome.

## Color tokens (al_tag / status badges)

| Token | Hex | Usage |
|---|---|---|
| `--color-needs-annotation` | `#e53935` (red) | al_tag=needs_annotation badge |
| `--color-novel` | `#f9a825` (amber) | al_tag=novel badge |
| `--color-none` | `#43a047` (green) | al_tag=none badge |
| `--color-status-ok` | `#43a047` | pose_status/map_status=success |
| `--color-status-pending` | `#f9a825` | map_status=pending |
| `--color-status-error` | `#e53935` | pose_status/map_status=failed |
| `--color-status-skipped` | `#9e9e9e` (grey) | map_status=skipped |

## Al_tag badge spec

Use Streamlit `st.badge()` or inline `<span>` with background-color. **Never** use
`border-left` color strips — use chips/badges.

| al_tag | Label | Color |
|---|---|---|
| `needs_annotation` | `ANNOTATE` | red |
| `novel` | `NOVEL` | amber |
| `none` | _(hidden)_ | — |

Only show badge when al_tag ≠ none (no badge = normal frame).

## Status badge spec

| State | Symbol | Color |
|---|---|---|
| success | ✅ | green |
| pending | ⏳ | amber |
| failed | ❌ | red |
| skipped | — | grey |

## Result card anatomy

Reusable function `_result_card(frame)` in `ui/components/result_card.py`.
Used in: Search results, Mission frame grid, Annotation Queue, Change detection pairs.
Do NOT inline in `ui/app.py` — 4 call sites mandate a component file.

```
┌─────────────────────────────┐
│ [thumbnail 256×256]         │
├─────────────────────────────┤
│ caption text (truncated 80c)│
│ [ANNOTATE] [mission_id]     │  ← badges
│ score=0.87  2d ago          │
│ [🔍 Find more like this]    │
└─────────────────────────────┘
```

- Thumbnail: full frame served via `st.image(frame_path)` (Streamlit resizes to fit card width). No pre-generation needed.
- Caption: max 80 chars, ellipsis
- al_tag badge: shown only if ≠ none
- mission_id badge: clickable → mission detail page
- Score: displayed as 2dp float
- Timestamp: relative ("2d ago"), tooltip = absolute ISO datetime
- "Find more like this" button: always shown (uses DINO if available, CLIP fallback)

## Chart spec (Plotly)

### Timeline (x=timestamp, y=al_score)
- x: timestamps as datetime
- y: al_score (0–1)
- Color: al_tag (green/amber/red per color tokens above)
- Hover: frame_id, caption, al_tag, timestamp
- Click: expand frame inline below chart
- Y-axis label: "Uncertainty Score"
- Threshold line at AL_TAG_K/frame_count (top-K boundary)

### 3D Camera Path Scatter (pycolmap poses)
- x/y/z: ENU translation vectors (meters)
- Color: al_tag (green/amber/red)
- Size: al_score * 8 + 3 (min 3px, max 11px)
- Hover: frame_id, caption, timestamp
- Click: expand frame inline
- Axis labels: "East (m)", "North (m)", "Up (m)"
- Plotly engine: `scattergl` (WebGL) when frame_count > 500

## Typography

Use Streamlit default font (Inter). No custom fonts. Section headings use
`st.subheader()`. No decorative dividers.

## Navigation breadcrumbs

All pages below home show breadcrumb at top using `st.caption()`:
- Mission detail: `← Missions > [mission_id]`
- Frame expanded from queue: `← Annotation Queue > [frame_id]`

## Annotation queue export format (v1, CVAT-import compatible for v2)

Export ZIP contains: `manifest.json` + JPEGs named `{frame_id}.jpg`.

```json
{
  "export_version": "1",
  "exported_at": "ISO8601",
  "frames": [
    {
      "frame_id": "abc123",
      "mission_id": "mission_xyz",
      "frame_path": "frames/mission_xyz/frame_001.jpg",
      "caption": "rocky hillside with dense vegetation",
      "al_tag": "needs_annotation",
      "al_score": 0.72,
      "timestamp": "ISO8601",
      "gps_json": {"lat": 37.4, "lon": -122.1, "alt": 50.0}
    }
  ]
}
```

This format is the v2 CVAT import contract. Do not change field names without a version bump.

## Change detection \u2014 "Find more like this" placement

Both frame A (older) and frame B (newer) in a change pair get a [Find more like this] button.
User may want to find more of either the old state or the new state.

## Annotation queue (v1)

v1 supports tagging only — CVAT integration is v2. UI must set expectations:
- Show disabled `[Annotate in CVAT ↗]` button with tooltip "Coming in v2"
- Offer export: `[Export frames as ZIP]` (JPEGs + JSON manifest with frame_id,
  caption, al_tag, mission_id, timestamp) for external annotation tools
- Export format must be import-compatible with v2 CVAT integration (document format)

## Missions list

Table columns: mission_id (link) | indexed (relative time) | frames | SfM | Map | al_tag summary
Sort: most recent first (default). User can resort by clicking column headers.

## 3DGS viewer (SuperSplat iframe)

- Height: `max(600, viewport_height * 0.7)` px, full width
- map_status=pending: spinner + "Generating 3D map... (~10 min)" + [Refresh] button +
  "While you wait: browse frames above" nudge. Auto-polls via `@st.fragment` every 30s
  (requires Streamlit ≥ 1.37 — upgrade Dockerfile.ui from 1.31.1). Isolates poll to
  the 3DGS section only; prevents full-page rerun flicker while frame grid is loaded.
- map_status=failed: red banner with reason + download link for sparse point cloud
- map_status=skipped (SfM failed): grey info box "3D viewer unavailable (SfM failed)"
- map_status=success: `st.components.v1.iframe(...)` — full width, height as above
- Offer "Open in new tab" link for users wanting full-screen SuperSplat

## Change detection viewer

- Shown only if `change_detections` count > 0 for this mission (hidden section otherwise)
- Frame pairs sorted by `change_score DESC` (highest confidence first)
- Layout: two columns side-by-side, equal size
- Below each frame: caption, mission_id badge, timestamp, al_tag badge
- Paginate: 5 pairs per page
- Filters: change_score threshold slider (default 0.35), mission date range

## Responsive design (full mobile support)

| Viewport | Layout changes |
|---|---|
| Mobile (<768px) | Frame grid: 1 col (not 4). Sidebar nav collapses. Charts: hide 3D scatter (touch-hostile), show timeline only. 3DGS iframe: hide (show "Open in new tab" link). Result card: full width. |
| Tablet (768–1024px) | Frame grid: 2 cols. Charts render at reduced height. 3DGS iframe: show at 400px height. |
| Desktop (>1024px) | Full layout as specified. |

Streamlit uses `st.columns()` — column count must be conditioned on `st.session_state._viewport` (detect via JS + `st.components.v1.html`).

Touch targets: all buttons ≥ 44px height. [Find more like this] button must not be too small to tap.

## Accessibility

- All images: `alt` text = caption text (or "[frame, no caption]" if NULL)
- ARIA landmarks: each `st.container()` section labeled (via custom HTML)
- Keyboard navigation: Streamlit's native tab order is sufficient for buttons/links
- Color contrast: all badge text must meet WCAG AA (4.5:1 against white background)
  - Red badge text: white (#fff) on #e53935 — check contrast; use #c62828 if below threshold
  - Amber badge text: black (#000) on #f9a825 — passes AA
- Error messages: never rely on color alone — include icon + text (e.g., "❌ SfM failed")
- Plotly charts: not fully accessible (screen reader limitation), add text summary below each chart: "Timeline shows 487 frames. Top 24 flagged needs_annotation, 12 novel, 451 none."

## Search result context

Text/image search results must show:
- mission_id badge (clickable)
- al_tag badge (if ≠ none)
- Caption (why it matched)
- Score (2dp)
- Relative timestamp
- search_type shown: "frame" or "tile" (so user understands granularity)
