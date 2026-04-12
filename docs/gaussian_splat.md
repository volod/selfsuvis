# 3D Gaussian Splat map (Step 27)

Step 27 builds a 3D Gaussian Splat from the SfM point cloud produced by pycolmap
(Step 26), using [gsplat](https://github.com/nerfstudio-project/gsplat).

## Initialisation modes

Two modes are selected automatically based on whether pycolmap produced usable poses:

| Mode | When used | Quality |
|---|---|---|
| `gsplat_sfm` | pycolmap installed + ≥3 SfM poses recovered | Best |
| `gsplat_free` | pycolmap unavailable or SfM failed | Good |

In `gsplat_free` mode the initial point cloud is derived from a PCA decomposition
of the CLIP embeddings — no depth sensor required.

## Prerequisites

`gsplat` is included in `requirements_prod.txt` and installed by `make venv`.
Verify the install:

```bash
python -c "from gsplat.rendering import rasterization; print('gsplat OK')"
```

GPU with CUDA is strongly recommended. CPU-only training is supported but slow
(~10× longer per iteration).

## Outputs

All outputs are written to `<output-dir>/<video-name>/3d_map/`:

| File | Contents |
|---|---|
| `gaussian_splat.ply` | Standard 3DGS PLY — open in SuperSplat, Luma AI, or any 3DGS viewer |
| `view_splat.html` | Standalone browser viewer (requires a local HTTP server) |
| `sparse_map.ply` | Sparse point cloud (SfM keypoints or PCA fallback) |
| `semantic_environment_graph.json` | YOLO-based scene graph with object labels and 3D positions |

## Viewing the splat

**Option A — drag-and-drop (no server needed):**

Open https://playcanvas.com/supersplat/editor and drag `gaussian_splat.ply` into
the browser window.

**Option B — built-in HTML viewer:**

```bash
cd <output-dir>/<video-name>/3d_map/
python -m http.server 8765
# Open http://localhost:8765/view_splat.html
```

Controls: left-drag to orbit · right-drag to pan · scroll to zoom

**Option C — point cloud only (no gsplat required):**

```bash
python main.py --mode local --view-npz <output-dir>/<video-name>/3d_map/sparse_map.npz
```

This opens a matplotlib 3D scatter plot of the sparse SfM/PCA point cloud.
Useful for quick inspection on machines without a GPU.

## Skipping Step 27

```bash
python main.py --mode local --input <video.mp4> --no-gsplat
```

## Training parameters

The gsplat training loop runs for 3 000 iterations by default.
Key env vars (set in `.env` or before the run command):

| Variable | Default | Effect |
|---|---|---|
| `GSPLAT_ITERATIONS` | `3000` | Training iterations — increase for higher quality |
| `GSPLAT_DENSIFY_UNTIL` | `1500` | Stop densifying Gaussians after this iteration |
| `GSPLAT_LR` | `1e-3` | Learning rate for Gaussian parameters |
| `GSPLAT_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |

## Integration with the sensor fusion pipeline

The 3D Gaussian Splat is used downstream by:

- **Step 28 (fine-tuning)** — Gaussian position features are concatenated with
  CLIP embeddings to improve frame-level retrieval with spatial constraints.
- **Robot pose API** (`POST /query/pose`) — the sparse point cloud provides
  prior map geometry for GPS-anchored nearest-neighbour lookups.
- **Change detection** — cross-mission splat comparison detects structural changes
  in the environment (new obstacles, removed landmarks).
