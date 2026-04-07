# Gemma API Operations Runbook

> Covers: Ollama health checks, model restart, quality validation, Florence fallback
> activation, and exponential backoff guidance for the Gemma captioning sidecar.

---

## 1. Architecture overview

```
indexer.py
  └─ _run_gemma_caption_pass()          ← production captioner (GEMMA_API_URL set)
       ├─ POST /v1/chat/completions      ← Ollama / vLLM sidecar
       │   chunk_size=3, timeout=50s, retries=1
       └─ Florence fallback             ← on 2nd chunk failure, or frames beyond GEMMA_MAX_CAPTION_FRAMES
```

Environment variables (`.env`):

| Variable | Default | Description |
|---|---|---|
| `GEMMA_API_URL` | `""` | Set to enable Gemma: `http://localhost:11434/v1` (Ollama) or `http://localhost:8000/v1` (vLLM) |
| `GEMMA_API_MODEL` | `gemma4:e4b` | Model tag served by Ollama / vLLM |
| `GEMMA_API_TIMEOUT_SEC` | `60` | Per-request timeout for generative analysis |
| `GEMMA_MAX_CAPTION_FRAMES` | `200` | Max frames per mission captioned by Gemma; rest fall back to Florence |
| `GEMMA_CAPTION_CHUNK_SIZE` | `3` | Frames per async chunk |
| `GEMMA_CAPTION_RETRIES` | `1` | Retry count before Florence fallback |

---

## 2. Health check

```bash
# Confirm Ollama is serving and the target model is loaded
curl -s http://localhost:11434/api/tags | jq '.models[].name'

# Send a minimal completions ping (should return HTTP 200)
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e4b","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
```

Expected: `200`. Any `4xx` or `5xx` indicates the model is not loaded — see §3.

For **vLLM**:

```bash
curl -s http://localhost:8000/health
# → {"status":"ok"}
```

---

## 3. Model restart

### Ollama

```bash
# Pull the model if not already present
ollama pull gemma4:e4b

# Restart Ollama service (systemd)
sudo systemctl restart ollama

# Or if running in Docker
docker restart ollama
```

Verify the model loads by repeating the health check in §2.

### vLLM

```bash
# Restart the vLLM container (adjust service name as needed)
docker restart vllm

# Check logs for model load confirmation
docker logs --tail=50 vllm | grep "Model loaded"
```

---

## 4. Quality validation procedure

Validate the Gemma endpoint with a representative frame sample from a recent mission:

```bash
curl -s -H "Content-Type: application/json" \
  -d '{"text":"describe vehicles and road condition"}' \
  http://localhost:8000/query/text?top_k=5&search_type=frame
```

Use API logs and sampled outputs to verify response quality, latency, and schema stability before switching production traffic.

If gate fails:
1. Try a larger quantization level: `ollama pull gemma4:12b` and re-run validation.
2. If still failing, keep Qwen: unset `GEMMA_API_URL` in `.env` — indexer automatically
   falls back to Florence for all captions.

---

## 5. Florence fallback activation

To disable Gemma captioning entirely and revert to Florence for all frames:

```bash
# In .env
GEMMA_API_URL=
```

Then restart the worker:

```bash
docker restart worker
# or if running locally:
kill -HUP $(pgrep -f "python worker/main.py")
```

The indexer logs confirm which captioner is active at startup:

```
INFO  Gemma captioning pass: 200 frames via Gemma API, 47 via Florence fallback
# vs.
INFO  Florence captioning pass: 247 frames (batch_size=16)
```

---

## 6. Exponential backoff guidance

The current chunk retry (1 retry, 50s timeout) is appropriate for <10% timeout rates.

**If timeout rate exceeds 10%** based on API logs or user-facing request timings:

1. **Reduce `GEMMA_CAPTION_CHUNK_SIZE`** from 3 → 1 to serialize image encoding load.
2. **Increase `GEMMA_API_TIMEOUT_SEC`** to 90 or 120.
3. **Implement exponential backoff** in the worker by setting a longer retry delay:

```python
# Manual workaround: in pipeline/indexer.py _caption_chunk_with_retry,
# add a sleep before the retry:
import time
time.sleep(2 ** attempt)  # 1s, 2s, 4s ...
```

A first-class `GEMMA_CAPTION_BACKOFF_BASE_SEC` config var and retry loop is tracked
as a future enhancement — implement if the timeout rate stays above 10% for >1 week.

4. **Scale Ollama parallelism**: start Ollama with `--parallel 4` to enable concurrent
   batch processing (default is 1 = serial). Requires sufficient VRAM for N concurrent
   KV-cache slots.

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

---

## 7. VRAM budget (RTX 4060 Ti / 16 GiB)

| Component | VRAM |
|---|---|
| CLIP ViT-B-16 | ~470 MiB |
| DINOv3 ViT-B/14 | ~245 MiB |
| Florence-2-large (when active) | ~2.7 GiB |
| Gemma4:e4b Q4 (Ollama) | ~3.0 GiB |
| **Total (Gemma + CLIP + DINO)** | **~3.7 GiB** |

Gemma captioning is sequential with Florence (not concurrent) — the worker offloads
CLIP/DINO before running Florence, and Gemma runs as a sidecar HTTP call (no direct
VRAM allocation from the worker process).

---

## 8. Log patterns to watch

| Pattern | Meaning |
|---|---|
| `Gemma caption chunk failed after 2 attempt(s)` | Chunk fell back to Florence; check Ollama health |
| `Florence fallback batch failed` | Both Gemma and Florence are unhealthy for this chunk |
| `caption_model: gemma-api:gemma4:e4b` | Frame captioned by Gemma |
| `caption_model: florence-2-large:v1:fp16` | Frame captioned by Florence |
| `caption_confidence: 0.5` | Caption failed or used empty fallback |
