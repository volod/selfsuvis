# Agentic Knowledge Flow

The local pipeline is not just a bag of isolated model calls.
Later steps reuse earlier evidence through an accumulated context object.

## Core Idea

Earlier steps produce structured evidence.
That evidence is stored and reused by later steps so they do not reason from raw pixels alone.

The main accumulator is implemented in:

- [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py)

## What Gets Accumulated

Examples of evidence added over time:

- Gemma scene context
- Florence captions
- ASR transcripts
- OCR text
- depth summaries
- detections
- rolling Qwen state

## Why This Matters

Without this pattern:

- every model reasons in isolation
- cross-modal contradictions are hidden
- later steps repeat work
- the final report is harder to audit

With this pattern:

- captions can use domain hints
- Qwen can reason across speech, text, depth, and objects
- the audit stage can trace context reuse

## Human Reading Strategy

If you want to understand the “agentic” part of the pipeline, inspect these in order:

1. [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py)
2. [`pipeline/workflows/local/steps_caption.py`](../../pipeline/workflows/local/steps_caption.py)
3. [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)
4. [`docs/pipeline.md`](../pipeline.md)

## Questions To Ask While Reading

- What evidence is added here?
- What later step consumes it?
- Is the evidence time-aligned or video-global?
- Can stale or wrong context propagate from this point onward?

## Related Docs

- [Perception core](02_perception_core_steps_01_08.md)
- [Sensors and fusion](03_sensor_steps_09_20.md)
- [Tracking and mapping](04_tracking_mapping_steps_21_27.md)
- [Adaptation and audit](05_adaptation_eval_steps_28_35.md)
