# Adaptation, Evaluation, And Audit: Steps 28-35

This phase asks the practical engineering questions:

- can the model adapt to mission-specific data?
- can it be compressed?
- did it actually improve?
- can a human audit the result?

<a id="step-28-ssl-dino-fine-tuning"></a>
## Step 28. SSL DINO fine-tuning

What it does:
Adapt the representation with self-supervised learning on mission frames.

Why it matters:
It narrows the gap between general internet pretraining and local mission data.

Implementation:
- [`pipeline/workflows/local/steps_ssl.py`](../../pipeline/workflows/local/steps_ssl.py)
- [`pipeline/training/ssl.py`](../../pipeline/training/ssl.py)

<a id="step-29-knowledge-distillation"></a>
## Step 29. Knowledge distillation

What it does:
Transfer useful structure from a stronger teacher into a smaller student.

Why it matters:
This is how the pipeline turns high-quality but heavy models into deployable ones.

Implementation:
- [`pipeline/workflows/local/steps_distill.py`](../../pipeline/workflows/local/steps_distill.py)
- [`pipeline/training/distill.py`](../../pipeline/training/distill.py)

<a id="step-30-onnx-export-and-gallery-build"></a>
## Step 30. ONNX export and gallery build

What it does:
Export the adapted model and build the edge inference gallery.

Why it matters:
A model that cannot be exported and served is still a research artifact.

Implementation:
- [`pipeline/workflows/local/steps_distill.py`](../../pipeline/workflows/local/steps_distill.py)

<a id="step-31-fine-tuned-search-test"></a>
## Step 31. Fine-tuned search test

What it does:
Re-run retrieval with the adapted representation.

Why it matters:
It checks whether training improved task utility rather than only loss values.

Implementation:
- [`pipeline/workflows/local/steps_embed.py`](../../pipeline/workflows/local/steps_embed.py)

<a id="step-32-model-comparison-and-video-description"></a>
## Step 32. Model comparison and video description

What it does:
Compare baseline and adapted retrieval, then derive a clip-level summary.

Why it matters:
It gives both a numerical and a readable summary of what changed.

Implementation:
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)
- [`pipeline/workflows/local/steps_report.py`](../../pipeline/workflows/local/steps_report.py)

<a id="step-33-multi-model-comparison"></a>
## Step 33. Multi-model comparison

What it does:
Compare outputs from major multimodal analyzers.

Why it matters:
Agreement is useful, but disagreement is often more informative.

Implementation:
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)

<a id="step-34-video-synthesis"></a>
## Step 34. Video synthesis

What it does:
Produce a final report from many intermediate artifacts.

Why it matters:
This is where the pipeline becomes a human-facing product.

Implementation:
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)

<a id="step-35-agentic-flow-audit"></a>
## Step 35. Agentic flow audit

What it does:
Write a provenance-style audit of how context moved through the pipeline.

Why it matters:
This is the inspection layer for debugging, trust, and failure analysis.

Implementation:
- [`pipeline/workflows/local/runner.py`](../../pipeline/workflows/local/runner.py)
- [`docs/pipeline.md`](../pipeline.md)

## What A Human Should Learn In This Phase

Learn to separate these questions:

- Did the representation get better?
- Did the smaller model preserve the right structure?
- Did outputs become more useful for humans?
- Can I trace how the final conclusion was formed?

That separation matters because a system can optimize one of those and fail the others.
