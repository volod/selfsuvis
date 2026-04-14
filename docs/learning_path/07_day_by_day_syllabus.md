# Day-By-Day Syllabus

This is a realistic 21-day study plan for a human who wants to understand the local pipeline deeply without trying to learn everything in one sitting.

## Week 1: Build The Base Mental Model

### Day 1

Topics:
- read `README.md`
- read [`local_path.md`](../../local_path.md)
- inspect one output directory from a previous run

Outcome:
- know the major artifacts and the end-to-end shape of the pipeline

### Day 2

Topics:
- Step 1
- `ffmpeg`
- video sampling

Exercise:
- compare one short clip extracted at multiple FPS values

### Day 3

Topics:
- Step 2
- embeddings
- cosine similarity
- vector search

Exercise:
- inspect retrieval neighbors for a few frames and note obvious failures

### Day 4

Topics:
- Step 3
- Gemma multimodal analysis
- scene changes and retrieval probes

Exercise:
- inspect `gemma_analysis.md` and summarize what the model thinks the mission contains

### Day 5

Topics:
- Step 4
- image captioning
- prompt steering

Exercise:
- compare Florence captions against your own captions for ten frames

### Day 6

Topics:
- Steps 5-6
- ASR and OCR
- text as evidence

Exercise:
- find a case where speech or visible text changes the interpretation of a frame

### Day 7

Topics:
- Steps 7-8
- depth and detection

Exercise:
- inspect where depth is useful and where it is misleading

## Week 2: Learn The Sensor Expansion

### Day 8

Topics:
- Step 9
- IQ basics
- spectral features

Exercise:
- inspect RF-related fields and explain them in plain language

### Day 9

Topics:
- Steps 10-12
- thermal
- multispectral
- event cameras

Outcome:
- understand why these are not “just more cameras”

### Day 10

Topics:
- Steps 13-15
- LiDAR
- radar
- GNSS-R

Exercise:
- write one paragraph on what each modality can reveal that RGB cannot

### Day 11

Topics:
- Steps 16-19
- inertial
- atmospheric
- gas/radiation
- acoustic

Exercise:
- explain which of these sensors would matter most in your target mission type

### Day 12

Topics:
- Step 20
- timestamp alignment
- fusion logic

Exercise:
- trace one fused observation from raw sidecar input to the final context block

## Week 3: Structure, Adaptation, And Audit

### Day 13

Topics:
- Steps 21-22
- segmentation
- tracking
- language-guided perception

Exercise:
- inspect `gemma_tracking_summary.md` and confirm whether tracking priorities make sense

### Day 14

Topics:
- Steps 23-25
- temporal embeddings
- Qwen
- UniDriveVLA

Exercise:
- compare `detailed_captions.md` and `unidrive_analysis.md`

### Day 15

Topics:
- Steps 26-27
- search tests
- 3D mapping
- Gaussian Splat

Exercise:
- identify one retrieval failure and one mapping limitation

### Day 16

Topics:
- Step 28
- SSL
- contrastive learning

Exercise:
- explain why mission-specific adaptation can help retrieval

### Day 17

Topics:
- Steps 29-30
- distillation
- export

Exercise:
- summarize what information the student model must preserve

### Day 18

Topics:
- Steps 31-33
- evaluation
- model agreement

Exercise:
- compare what improved after adaptation and what merely changed

### Day 19

Topics:
- Steps 34-35
- synthesis
- audit
- provenance

Exercise:
- read `agentic_flow.md` and note where wrong context could propagate

### Day 20

Topics:
- end-to-end review

Exercise:
- re-run one small local example and inspect outputs in order

### Day 21

Topics:
- consolidation

Exercise:
- write your own one-page explanation of the full local pipeline from memory
