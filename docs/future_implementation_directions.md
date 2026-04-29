# Advanced Directions: Global Threats, Sensor Meshes, and Cross-Modal World Models

This document is the forward-looking companion to the current local pipeline.
It does not repeat the local threat-stack mechanics that are already explained
in the dedicated deep dives:

- [Temporal SSL and physical state](learning_path/14_temporal_ssl_physical_state.md)
- [Threat primitives and local inference](learning_path/15_threat_primitives_local_inference.md)

Read those two documents for track-aware SSL, local physical-state summaries,
structured threat primitives, local threat scoring, and operator-facing threat
provenance. This file starts from the next question:

If you want implementation-oriented work items rather than conceptual
directions, read [future_implementation_todos.md](future_implementation_todos.md).

> after a single-video local threat stack exists, what are the hardest and most
> valuable directions that still remain?

Use this document for four questions:

1. Which advanced topics extend the system instead of only rephrasing local outputs?
2. What new mathematical and systems problems appear when threat reasoning expands beyond one video?
3. Which research directions are technically credible for `selfsuvis` rather than fashionable but premature?
4. In what order should a human study the remaining work?

## 1. Remaining Direction Map

The remaining work is best understood as six coupled directions:

1. **Cross-modal and geometry-aware representation learning**
2. **Environmental field models beyond discrete objects**
3. **Global threat inference over sectors, routes, and multiple nodes**
4. **Trust-aware contradiction modeling**
5. **Realtime sensor-mesh runtime discipline**
6. **Decision, calibration, and evaluation layers**

That order is deliberate. The easiest mistake is to jump from a local threat
score straight to planners, dashboards, or reinforcement learning. That usually
creates a larger system with the same evidential weaknesses. The more defensible
path is:

- strengthen representations first
- expand the physical world model second
- aggregate local evidence into global state third
- only then add policy logic above calibrated uncertainty

## 2. Cross-Modal and Geometry-Aware Representation Learning

### 2.1 Cross-modal SSL

The current adaptation story is still dominated by visual agreement across time.
The next real gain is to make representations sensitive to **agreement and
disagreement across sensing modalities**.

Examples:

- RGB <-> depth consistency
- camera motion <-> IMU dynamics
- tracks <-> radar radial velocity
- occupancy estimates <-> thermal anomalies
- semantic captions <-> acoustic or RF evidence

This matters because a threat system should not only recognize repeated visual
appearance. It should also learn when the joint sensor configuration is
physically plausible. A vehicle approaching the platform should have:

- temporally coherent image motion
- depth change
- track persistence
- velocity consistent with pose change

If one of those channels disagrees, the latent space should express that
tension instead of hiding it downstream in text.

Advanced topics worth studying here:

- cross-modal contrastive learning with weak synchronization
- masked sensor modeling for missing-sidecar robustness
- teacher-student transfer across heterogeneous modalities
- correspondence learning when sensor rates and fields of view differ
- uncertainty-aware pretraining where the target includes confidence, not only class or identity

The real design challenge is not only model architecture. It is defining which
cross-modal alignments are physically meaningful and which are accidental
co-occurrences.

### 2.2 Geometry-aware SSL

The pipeline already uses depth, tracking, and SfM later in the stack, but those
signals still deserve a larger role during representation learning.

Advanced direction:

- make multiview overlap a supervision signal
- make pose consistency part of the embedding objective
- make long-track identity depend on geometric continuity, not only appearance
- make free-space structure and occlusion ordering visible in the latent space

This pushes the backbone away from "good at matching images" toward
"good at preserving scene structure." That distinction matters for threat work,
because many critical errors come from geometric confusion:

- a near obstacle treated as far
- an occluded object treated as absent
- a pose failure treated as stable motion
- a drifted track treated as persistence

Recommended study topics:

- multiview consistency losses
- pose-conditioned embedding spaces
- equivariant and geometry-preserving representations
- occupancy-aware latents
- differentiable rendering and bundle-adjustment-style supervision

## 3. Environmental Field Models

The current physical abstractions are strongest around tracked objects, local
occupancy, and pose confidence. The next physical extension is to model hazards
that behave like **fields**, not entities.

Examples:

- smoke, dust, fog, and rain density
- RF interference intensity
- thermal plumes or hotspots
- gas, chemical, or radiation concentration
- turbulence, wind shear, or rotor wash structure

Why field models are different:

- they spread continuously instead of moving as a single object
- they often change the reliability of other sensors before becoming visible hazards themselves
- their risk depends on gradients, transport, and persistence, not only presence

This creates a different inference problem. Instead of asking "what object is
there?" the system asks:

- where is hazard intensity increasing?
- which region is likely to become unsafe next?
- how quickly is the field moving toward the platform or route?
- which sensors are currently degraded because of the field?

Advanced topics worth deeper study:

- advection-diffusion modeling for plume-like hazards
- Gaussian process field estimation when measurements are sparse
- occupancy flow fields and dynamic traversability surfaces
- weather-conditioned sensor reliability models
- coupling field estimates with route planning and sensor placement

The deeper lesson is that a strong threat stack cannot remain object-centric
forever. Many operational risks are environmental and only indirectly visible
through their effect on sensor quality and navigability.

## 4. Global Threat Inference

This is the most strategically important remaining direction.

Local threat inference answers:

- what is risky in this clip?
- what should this platform do right now?

Global threat inference answers harder questions:

- which sectors are degrading over time?
- which routes are repeatedly unsafe?
- are multiple platforms observing the same evolving hazard?
- does a local anomaly belong to a larger corridor-level pattern?

That requires new state, not just a bigger report. Plausible outputs include:

- `global_threat_map`
- `sector_risk_levels`
- `persistent_anomalies`
- `route_advisories`
- `threat_corridor_graph`
- `multi_node_disagreement_zones`

The advanced problem here is aggregation under uncertainty. Local primitives are
not independent. If three nearby platforms report visibility degradation during
the same interval, the system should not simply add three scores. It needs to
reason about:

- shared environmental cause vs duplicated evidence
- sector identity and spatial overlap
- temporal lag between nodes
- sensor health and freshness
- persistence across missions rather than within one clip only

This pushes the system toward spatiotemporal inference over a graph:

- nodes = platforms, sectors, routes, or map tiles
- edges = spatial adjacency, route continuity, or evidence correlation
- time = rolling windows with decay and recurrence

Recommended study topics:

- dynamic Bayesian networks
- factor graphs for spatiotemporal state
- graph neural networks for sector-level aggregation
- map-tile risk accumulation with temporal decay
- route risk estimation under missing and stale observations

## 5. Threat Memory and Contradiction Modeling

### 5.1 Cross-mission threat memory

A clip-bounded threat score is useful, but serious deployments need memory.
Without persistence across runs, the system cannot answer:

- does this route always degrade pose quality?
- does a specific sector repeatedly show RF anomalies?
- are certain conditions reproducibly associated with sensor disagreement?

Cross-mission memory requires more than storing old JSON files. It needs:

- stable sector or tile identifiers
- recurrence models with time decay
- mission metadata indexing for weather, platform, and payload configuration
- aggregation rules that distinguish repeat hazards from repeat sensor failures

The advanced topic here is **episodic evidence compression**: how to preserve
the operationally important part of many runs without storing every frame-level
detail forever.

### 5.2 Contradiction-aware scoring

Displaying disagreement is useful, but the harder step is to make disagreement
part of the inference state itself.

Examples:

- geometry says occupancy is dense, semantics say route is clear
- UniDrive planning looks normal, pose confidence collapses
- the tracker says an object persists, IoU continuity says identity broke
- captions remain confident while depth confidence drops

Advanced contradiction handling means:

- repeated disagreement reduces automation trust
- subsystem-specific contradiction patterns become learned reliability features
- disagreement can trigger sensor inspection, fallback policy, or model reweighting
- contradiction history itself becomes a mission signal, not only a UI note

This is where calibration, provenance, and world modeling meet. A mature system
should eventually distinguish:

- "two sources disagree because the scene is genuinely ambiguous"
- "two sources disagree because one subsystem is drifting"
- "two sources disagree because they measure different physical aspects of the scene"

## 6. Realtime Sensor-Mesh Runtime

The move from a local runner to a sensor mesh is fundamentally a systems change.
Most failures at this stage will be freshness, latency, and degraded-mode issues
before they are modeling issues.

### 6.1 Freshness, latency, and identity

A realtime mesh needs explicit treatment of:

- event time vs processing time
- per-sensor latency estimates
- freshness windows
- stale-data propagation rules
- identity of source node, sensor, and sector

The operational rule is simple:

> stale evidence must never silently behave like current evidence.

That implies timestamps and confidence are not metadata afterthoughts. They are
part of the threat state.

### 6.2 Multi-node fusion

A sensor mesh is a distributed evidence graph. The core questions become:

- how should one platform inherit or discount another platform's threat estimate?
- when do overlapping measurements reinforce vs duplicate each other?
- when should a node trust its local geometry over a remote semantic report?

The advanced work here involves:

- node-health modeling
- bounded-latency aggregation
- consistency checks across heterogeneous payloads
- handoff of threat state when one node leaves and another enters a sector
- degraded-mode operation when parts of the mesh disappear

### 6.3 Backpressure and degraded-mode behavior

Threat systems often fail because queues grow, expensive models stall, or
sidecars disappear. A credible runtime must define:

- which updates are lossy and can be dropped
- which updates must be preserved
- how scores degrade when a high-value source goes missing
- how operators see freshness and model-health failures

Advanced systems topics to study:

- queueing and backpressure control
- stream processing with event-time semantics
- causal update loops
- bounded-latency inference contracts
- explicit degraded-mode design

## 7. Decision Layers Beyond The Fixed Action Vocabulary

The current local action vocabulary is intentionally small:

- `continue`
- `reduce_speed`
- `reroute`
- `abort`
- `inspect_sensor`

That is the right abstraction for local operator support. It is not yet a full
policy layer.

The next decision-layer directions are:

- route-feasibility scoring instead of only action labels
- action recommendation conditioned on mission objective and risk tolerance
- active sensor scheduling
- constrained planning under uncertainty
- receding-horizon planning over fused world state

The key discipline is sequencing. Planning and reinforcement learning are only
defensible after the state, threat, freshness, and calibration layers become
stable. Otherwise the policy learns to exploit modeling noise.

## 8. Calibration And Evaluation

This is the most underrated remaining direction.

A stronger threat system is not simply one that produces more alarms. It is one
whose scores, uncertainties, and disagreements are useful for a human or an
automated planner making time-sensitive decisions.

That requires:

- calibration of threat scores against real outcomes
- persistence-threshold tuning by mission domain
- disagreement-rate monitoring by subsystem pair
- operator trust studies for provenance display
- alert-fatigue analysis
- local-vs-global false-positive and false-negative analysis

The real evaluation question is:

> did the system improve the decision, not only the narrative?

This is where many otherwise impressive multimodal systems become operationally
weak. They can describe a situation but cannot justify threshold choices,
uncertainty behavior, or intervention cost.

## 9. Suggested Study Sequence

If you already understand the current runner, this is the highest-return order:

1. Re-read the fusion docs with uncertainty, freshness, and contradiction in mind.
2. Read [14_temporal_ssl_physical_state.md](learning_path/14_temporal_ssl_physical_state.md) and [15_threat_primitives_local_inference.md](learning_path/15_threat_primitives_local_inference.md) so the current local abstractions are clear.
3. Study cross-modal and geometry-aware representation learning.
4. Study occupancy, flow, and environmental field models.
5. Design one global threat use case for one mission domain only.
6. Add cross-mission memory and contradiction-aware scoring before touching planning.
7. Treat policy learning and RL as the final stage, not the first.

If you skip directly to dashboards or planners, you usually optimize noise.

## 10. Concrete Proposal For `selfsuvis`

The most credible next expansion path is:

1. **Cross-modal and geometry-aware SSL**
   Make sensor agreement and multiview consistency first-class training signals.
2. **Environmental field layer**
   Add RF, weather, plume, and visibility-field estimates beside object and occupancy state.
3. **Global threat layer**
   Aggregate local primitives and local threat outputs across sectors, time windows, and nodes.
4. **Trust-aware contradiction handling**
   Convert disagreement from a display feature into an explicit reliability signal.
5. **Realtime mesh runtime**
   Add freshness-aware, bounded-latency, degraded-mode streaming infrastructure.
6. **Calibration and policy layer**
   Tune thresholds, validate outcomes, and only then move toward stronger planning logic.

If only one remaining direction can be funded now, choose this:

- **global threat inference built on top of the local threat stack**

That is the shortest path from the current repo to a broader autonomy-support system.

## 11. Main Papers And Topics To Read Next

### Representation learning

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)
  [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- Oquab et al., "DINOv2" (2023)
  [https://arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
- Tong et al., "VideoMAE" (2022)
  [https://arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602)

### Physical and field modeling

- Thrun, Burgard, Fox, *Probabilistic Robotics* (2005)
- Barfoot, *State Estimation for Robotics* (2017)
- Elfes, "Using Occupancy Grids for Mobile Robot Perception and Navigation" (1989)
- O'Callaghan and Ramos, "Gaussian Process Occupancy Maps" (2012)
  [https://arxiv.org/abs/1204.1081](https://arxiv.org/abs/1204.1081)

### World models and decision-making

- Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2019)
  [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023)
  [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- Garcia and Fernandez, "A Comprehensive Survey on Safe Reinforcement Learning" (2015)
  [https://jmlr.org/papers/v16/garcia15a.html](https://jmlr.org/papers/v16/garcia15a.html)

### Realtime systems and active perception

- Bajcsy et al., "Revisiting Active Perception" (2018)
  [https://arxiv.org/abs/1807.02041](https://arxiv.org/abs/1807.02041)
- Read queueing, backpressure, and event-time processing material from distributed systems literature alongside the ML papers. At this stage, timing, freshness, and trust are likely to fail before model capacity does.

---
[← Runtime and study guide](learning_path/01_runtime_and_study_guide.md) | [Perception core →](learning_path/02_perception_core_steps_01_08.md)
