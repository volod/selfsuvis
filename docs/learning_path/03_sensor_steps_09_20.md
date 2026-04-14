# Physical Sensors And Fusion: Steps 9-20

This phase expands perception beyond RGB video.
The goal is not to force every mission to use every sensor.
The goal is to understand how the pipeline can absorb side-channel evidence when it exists.

<a id="step-9-rf--sdr-sensing"></a>
## Step 9. RF / SDR sensing

What it does:
Derive radio-environment features from IQ captures or an audio proxy.

Why it matters:
The scene can be operationally important even when nothing visually changes.

Implementation:
- [`pipeline/vision/rf_analyzer.py`](../../pipeline/vision/rf_analyzer.py)
- [`scripts/prepare_sensor_data.sh`](../../scripts/prepare_sensor_data.sh)

Human focus:
- IQ basics
- SNR
- spectral flatness
- occupied bandwidth

<a id="step-10-thermal--infrared-imaging"></a>
## Step 10. Thermal / infrared imaging

What it does:
Add heat signatures beside visible-light imagery.

Why it matters:
Thermal cues reveal living agents, engines, and heat leaks that RGB can miss.

<a id="step-11-multispectral--hyperspectral-imaging"></a>
## Step 11. Multispectral / hyperspectral imaging

What it does:
Add extra bands for material and vegetation analysis.

Why it matters:
Materials that look similar in RGB often separate in spectral space.

<a id="step-12-event-camera-neuromorphic-sensing"></a>
## Step 12. Event camera (neuromorphic sensing)

What it does:
Represent brightness changes asynchronously.

Why it matters:
Event cameras are strong for fast motion and high dynamic range.

<a id="step-13-lidar--active-ranging"></a>
## Step 13. LiDAR / active ranging

What it does:
Add direct distance measurements and point geometry.

Why it matters:
LiDAR provides explicit structure rather than inferred monocular structure.

<a id="step-14-radar-fmcw--doppler--sar"></a>
## Step 14. Radar (FMCW / Doppler / SAR)

What it does:
Add range and motion sensing that can work through poor visibility.

Why it matters:
Radar complements both camera and LiDAR in harsh conditions.

<a id="step-15-gnss-r-and-satellite-signal-reception"></a>
## Step 15. GNSS-R and satellite signal reception

What it does:
Add reflected navigation-signal evidence and external traffic / environment feeds.

Why it matters:
This step broadens context beyond the camera platform itself.

<a id="step-16-inertial-and-barometric-sensing"></a>
## Step 16. Inertial and barometric sensing

What it does:
Add platform motion, orientation, and pressure-based altitude context.

Why it matters:
It anchors visual observations to how the platform actually moved.

<a id="step-17-atmospheric--environmental-sensing"></a>
## Step 17. Atmospheric / environmental sensing

What it does:
Add local weather and ambient conditions.

Why it matters:
Weather changes both mission conditions and sensor quality.

<a id="step-18-chemical--gas--radiation-sensing"></a>
## Step 18. Chemical / gas / radiation sensing

What it does:
Add non-visual hazard measurements.

Why it matters:
These are often the highest-value “invisible” signals in a mission log.

<a id="step-19-acoustic-sensing"></a>
## Step 19. Acoustic sensing

What it does:
Add sound evidence from speech, vehicles, impacts, machinery, wildlife, or ambience.

Why it matters:
Sound often confirms or contradicts what the camera seems to show.

<a id="step-20-sensor-fusion-analysis"></a>
## Step 20. Sensor fusion analysis

What it does:
Time-align and merge sensor evidence into a unified context record.

Why it matters:
Independent sensors are useful.
Aligned sensors are much more useful.

Implementation:
- [`docs/pipeline.md`](../pipeline.md)
- [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py)

Human focus:
- timestamp alignment
- lag tolerance
- missing-data handling
- contradictions between modalities

## What A Human Should Learn In This Phase

Do not try to become a specialist in every sensor first.
The practical order is:

1. Learn what each sensor is good at.
2. Learn what each sensor cannot tell you.
3. Learn how the pipeline stores each sensor’s evidence.
4. Learn how fusion resolves or exposes disagreement.

## Related Reading

- [Pipeline architecture](../pipeline.md)
- [Agentic knowledge flow](06_agentic_knowledge_flow.md)
- [Day-by-day syllabus](07_day_by_day_syllabus.md)
