# Physical Sensors And Fusion: Steps 9-20

This phase expands perception beyond RGB video.
The goal is not to force every mission to use every sensor.
The goal is to understand how the pipeline can absorb side-channel evidence when it exists, and how to design for graceful degradation when it does not.

A practical rule: **treat each sensor as a hypothesis generator, not a ground truth source.**
The pipeline merges them in Step 20 precisely because no single sensor is always right.

---

<a id="step-9-rf--sdr-sensing"></a>
## Step 9. RF / SDR sensing

**What it does:**
Ingest IQ captures (raw radio-frequency samples) or an audio proxy, compute spectrograms and signal statistics, and derive radio-environment features: occupied bandwidth, SNR, spectral flatness, peak frequencies.

**Why it matters:**
The radio environment around a mission can be operationally significant even when nothing visually changes.
Sudden frequency occupation or jamming signatures can indicate activity that no camera sees.
For drone missions, detecting interference in control-link bands (2.4 GHz, 5.8 GHz) is a safety signal.

**Implementation:**
- [`pipeline/vision/rf_analyzer.py`](../../pipeline/vision/rf_analyzer.py)
- [`scripts/prepare_sensor_data.sh`](../../scripts/prepare_sensor_data.sh)

**Key concepts:**

*IQ data:*
An SDR captures signal as complex samples: I (in-phase) and Q (quadrature) components.
The combination gives the full amplitude and phase of the signal, not just its magnitude.
Without IQ you can only measure signal strength; with IQ you can measure modulation, bandwidth, and Doppler shift.

*Spectrogram:*
Computed by sliding a short-time Fourier transform (STFT) over the IQ stream.
Result: a 2D array of time vs frequency vs power.
The pipeline extracts summary statistics from this rather than storing the full spectrogram.

*Spectral flatness:*
A measure of how "noise-like" a signal is.
White noise has spectral flatness near 1. A pure tone has flatness near 0.
Jamming signals tend to be broadband (flatness near 1) while communications signals are narrowband (flatness near 0).

*SNR:*
Signal-to-noise ratio in dB. Low SNR means the signal is buried in noise; estimates are unreliable below ~10 dB.

**Output artifact:**
RF feature JSON per time window: `{center_freq, bandwidth, snr_db, flatness, peak_power}` plus a time-aligned map to `t_sec`.

**Human focus:**
- Understand IQ basics: I and Q are the real and imaginary parts of a complex baseband signal.
- Learn to read a spectrogram: x-axis = time, y-axis = frequency, color = power in dB.
- Know the difference between occupied bandwidth (how wide is the emission?) and SNR (how clean is the measurement?).
- Understand spectral flatness as a modulation indicator.
- Learn where this step is likely absent: most consumer drone videos have no IQ sidecar.

**Common failure modes:**
- No IQ capture in mission → step is skipped; RF context is empty.
- Sample rate mismatch between IQ capture and expected SDR rate → frequency axis is scaled wrong.
- Very short capture window → spectral resolution is poor (frequency bins are wide).
- Wideband interference saturates receiver → all SNR estimates are unreliable.

---

<a id="step-10-thermal--infrared-imaging"></a>
## Step 10. Thermal / infrared imaging

**What it does:**
Ingest thermal (LWIR, ~8-14 µm wavelength) or near-infrared frames alongside visible-light frames.
Align them spatially to the RGB camera. Derive heat signature features: hot spots, thermal gradients, temperature zones.

**Why it matters:**
Thermal radiation reveals what RGB cannot:
living agents behind foliage or at night, engine heat in parked vehicles, heat leaks in buildings, and body temperature of personnel.
A vehicle that looks inactive in visible light may show a running engine in thermal.

**Implementation:**
- [`pipeline/vision/factory.py`](../../pipeline/vision/factory.py) — sensor adapter registration
- Thermal frames are processed as grayscale imagery with specialized normalization
- Sidecar files: `.seq` (FLIR), `.tiff` 16-bit, or proprietary drone thermal stream

**Key concepts:**

*Wavelength bands:*
- Near-infrared (NIR, 0.7-1.4 µm): reflected solar radiation, sensitive to vegetation (NIR is highly reflective for healthy leaves). Not temperature measurement.
- Short-wave IR (SWIR, 1.4-3 µm): sensitive to moisture and glass penetration.
- Mid-wave IR (MWIR, 3-5 µm): good for detecting hot objects (engines, fires) against cool backgrounds.
- Long-wave IR (LWIR, 8-14 µm): thermal radiation from room-temperature objects; what most drone thermal cameras use.

*Radiometric vs non-radiometric thermal:*
Radiometric cameras return absolute temperature per pixel (in Kelvin or Celsius).
Non-radiometric cameras return relative IR intensity. Most affordable drone thermal cameras are non-radiometric.
The pipeline normalizes both to [0, 1] intensity range.

*Spatial alignment (registration):*
Thermal and RGB cameras have different fields of view and lens centers.
Alignment requires either factory calibration parameters or homography estimation.
Misalignment is common and causes thermal overlay errors.

**Output artifact:**
Per-frame thermal feature JSON: `{hot_spot_count, max_temp_relative, mean_temp_relative, thermal_zones}`.
Overlay image: thermal colormap (e.g., "inferno" or "rainbow") saved beside the RGB frame.

**Human focus:**
- Understand emissivity: different materials emit different fractions of their true thermal radiation. Metal is reflective and appears cold even when hot.
- Learn the difference between detecting temperature and detecting heat contrast.
- Understand when thermal is most useful: at night, in smoke, or against foliage.
- Learn where thermal fails: reflective surfaces, cold environments where all objects are near ambient, and very hot scenes where contrast collapses.

**Common failure modes:**
- Thermal camera not present → step skipped; no thermal context.
- Non-radiometric camera with automatic gain control → intensity values shift between frames, making temporal comparison unreliable.
- Spatial misregistration → thermal hot spot is attributed to wrong RGB pixel region.
- High ambient temperature (desert noon) → temperature contrast between objects collapses; thermal advantage disappears.

---

<a id="step-11-multispectral--hyperspectral-imaging"></a>
## Step 11. Multispectral / hyperspectral imaging

**What it does:**
Ingest imagery with more than three spectral bands (e.g., Red, Green, Blue, NIR, RedEdge for multispectral; hundreds of narrow bands for hyperspectral).
Compute vegetation indices (NDVI), material indices (NDWI, NDSI), and spectral signatures for each pixel.

**Why it matters:**
Materials that look identical in RGB often separate in spectral space.
Healthy vs stressed vegetation, different soil compositions, water vs non-water surfaces, certain minerals, and camouflage materials all have distinct spectral signatures.
This is widely used in agriculture, forestry, geology, and military reconnaissance.

**Implementation:**
- Multi-band TIFF ingestion via `rasterio` or `GDAL`
- Index computation: NDVI = (NIR - Red) / (NIR + Red)
- Spectral signature extraction and comparison to a reference library

**Key concepts:**

*Spectral index:*
A ratio of two or more bands designed to highlight a specific material or condition.
- NDVI (Normalized Difference Vegetation Index): measures plant health using NIR and Red. Range -1 to +1; healthy vegetation > 0.4.
- NDWI (Normalized Difference Water Index): highlights water bodies.
- NDSI (Normalized Difference Snow Index): highlights snow/ice.

*Hyperspectral vs multispectral:*
Multispectral: 4-20 bands, typically pre-defined wavelengths. Lightweight cameras, common on drones.
Hyperspectral: 100-500 narrow contiguous bands. Large data volume, requires specialized processing (PCA, matched filter).
The pipeline currently targets multispectral; hyperspectral support is aspirational.

*Band registration:*
Multi-band cameras often have separate sensors per band with slight spatial offsets.
Band-to-band registration (alignment) must happen before index computation.
Errors produce "rainbow" artifacts around object edges.

**Output artifact:**
Per-frame spectral index map (GeoTIFF) and summary statistics: `{ndvi_mean, ndvi_std, water_fraction, stressed_veg_fraction}`.

**Human focus:**
- Learn what NDVI looks like and what a healthy vs stressed vegetation field looks like in it.
- Understand why spectral signatures can reveal camouflage: paint reflects differently than natural vegetation in NIR.
- Know the key limitation: multispectral cameras need calibration panels or atmospheric correction for cross-flight comparison.

**Common failure modes:**
- No multispectral payload → step skipped.
- Poor illumination or shadow → spectral ratios are distorted in shadowed pixels.
- Band registration errors → index maps have edge artifacts.
- Atmospheric scattering (high-altitude flights) → apparent reflectance values shift; NDVI underestimates vegetation.

---

<a id="step-12-event-camera-neuromorphic-sensing"></a>
## Step 12. Event camera (neuromorphic sensing)

**What it does:**
Ingest asynchronous event streams from a Dynamic Vision Sensor (DVS).
Convert events to a time-surface or event frame representation.
Extract motion speed, direction of movement, and temporal texture features.

**Why it matters:**
Standard cameras capture frames at a fixed rate regardless of scene activity.
Event cameras fire a pixel event only when that pixel's brightness changes, with microsecond resolution.
This makes them far better than frame cameras for:
- Very fast motion (propellers, projectiles, fast vehicles)
- High dynamic range scenes (bright sun + dark shadow in the same frame)
- Low latency detection (events arrive in real time, not at 30 fps)

**Implementation:**
- Event stream ingest from `.aedat`, `.raw`, or `.rosbag` formats
- Time-surface computation: each pixel stores the timestamp of its last event
- Event frame accumulation: sum events over a window for visualization

**Key concepts:**

*Event polarity:*
Each event carries a polarity: positive (brightness increased) or negative (brightness decreased).
Motion boundaries produce pairs of positive-then-negative events as the edge passes the pixel.
This gives edge detection for "free" without any image processing.

*Time surface:*
A 2D image where each pixel stores the timestamp of its most recent event.
Recent events appear bright; pixels with no recent events fade.
This is the natural representation for training event-based neural networks.

*No motion → no signal:*
A static scene produces almost no events (only noise).
This is a feature (efficient, no redundant data) and a limitation (the camera goes "blind" to stationary objects).

**Output artifact:**
Per-time-window event statistics: `{event_rate, dominant_motion_direction, mean_velocity_px_per_sec}`.
Accumulated event frames (optional): grayscale images for visualization.

**Human focus:**
- Understand why event cameras are fundamentally different from frame cameras (asynchronous, pixel-independent, microsecond timestamps).
- Learn when this sensor adds value vs when it is redundant (only useful for fast motion or HDR scenes).
- Know that training data for event cameras is still scarce; most pretrained models are not production-ready.

**Common failure modes:**
- No event camera in mission → step skipped.
- Very slow motion → event rate near zero; little useful signal.
- Extreme illumination change (camera pans toward sun) → massive event burst; event rate statistics become meaningless.
- Noise events (sensor defects) → "hot pixels" emit constant events and must be filtered.

---

<a id="step-13-lidar--active-ranging"></a>
## Step 13. LiDAR / active ranging

**What it does:**
Ingest a LiDAR point cloud (`.pcd`, `.las`, `.bag`, or per-frame `.bin`) aligned to the mission timestamp.
Compute per-frame range statistics: minimum range, maximum range, point density, ground plane height, and obstacle height profile.
Optionally perform ground plane extraction and object cluster detection.

**Why it matters:**
LiDAR provides explicit distance measurements rather than inferred monocular estimates.
It directly measures the 3D structure of the scene, enabling:
- Accurate obstacle detection at precise distances
- Ground plane extraction even on uneven terrain
- Height measurement of objects and terrain features
- Point cloud registration for map building

**Implementation:**
- Point cloud ingestion via `open3d` or `pyntcloud`
- Timestamp-synchronized to frame list via `t_sec` interpolation
- Ground segmentation via RANSAC plane fitting
- Cluster detection via DBSCAN or Euclidean clustering

**Key concepts:**

*Mechanical vs solid-state LiDAR:*
Mechanical (spinning) LiDAR (e.g., Velodyne VLP-16): 360° horizontal coverage, 16-128 scan lines, 10-20 Hz.
Solid-state LiDAR (e.g., Livox Avia): narrow field of view (~70°), no spinning parts, higher durability.
The pipeline accepts both but handles their different point cloud densities differently.

*Returns and intensity:*
Each LiDAR pulse may produce 0, 1, or multiple returns per ray (first return, last return, strongest return).
Multiple returns happen when a pulse passes through foliage and hits the ground below.
Return intensity indicates surface reflectivity.

*Point cloud alignment to RGB:*
To combine LiDAR structure with RGB texture, you need the extrinsic calibration (rotation + translation) between LiDAR and camera.
Without calibration, point clouds and images can only be used independently.

**Output artifact:**
Per-frame LiDAR statistics JSON: `{min_range_m, max_range_m, point_count, ground_height_m, cluster_count}`.
Optional: downsampled point cloud for 3D map use in Step 27.

**Human focus:**
- Understand the physical operation: a pulsed laser measures time-of-flight to compute range.
- Learn point cloud density vs range tradeoff: farther objects have sparser measurements.
- Understand why LiDAR beats monocular depth: it gives absolute metric distance, not relative estimates.
- Know the main failure mode: rain, fog, and dust scatter the laser, causing range errors or total signal loss.

**Common failure modes:**
- No LiDAR → step skipped; monocular depth from Step 7 is the only geometry.
- Time sync error between LiDAR and camera → point cloud appears offset in time; obstacle positions are wrong.
- Velodyne spinning at 10 Hz vs camera at 30 Hz → point cloud undersamples during fast movement.
- Retroreflective surfaces (road markings, safety vests) → overdriven returns with range errors.

---

<a id="step-14-radar-fmcw--doppler--sar"></a>
## Step 14. Radar (FMCW / Doppler / SAR)

**What it does:**
Ingest radar range-Doppler data from an FMCW (Frequency Modulated Continuous Wave) or pulse radar.
Extract range profiles, velocity estimates (Doppler), and optionally SAR (Synthetic Aperture Radar) imagery.

**Why it matters:**
Radar works in conditions where cameras and LiDAR fail: rain, fog, smoke, dust, and darkness.
FMCW radar provides simultaneous range and velocity measurements, making it ideal for tracking moving targets.
SAR produces high-resolution imagery independent of illumination, enabling day/night/all-weather mapping.

**Implementation:**
- Raw FMCW cube processing: fast time (range), slow time (Doppler), antenna (angle)
- Range-FFT + Doppler-FFT to produce a range-Doppler map
- CFAR (Constant False Alarm Rate) detection for target extraction
- Angle estimation from multiple receive antennas (MIMO)

**Key concepts:**

*FMCW principle:*
The radar transmits a signal whose frequency rises linearly (a chirp).
The reflected signal arrives at the receiver with a time delay proportional to range.
Mixing the transmitted and received chirp produces a beat frequency proportional to range.

*Doppler shift:*
A moving target shifts the reflected chirp frequency proportionally to its radial velocity.
Positive Doppler = target approaching. Negative = receding.
Doppler gives velocity "for free" without multiple frames.

*Range resolution:*
Range resolution = c / (2 × bandwidth). Higher bandwidth → finer resolution.
A 77 GHz automotive radar with 4 GHz bandwidth resolves targets ~4 cm apart.

*CFAR detection:*
Targets are detected by comparing each range-Doppler cell to the statistical noise floor of neighboring cells.
This adapts to varying clutter levels automatically.

**Output artifact:**
Per-time-window radar detections: `{range_m, velocity_mps, azimuth_deg, snr_db}` for each detected target.
Optional: range-Doppler heatmap image.

**Human focus:**
- Learn the FMCW chirp principle: increasing frequency + time delay → beat frequency = range.
- Understand range-Doppler maps: two axes give simultaneous range and velocity of every target.
- Know the key radar artifacts: range sidelobes, velocity ambiguity (aliasing), multipath reflections.
- Understand when radar is worth the complexity: primarily for all-weather operation and velocity measurement.

**Common failure modes:**
- No radar → step skipped.
- Static clutter (ground, buildings) overwhelms detection → Doppler filtering needed to separate stationary from moving.
- Velocity ambiguity: targets faster than `PRF/2` wrap around in Doppler space.
- Multipath: ground reflection interferes with direct path signal, causing ghost targets.

---

<a id="step-15-gnss-r-and-satellite-signal-reception"></a>
## Step 15. GNSS-R and satellite signal reception

**What it does:**
Ingest GPS/GNSS position data from the platform, extract position accuracy metrics, and optionally process GNSS-R (reflectometry) data.
Pull external evidence from satellite-derived products: AIS shipping data, ADS-B aircraft tracking, space weather indices.

**Why it matters:**
This step broadens context beyond the camera platform itself.
GNSS provides the absolute geographic reference that anchors all other observations to real-world coordinates.
GNSS-R adds soil moisture, sea surface height, and ice cover sensing from reflected navigation signals.
External satellite feeds (AIS, ADS-B) add activity evidence that the onboard sensors cannot observe.

**Implementation:**
- GPS sidecar extraction: `pipeline/gps_extractor.py`
- GNSS accuracy metrics: HDOP, VDOP, satellite count, fix type
- Optional AIS/ADS-B integration via external API (not enabled by default)

**Key concepts:**

*GNSS dilution of precision (DOP):*
DOP values describe how satellite geometry affects position accuracy.
HDOP < 1 is excellent. HDOP 1-2 is good. HDOP > 5 means poor geometry; position error grows significantly.
DOP is not the same as actual error: low DOP + good signal = accurate; low DOP + multipath = still wrong.

*GNSS-R (reflectometry):*
GNSS signals reflected off the Earth's surface carry information about the surface properties.
Delay-Doppler maps of the reflected signal can measure soil moisture, sea roughness, and snow depth.
This is a specialized technique requiring dual-antenna receivers; rare in standard drone missions.

*AIS and ADS-B:*
AIS (Automatic Identification System): maritime vessel transponder data. Available globally.
ADS-B (Automatic Dependent Surveillance-Broadcast): aircraft transponder data. Available globally.
Both provide activity context for missions near ports, airports, or shipping lanes.

**Output artifact:**
Per-frame GPS quality JSON: `{lat, lon, alt_m, hdop, vdop, fix_type, satellite_count}`.
Optional: AIS/ADS-B contact list for the mission area during the mission time window.

**Human focus:**
- Understand DOP: why satellite geometry matters as much as signal strength for position accuracy.
- Know what GNSS fix types mean: no fix, 2D fix, 3D fix, DGPS, RTK.
- Understand the difference between GNSS accuracy (reported by the receiver) and GNSS precision (actual position error after all sources of error).
- Learn when GNSS fails: urban canyons, dense foliage, electromagnetic interference, spoofing.

**Common failure modes:**
- No GPS sidecar → GPS context is null; map registration fails.
- GPS-denied environment (indoor, jammed) → position is reported but wrong; downstream map is corrupted.
- Time synchronization between GPS and video clock → timestamp offset causes trajectory errors.
- GNSS-R requires specific hardware → almost always unavailable in practice.

---

<a id="step-16-inertial-and-barometric-sensing"></a>
## Step 16. Inertial and barometric sensing

**What it does:**
Ingest IMU (Inertial Measurement Unit) data: accelerometer, gyroscope, and optionally magnetometer.
Ingest barometric altitude from pressure sensor.
Compute platform motion state: velocity estimates, angular rate, attitude (roll/pitch/yaw), and altitude profile.

**Why it matters:**
The IMU anchors visual observations to how the platform actually moved.
Without IMU, a camera tilt looks like a scene change; with IMU, it is identified as a platform rotation.
IMU data is essential for:
- Distinguishing camera motion from scene motion in optical flow
- Deblurring frames during high-acceleration maneuvers
- Visual-inertial odometry (VIO) for GPS-denied navigation
- Validating that GPS position jumps are real vs sensor glitches

**Implementation:**
- IMU sidecar ingestion from `.csv`, `.bin`, or proprietary drone flight log format
- Complementary filter or Madgwick filter for attitude estimation
- Barometric altitude integration with GPS altitude cross-check

**Key concepts:**

*IMU noise model:*
Accelerometers and gyroscopes have two dominant noise sources:
- White noise: random measurement fluctuations at each sample.
- Bias drift: slow, unpredictable drift in the zero reading over time.
Integration of accelerometer readings accumulates error quadratically; integration of gyroscope readings drifts linearly.
This is why IMU alone is useless for long-range position estimation without GPS or vision correction.

*Complementary filter:*
Combines accelerometer (stable long-term, noisy short-term) with gyroscope (accurate short-term, drifts long-term) using a frequency-domain filter.
Result: stable attitude estimate that is accurate in both short bursts and over time.

*Barometric altitude:*
Air pressure decreases with altitude: roughly 12 Pa per meter near sea level.
Barometric altitude is accurate for relative height changes but drifts with weather pressure changes.
Combined with GPS altitude, it gives a smooth altitude trace.

**Output artifact:**
Per-timestamp IMU state: `{roll_deg, pitch_deg, yaw_deg, accel_mss, angular_rate_dps, baro_alt_m}`.
Derived motion label: `hover`, `ascending`, `descending`, `fast_translation`, `rotation`.

**Human focus:**
- Understand the difference between sensor-frame acceleration and world-frame acceleration (requires attitude first).
- Learn why IMU integration produces position estimates that are only useful for short intervals (seconds, not minutes).
- Know the complementary filter as the practical alternative to a full EKF for attitude.
- Understand why IMU and GPS must be time-synchronized: a 100 ms offset at 10 m/s flight speed is 1 meter of position error.

**Common failure modes:**
- No IMU sidecar → platform motion is unobservable; optical flow appears mixed.
- IMU not calibrated (factory default gains) → attitude estimates are systematically wrong.
- Vibration from motors → high-frequency noise in accelerometer; attitude filter oscillates.
- Gimbal isolation → IMU measures airframe motion but camera is stabilized; both must be logged.

---

<a id="step-17-atmospheric--environmental-sensing"></a>
## Step 17. Atmospheric / environmental sensing

**What it does:**
Ingest local weather data from an onboard environmental sensor or an external API: temperature, humidity, wind speed and direction, barometric pressure, UV index, precipitation.
Classify current atmospheric condition: clear, cloudy, foggy, windy, or rainy.
Flag frames where weather degrades sensor performance.

**Why it matters:**
Weather changes both what the mission can achieve and how reliably other sensors perform.
Rain degrades LiDAR and radar accuracy.
Fog reduces camera range and saturates thermal contrast.
High wind affects platform stability, introducing vibration in all sensors.
Knowing weather state lets the pipeline weight sensor outputs appropriately.

**Implementation:**
- Onboard sensor: temperature/humidity/pressure from flight controller log
- External API: `pipeline/net_utils.py` fetch from weather provider if GPS coordinates are known
- Condition classifier: rule-based from humidity, visibility, wind speed thresholds

**Key concepts:**

*Turbulence and vibration:*
At wind speeds above ~10 m/s, rotor-wing drones experience significant vibration.
This vibration appears as camera shake, IMU noise spikes, and GPS position jitter.
Weather context explains these artifacts without attributing them to sensor failure.

*Humidity and thermal contrast:*
High relative humidity increases atmospheric absorption in the LWIR band.
At very high humidity, thermal camera range decreases significantly; distant objects lose contrast.

*Visibility and range:*
Fog or haze reduces the range at which objects can be resolved visually.
The camera sees a flat, low-contrast scene beyond a certain range.
Knowing the predicted or measured visibility allows the pipeline to flag frames where depth estimates are likely wrong.

**Output artifact:**
Per-mission atmospheric summary: `{temperature_c, humidity_pct, wind_speed_mps, wind_dir_deg, visibility_km, condition_label}`.
Per-frame weather quality flag: `{sensor_degradation_risk}` for frames where conditions are likely to impair sensor accuracy.

**Human focus:**
- Understand how wind speed affects drone stability and what that means for frame quality.
- Know the atmospheric windows: wavelengths where the atmosphere is transparent (visible, NIR, LWIR) vs absorbing (MWIR in humid conditions).
- Learn when to trust vs discount sensor readings based on weather.

**Common failure modes:**
- No onboard weather sensor → conditions unknown; API fallback depends on GPS coordinates being valid.
- API rate limiting → weather not fetched; condition label defaults to "unknown".
- Pressure used for altitude (barometry) is confused with pressure used for weather: they are the same sensor but serve different purposes.

---

<a id="step-18-chemical--gas--radiation-sensing"></a>
## Step 18. Chemical / gas / radiation sensing

**What it does:**
Ingest measurements from gas sensors (CO, CO₂, methane, VOCs, O₃), radiation detectors (Geiger-Müller, scintillator), or chemical agent detectors.
Threshold-alert on readings above safety or operational limits.
Geo-tag anomalous readings with GPS coordinates for spatial mapping.

**Why it matters:**
These are often the highest-value "invisible" signals in a mission log.
A visual inspection of a refinery from above cannot detect a methane leak.
A radiation anomaly near an infrastructure target is not visible in any camera.
When present, these sensors have the clearest and most actionable output of any modality.

**Implementation:**
- Sensor sidecar ingestion from CSV, serial port log, or proprietary detector format
- Unit conversion and calibration: raw ADC → PPM (gas), CPM → µSv/h (radiation)
- Geo-tagging: align sensor timestamp to GPS track
- Threshold alerting: configurable per-sensor alert levels

**Key concepts:**

*Gas sensor operating principles:*
- Electrochemical: measures gas by the current produced when it reacts at an electrode. Accurate but slow (minutes to equilibrate).
- Metal oxide semiconductor (MOS): changes resistance in presence of gas. Fast but sensitive to humidity and temperature.
- Infrared absorption (NDIR): measures gas absorption of an IR beam. Very specific per gas type, not affected by humidity.

*Radiation detection:*
- Geiger-Müller tube: counts individual ionizing particles. Very sensitive at low cost. Does not distinguish particle types.
- Scintillation detector: measures energy spectrum of radiation. Can identify isotopes.
- CPM to dose conversion requires geometry and detector efficiency factors.

*Sensor cross-sensitivity:*
Most gas sensors respond to multiple species. A CO sensor also responds to hydrogen and VOCs.
Cross-sensitivity tables from the manufacturer must be used to correct readings.

**Output artifact:**
Per-sample sensor readings: `{timestamp, lat, lon, gas_ppm, radiation_usv_h, alert_triggered}`.
Spatial alert map: KML or GeoJSON file with alert locations overlaid on the mission GPS track.

**Human focus:**
- Understand electrochemical sensor warm-up time: cold readings during first 30-60 seconds are unreliable.
- Learn when MOS sensors are unreliable: high humidity, rain, or condensation.
- Know the difference between detection (is it there?) and quantification (how much?): most drone sensors do the former.
- Understand why geo-tagging precision matters: a 10-meter GPS error puts a gas plume in the wrong location.

**Common failure modes:**
- No sensor payload → step skipped; all chemical/radiation context is absent.
- Sensor not warmed up → first readings are falsely high or low.
- GPS-sensor timestamp mismatch → readings are geo-tagged to wrong locations.
- Wind plume dilution → sensor is downwind but concentration is below detection threshold even with an active source.

---

<a id="step-19-acoustic-sensing"></a>
## Step 19. Acoustic sensing

**What it does:**
Separate from ASR (Step 5, which handles speech): ingest audio and extract non-speech acoustic features.
Classify environmental sounds: vehicle engines, aircraft, gunfire, machinery, wildlife, water, wind.
Detect acoustic events with onset timestamps.

**Why it matters:**
Sound often confirms or contradicts what the camera shows.
A vehicle that looks stationary in the frame but has a running engine is audible.
An aircraft that has left the camera frame is still acoustically present.
Gunfire or impact sounds can precede visible effects by multiple frames.
Acoustic event timestamps provide an independent temporal anchor for correlating other sensors.

**Implementation:**
- Audio feature extraction: MFCC, mel spectrogram, chroma, spectral centroid
- Environmental sound classification: pretrained model (YAMNet, VGGish, BEATs)
- Onset detection: energy envelope + spectral flux thresholding
- Separation of ASR-band speech from environmental audio: VAD gating

**Key concepts:**

*MFCCs (Mel Frequency Cepstral Coefficients):*
The standard feature representation for audio classification.
Computed by: FFT → mel filterbank → log → discrete cosine transform.
MFCCs capture the spectral envelope of sound (how energy is distributed across frequency) in a perceptually-scaled way.

*Acoustic event detection vs scene classification:*
Event detection: something happened at time T (onset + offset). E.g., "gunfire at 12.3 s".
Scene classification: what kind of environment is this audio? E.g., "urban traffic". Operates on windows.
Both are useful; the pipeline uses both.

*Propeller wash and wind noise:*
Drone flights have high-energy low-frequency noise from propellers.
This masks other sounds at close range.
High-pass filtering or notch filtering removes the rotor fundamental frequency, revealing quieter sounds.

**Output artifact:**
Per-window acoustic features: `{dominant_class, confidence, event_list, onset_timestamps}`.
Acoustic timeline: list of detected events with onset/offset times and class labels.

**Human focus:**
- Understand MFCCs as the audio equivalent of visual feature descriptors.
- Learn the difference between sound event detection (temporal) and audio scene classification (global).
- Know the dominant failure mode: propeller noise from the drone itself masks almost everything else below 5 kHz.
- Understand why acoustic timestamps are valuable: they are independent of camera rate and GPS accuracy.

**Common failure modes:**
- Drone rotor noise overwhelms everything → classification fails unless rotor notch filter is applied.
- Low sample rate audio (8 kHz) from cheap microphone → cannot resolve frequencies above 4 kHz; most acoustic events are attenuated.
- Wind noise in microphone → broadband noise masks acoustic events.
- No external microphone (only onboard) → all sound is dominated by airframe vibration.

---

<a id="step-20-sensor-fusion-analysis"></a>
## Step 20. Sensor fusion analysis

**What it does:**
Time-align all sensor evidence collected in Steps 9-19 to a common timeline, merge them into a unified per-frame context block, resolve contradictions, and flag high-confidence anomalies.

**Why it matters:**
Independent sensors are useful.
Time-aligned sensors are much more useful: a radar detection at 120 m at 2.4 s can be linked to a LiDAR cluster at 2.4 s and a visual object detection at the same timestamp.
Agreement across modalities increases confidence.
Disagreement across modalities is informative: it either means one sensor is wrong or that something is genuinely strange.

**Implementation:**
- [`pipeline/workflows/local/_common.py`](../../pipeline/workflows/local/_common.py) — `VideoKnowledge` accumulator
- Timestamp alignment: each sensor stream is resampled or nearest-matched to the frame `t_sec` grid
- Lag tolerance: configurable per-sensor offset to account for processing delay

**Key concepts:**

*Timestamp alignment:*
Each sensor records at its own rate and clock.
The pipeline aligns everything to the video frame timestamps (`t_sec`) as the master clock.
A sensor reading within ±N seconds of a frame timestamp is considered "aligned" to that frame.
N varies per sensor: for LiDAR (10 Hz), ±0.05 s; for weather API (1/hour), ±1800 s.

*Fusion vs integration:*
Fusion: combine multiple sensor outputs into a single representation (e.g., weighted average of radar + LiDAR range estimates).
Integration: add a new independent observation alongside existing ones (e.g., acoustic event + visual detection at the same timestamp).
The pipeline mostly does integration (each sensor result is kept separately in `VideoKnowledge`) and lets Qwen reason across them.

*Missing data handling:*
When a sensor is absent or fails, the pipeline must not propagate nulls as if they were observations.
Missing data is explicitly marked absent; Qwen receives no context line for that sensor, rather than a confusing zero.

*Contradiction detection:*
When two sensors disagree (e.g., LiDAR says obstacle at 50 m, radar says nothing at 50 m), the pipeline logs the contradiction.
This is a signal for human review, not for automatic resolution.

**Output artifact:**
The `VideoKnowledge` accumulator with all sensor evidence merged.
For each frame `t_sec`: the full `context_for_frame(t_sec)` string that Qwen receives.
Contradiction log: list of timestamp + modality pairs where sensors disagreed.

**Human focus:**
- Trace one fused frame context from raw sensor sidecar inputs to the final `context_for_frame()` string.
- Understand timestamp alignment windows and how they interact with sensor update rates.
- Learn to distinguish genuine sensor disagreement from time synchronization error (same root cause, different interpretation).
- Know when fusion reduces uncertainty vs when it just adds noise.

**Common failure modes:**
- All sensors except RGB camera are absent → fusion step is a no-op; only camera evidence matters.
- Clock drift between sensors → temporal alignment fails; sensor readings are matched to wrong frames.
- Lag in slower sensors (weather API, GPS at 1 Hz) → readings appear stale; short events are missed.
- Fusing contradictory sensors without flagging → Qwen receives inconsistent context and produces unreliable output.

---

## What A Human Should Learn In This Phase

Do not try to become a specialist in every sensor first.
The practical study order:

1. Learn what each sensor is physically measuring (wavelength, electrical quantity, acoustic pressure, etc.).
2. Learn what each sensor cannot tell you (its blind spots, failure conditions, and physical limits).
3. Learn how the pipeline stores each sensor's evidence in `VideoKnowledge`.
4. Learn how fusion aligns them in time and what happens when they disagree.

The most valuable skill in this phase is understanding **sensor failure modes** — not just the happy path.
A system that trusts a failed sensor is more dangerous than one with no sensor at all.

## Related Docs

- [Perception core: Steps 1-8](02_perception_core_steps_01_08.md)
- [Agentic knowledge flow](06_agentic_knowledge_flow.md)
- [Tracking and mapping: Steps 21-27](04_tracking_mapping_steps_21_27.md)
- [Pipeline architecture](../pipeline.md)
- [Day-by-day syllabus](07_day_by_day_syllabus.md)
