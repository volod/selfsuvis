#!/usr/bin/env bash
# selfsuvis-prepare-sensor-data.sh — download public sample data for all physical sensor modalities.
#
# Usage:
#   ./scripts/ssv/ssv-prepare-sensor-data.sh [OUTPUT_DIR]
#
# OUTPUT_DIR defaults to data/sensors/.  The script creates one subdirectory
# per sensor step (step09_rf/, step10_thermal/, …) containing the downloaded
# samples.  Each directory also gets a README.txt with the dataset licence and
# the sidecar naming convention for selfsuvis.
#
# Steps covered:
#   Step 9  — RF / SDR             (RadioML 2018.01a shard — manual download note)
#   Step 10 — Thermal imaging      (FLIR ADAS sample — manual download note)
#   Step 11 — Multispectral        (Indian Pines + Salinas hyperspectral .mat files)
#   Step 12 — Event camera         (N-Caltech101 sample subset)
#   Step 13 — LiDAR                (KITTI odometry velodyne scan sample)
#   Step 14 — Radar                (RADIATE Oxford sequence sample — manual download note)
#   Step 15 — GNSS-R / ADS-B      (CYGNSS 1-orbit DDM + OpenSky 1-hour ADS-B)
#   Step 16 — IMU / Inertial       (EuRoC MAV MH_01 IMU CSV)
#   Step 17 — Atmospheric          (ERA5 single-level sample via CDS API)
#   Step 18 — Gas / radiation      (Open-Meteo AQI sample + Safecast GeoJSON)
#   Step 19 — Acoustic             (ESC-50 sample + xeno-canto bird recordings)

set -euo pipefail

OUT="${1:-data/sensors}"
mkdir -p "$OUT"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

log()  { echo -e "${GREEN}[prepare_sensor_data]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
note() { echo -e "${BOLD}[NOTE]${RESET} $*"; }

# -- Video discovery: use the existing video basename as the sensor data key ---
# Scan data/videos/ for a video file first.  If found, generated sidecar
# files will share that basename so they are ready to use without renaming.
# Falls back to "sample_mission_042" when no video is present yet.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
_SENSORS_DIR="$_REPO_ROOT/src/selfsuvis/scripts/sensors"
SENSOR_VIDEO_BASENAME="sample_mission_042"
_VIDEO_DIR="$_REPO_ROOT/data/videos"
if [[ -d "$_VIDEO_DIR" ]]; then
  _FOUND="$(ls "$_VIDEO_DIR"/*.mp4 "$_VIDEO_DIR"/*.mov "$_VIDEO_DIR"/*.avi \
              "$_VIDEO_DIR"/*.mkv 2>/dev/null | head -1 || true)"
  if [[ -n "$_FOUND" ]]; then
    _NAME="$(basename "$_FOUND")"
    SENSOR_VIDEO_BASENAME="${_NAME%.*}"   # strip extension
    log "Found video: $_FOUND"
    log "  → sensor sidecars will use basename '${SENSOR_VIDEO_BASENAME}'"
  fi
fi
export SENSOR_VIDEO_BASENAME

# -- helpers -------------------------------------------------------------------

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required command '$1' not found. Install it and re-run."; exit 1; }
}

maybe_download() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    log "Already exists: $dest — skipping."
  else
    log "Downloading $url"
    curl -fL --progress-bar -o "$dest" "$url"
  fi
}

write_readme() {
  local dir="$1" step="$2" name="$3" licence="$4" sidecar="$5"
  cat > "$dir/README.txt" <<EOF
Dataset:     $name
Step:        $step
Licence:     $licence
Sidecar fmt: $sidecar

Place sidecar next to the matching video:
  data/videos/mission_042.<ext>

See local_path.md and docs/learning_path/03_sensor_steps_09_20.md for full integration details.
EOF
}

require_cmd curl
require_cmd python3

# -- Step 9: RF / SDR ----------------------------------------------------------
DIR="$OUT/step09_rf"
mkdir -p "$DIR"
note "Step 9 — RF / SDR"
note "  DeepSig RadioML 2018.01a requires free account registration."
note "  Manual download: https://www.deepsig.ai/datasets"
note "  Place the .hdf5 shard at: $DIR/radioml_2018.01a.hdf5"
note "  Then rename to match your video: data/videos/mission_042.iq (float32 I/Q)"
echo ""
write_readme "$DIR" "9" "DeepSig RadioML 2018.01a" \
  "CC BY-SA 4.0 — free for research" \
  "${SENSOR_VIDEO_BASENAME}.iq (interleaved float32) or ${SENSOR_VIDEO_BASENAME}.sigmf-data + .sigmf-meta"

# Generate a minimal SigMF reference meta file.
# (The SigMF repo no longer hosts an examples/ directory — generating locally.)
_SIGMF_META="$DIR/${SENSOR_VIDEO_BASENAME}.sigmf-meta"
if [[ ! -f "$_SIGMF_META" ]]; then
  python3 - "$_SIGMF_META" <<'PYEOF'
import json, sys
meta = {
    "global": {
        "core:datatype": "cf32_le",
        "core:sample_rate": 2.4e6,
        "core:version": "1.2.0",
        "core:description": "Synthetic I/Q reference — selfsuvis RF sidecar"
    },
    "captures": [{"core:sample_start": 0, "core:frequency": 915e6}],
    "annotations": []
}
with open(sys.argv[1], "w") as f:
    json.dump(meta, f, indent=2)
print(f"Generated: {sys.argv[1]}")
PYEOF
fi

log "Step 9 RF sample directory: $DIR"

# -- Step 10: Thermal / Infrared -----------------------------------------------
DIR="$OUT/step10_thermal"
mkdir -p "$DIR"
note "Step 10 — Thermal / Infrared"
note "  FLIR ADAS Thermal Dataset requires registration."
note "  Manual download: https://www.flir.com/oem/adas/adas-dataset-form/"
note "  After download, place thermal frames at: $DIR/flir_adas/"
note "  Sidecar: data/videos/mission_042.thermal.mp4 (GREY16-encoded radiometric video)"
echo ""
write_readme "$DIR" "10" "FLIR ADAS Thermal Dataset" \
  "FLIR Research Use Licence (registration required)" \
  "${SENSOR_VIDEO_BASENAME}.thermal.mp4 (GREY16 radiometric video) or ${SENSOR_VIDEO_BASENAME}.thermal/ (TIFF sequence)"

# KAIST sample link (cannot wget without account; provide note)
note "  KAIST Multispectral: https://soonminhwang.github.io/rgbt-ped-detection/"
log "Step 10 thermal sample directory: $DIR"

# -- Step 11: Multispectral / Hyperspectral ------------------------------------
DIR="$OUT/step11_multispectral"
mkdir -p "$DIR"
log "Step 11 — Multispectral / Hyperspectral"

# Indian Pines hyperspectral (publicly hosted on UHU server)
INDIAN_PINES_URL="http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
maybe_download "$INDIAN_PINES_URL" "$DIR/Indian_pines_corrected.mat" || \
  warn "Indian Pines .mat download failed — download manually from: $INDIAN_PINES_URL"

SALINAS_URL="http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat"
maybe_download "$SALINAS_URL" "$DIR/Salinas_corrected.mat" || \
  warn "Salinas .mat download failed — download manually from: $SALINAS_URL"

cp "$_SENSORS_DIR/load_hyperspectral.py" "$DIR/"

write_readme "$DIR" "11" "Indian Pines & Salinas Hyperspectral" \
  "Public domain (academic benchmark)" \
  "${SENSOR_VIDEO_BASENAME}.multispectral/ directory containing per-band GeoTIFF: band_R.tif, band_G.tif, band_RE.tif, band_NIR.tif"
log "Step 11 multispectral samples: $DIR"

# -- Step 12: Event Camera -----------------------------------------------------
DIR="$OUT/step12_event_camera"
mkdir -p "$DIR"
log "Step 12 — Event Camera (neuromorphic)"
note "  N-Caltech101 dataset: https://www.garrickorchard.com/datasets/n-caltech101"
note "  DSEC dataset: https://dsec.ifi.uzh.ch/"
note "  Both require manual download; place event files at: $DIR/"
note "  Sidecar: data/videos/mission_042.events.raw (Prophesee RAW format)"
note "         or mission_042.events.h5 (iniVation DV format)"
note ""
note "  Install Prophesee MetavisionSDK for .raw decoding:"
note "    https://docs.prophesee.ai/stable/installation/index.html"
note "  Install tonic for PyTorch Dataset wrappers:"
note "    pip install tonic"
write_readme "$DIR" "12" "N-Caltech101 / DSEC event camera datasets" \
  "N-Caltech101: free for research. DSEC: CC BY 4.0" \
  "${SENSOR_VIDEO_BASENAME}.events.raw (Prophesee) or ${SENSOR_VIDEO_BASENAME}.events.h5 (iniVation)"
log "Step 12 event camera sample directory: $DIR"

# -- Step 13: LiDAR ------------------------------------------------------------
DIR="$OUT/step13_lidar"
mkdir -p "$DIR"
log "Step 13 — LiDAR / Active Ranging"
note "  KITTI odometry dataset: https://www.cvlibs.net/datasets/kitti/"
note "  Register for free, then download sequence 00 velodyne.zip (~2.8 GB for full sequence)"
note "  For a quick start, download only the calibration + first 5 scans:"
note "    sequence 00, frames 000000-000004 from the velodyne_points directory"
note "  Place .bin scans at: $DIR/kitti_seq00/velodyne/"
note "  Sidecar: data/videos/mission_042.lidar.pcd  (single merged scan)"
note "           or mission_042.lidar.mcap           (MCAP with PointCloud2 topics)"

cp "$_SENSORS_DIR/visualise_pcd.py" "$DIR/"

write_readme "$DIR" "13" "KITTI Odometry LiDAR + SemanticKITTI" \
  "KITTI: non-commercial research licence. SemanticKITTI: CC BY-NC-SA 4.0" \
  "${SENSOR_VIDEO_BASENAME}.lidar.pcd (merged scan) or ${SENSOR_VIDEO_BASENAME}.lidar.mcap (MCAP PointCloud2)"
log "Step 13 LiDAR sample directory: $DIR"

# -- Step 14: Radar ------------------------------------------------------------
DIR="$OUT/step14_radar"
mkdir -p "$DIR"
note "Step 14 — Radar (FMCW / Doppler / SAR)"
note "  RADIATE dataset: https://pro.hw.ac.uk/radiate/"
note "  View-of-Delft: https://github.com/tudelft-iv/view-of-delft-dataset"
note "  Both require manual download."
note "  Sidecar: data/videos/mission_042.radar.bin  (TI DCA1000 raw ADC IQ)"
note "           or mission_042.radar.csv           (pre-processed detections)"
note ""
note "  Install OpenRadar for FMCW signal processing:"
note "    pip install git+https://github.com/PreSenseRadar/OpenRadar.git"
write_readme "$DIR" "14" "RADIATE radar + LiDAR + stereo + GPS" \
  "RADIATE: non-commercial research. View-of-Delft: CC BY-NC-SA 4.0" \
  "${SENSOR_VIDEO_BASENAME}.radar.bin (TI DCA1000 IQ) or ${SENSOR_VIDEO_BASENAME}.radar.csv (detections)"
log "Step 14 radar sample directory: $DIR"

# -- Step 15: GNSS-R + Satellite Signals --------------------------------------
DIR="$OUT/step15_gnss_satellite"
mkdir -p "$DIR"
log "Step 15 — GNSS-R + Satellite Signal Reception"

# OpenSky 1-hour ADS-B sample (public API)
ADSB_SAMPLE="$DIR/opensky_sample.json"
if [[ ! -f "$ADSB_SAMPLE" ]]; then
  log "Downloading OpenSky ADS-B 1-hour sample (state vectors)..."
  # OpenSky anonymous API: states for 1 hour window
  # Using a small bounding box (Europe) to limit size
  curl -fsSL \
    "https://opensky-network.org/api/states/all?lamin=47.0&lomin=8.0&lamax=48.0&lomax=9.0" \
    -o "$ADSB_SAMPLE" 2>/dev/null || \
    warn "OpenSky API request failed (rate limit or network). Download manually: https://opensky-network.org/data/datasets"
fi

cp "$_SENSORS_DIR/generate_adsb_sidecar.py" "$DIR/"
python3 "$DIR/generate_adsb_sidecar.py" "$SENSOR_VIDEO_BASENAME" "$DIR" && \
  log "Generated synthetic ADS-B sidecar JSONL at $DIR/${SENSOR_VIDEO_BASENAME}.adsb.jsonl"

note "  CYGNSS GNSS-R DDMs: https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_V3.1"
note "  ESA SMOS: https://earth.esa.int/eogateway/missions/smos"
note "  Both require NASA Earthdata / ESA EO Sign-In registration."
note "  Sidecar: data/videos/mission_042.gnssr.bin (raw IQ for pyGNSSR)"
note "           or mission_042.adsb.jsonl         (dump1090 aircraft per second)"

write_readme "$DIR" "15" "CYGNSS GNSS-R + OpenSky ADS-B + MarineCadastre AIS" \
  "CYGNSS: NASA Open Data. OpenSky: CC BY 4.0. MarineCadastre AIS: public domain" \
  "${SENSOR_VIDEO_BASENAME}.adsb.jsonl (aircraft/sec) or ${SENSOR_VIDEO_BASENAME}.gnssr.bin (raw IQ)"
log "Step 15 GNSS/satellite sample directory: $DIR"

# -- Step 16: IMU / Inertial ---------------------------------------------------
DIR="$OUT/step16_imu"
mkdir -p "$DIR"
log "Step 16 — IMU + Inertial / Barometric Sensing"

cp "$_SENSORS_DIR/generate_imu_sidecar.py" "$DIR/"
python3 "$DIR/generate_imu_sidecar.py" "$SENSOR_VIDEO_BASENAME" "$DIR"

note "  EuRoC MAV: https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf"
note "  TUM-VI:    https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset"
note "  Both provide ASL CSV format (timestamp,ax,ay,az,gx,gy,gz)."
write_readme "$DIR" "16" "EuRoC MAV + TUM-VI IMU datasets" \
  "EuRoC: CC BY-SA 4.0. TUM-VI: non-commercial research" \
  "${SENSOR_VIDEO_BASENAME}.imu.jsonl (200 Hz) + ${SENSOR_VIDEO_BASENAME}.baro.jsonl (5 Hz) + ${SENSOR_VIDEO_BASENAME}.wind.jsonl (1 Hz)"
log "Step 16 IMU sample directory: $DIR"

# -- Step 17: Atmospheric / Environmental --------------------------------------
DIR="$OUT/step17_atmospheric"
mkdir -p "$DIR"
log "Step 17 — Atmospheric / Environmental Sensing"

cp "$_SENSORS_DIR/generate_env_sidecar.py" "$DIR/"
python3 "$DIR/generate_env_sidecar.py" "$SENSOR_VIDEO_BASENAME" "$DIR"

note "  ERA5 real data: https://cds.climate.copernicus.eu/ (requires CDS account)"
note "  Install cdsapi: pip install cdsapi"
note "  NOAA ISD: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database"
write_readme "$DIR" "17" "ERA5 Reanalysis + NOAA ISD" \
  "ERA5: Copernicus licence (free for research). NOAA ISD: public domain" \
  "${SENSOR_VIDEO_BASENAME}.env.jsonl (1 Hz: temp_c, humidity_pct, pressure_hpa, wind_speed_ms, wind_dir_deg, solar_w_m2)"
log "Step 17 atmospheric sample directory: $DIR"

# -- Step 18: Gas / Radiation --------------------------------------------------
DIR="$OUT/step18_gas_radiation"
mkdir -p "$DIR"
log "Step 18 — Chemical / Gas / Radiation Sensing"

# Real air quality reference sample — Open-Meteo (free, no API key required).
# Fetches the last 24 hours of hourly PM2.5, PM10, NO2, CO, and ozone for London.
# If the request fails (offline / rate-limited) we skip silently; the synthetic
# sidecar below is sufficient for pipeline testing.
_AQI_SAMPLE="$DIR/openmeteo_aqi_sample.json"
if [[ ! -f "$_AQI_SAMPLE" ]]; then
  log "Fetching Open-Meteo air quality sample (no API key required)..."
  curl -fsSL --max-time 15 \
    "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=51.5074&longitude=-0.1278&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone&timezone=Europe%2FLondon&past_days=1" \
    -o "$_AQI_SAMPLE" 2>/dev/null \
    && log "  Open-Meteo AQI sample saved: $_AQI_SAMPLE" \
    || log "  Open-Meteo request skipped (offline or rate-limited) — synthetic data used."
fi

cp "$_SENSORS_DIR/generate_gas_sidecar.py" "$DIR/"
python3 "$DIR/generate_gas_sidecar.py" "$SENSOR_VIDEO_BASENAME" "$DIR"

note "  Open-Meteo air quality (free, no key): https://open-meteo.com/en/docs/air-quality-api"
note "  Safecast radiation data: https://safecast.org/tilemap/"
note "  Safecast API: https://api.safecast.org/en-US/measurements.json"
write_readme "$DIR" "18" "Open-Meteo air quality + Safecast radiation" \
  "Open-Meteo: CC BY 4.0 (open data). Safecast: CC0 public domain" \
  "${SENSOR_VIDEO_BASENAME}.gas.jsonl (1 Hz: co2_ppm, voc_ppb, no2_ppb, pm25_ug_m3, pm10_ug_m3, dose_rate_usv_h)"
log "Step 18 gas/radiation sample directory: $DIR"

# -- Step 19: Acoustic ---------------------------------------------------------
DIR="$OUT/step19_acoustic"
mkdir -p "$DIR"
log "Step 19 — Acoustic Sensing"

# ESC-50 meta CSV (small; from public GitHub)
maybe_download \
  "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv" \
  "$DIR/esc50_meta.csv" || warn "ESC-50 CSV download failed; download manually: https://github.com/karolpiczak/ESC-50"

note "  ESC-50 audio files: https://github.com/karolpiczak/ESC-50/releases/tag/v2.0.0"
note "  Download ESC-50-master.zip (~600 MB) and extract to $DIR/esc50/"
note ""
note "  xeno-canto bird recordings: https://xeno-canto.org/"
note "  FSD50K: https://zenodo.org/record/4060432"
note ""
note "  Sidecar: data/videos/mission_042.audio.wav (48 kHz mono/stereo)"
note "           or mission_042.audio_array.h5     (channels × samples, float32)"

cp "$_SENSORS_DIR/generate_acoustic_sidecar.py" "$DIR/"
python3 "$DIR/generate_acoustic_sidecar.py" "$SENSOR_VIDEO_BASENAME" "$DIR"
write_readme "$DIR" "19" "ESC-50 + xeno-canto + FSD50K acoustic datasets" \
  "ESC-50: CC BY-NC-SA 3.0. xeno-canto: CC (per recording). FSD50K: CC0/CC BY 4.0" \
  "${SENSOR_VIDEO_BASENAME}.audio.wav (48 kHz WAV) or ${SENSOR_VIDEO_BASENAME}.audio_array.h5 (mic array)"
log "Step 19 acoustic sample directory: $DIR"

# -- Summary -------------------------------------------------------------------
echo ""
echo "==============================================================="
echo "  Sensor sample data prepared in: $OUT"
echo ""
echo "  Directories:"
ls "$OUT" | sed 's/^/    /'
echo ""
echo "  Video basename used for sensor sidecars: ${SENSOR_VIDEO_BASENAME}"
echo ""
echo "  Steps requiring manual download (registration required):"
echo "    Step  9  — DeepSig RadioML 2018.01a  https://www.deepsig.ai/datasets"
echo "    Step 10  — FLIR ADAS Thermal         https://www.flir.com/oem/adas/adas-dataset-form/"
echo "    Step 12  — N-Caltech101 / DSEC       https://www.garrickorchard.com/datasets/n-caltech101"
echo "    Step 13  — KITTI odometry velodyne   https://www.cvlibs.net/datasets/kitti/"
echo "    Step 14  — RADIATE radar             https://pro.hw.ac.uk/radiate/"
echo "    Step 15  — CYGNSS GNSS-R DDMs        https://podaac.jpl.nasa.gov/dataset/CYGNSS_L1_V3.1"
echo ""
echo "  Next step: run ./scripts/ssv/ssv-setup.sh to prepare models + Ollama"
echo "==============================================================="
