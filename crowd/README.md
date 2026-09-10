# Crowd Analytics — Booth AI Tracking System

Real-time crowd analytics for trade-show / retail booths, powered by
**Raspberry Pi 5 + Hailo-8 AI accelerator** with YOLOv8 pose estimation.

## What It Does

- Detects and tracks people via an RTSP camera feed using a GStreamer + Hailo AI pipeline.
- Classifies visitors into configurable polygon **zones** drawn through the web UI.
- Measures **dwell time** per person per zone and classifies engagement level:
  | Classification | Dwell Time |
  |---|---|
  | Crowd 1 | < 5 seconds (passing by) |
  | Crowd 2 | 5 – 15 seconds (glancing) |
  | Crowd 3 | 15 seconds – 15 minutes (engaged) |
  | **Staff** | > 15 minutes (auto-detected) |
- **Automatically identifies staff**: anyone dwelling in a zone for more than 15 minutes
  is promoted to staff and excluded from customer metrics. Staff status persists through
  tracker ID changes via spatial proximity ReID.
- Streams annotated video (bounding boxes, trails, ID labels) to a web dashboard.
- Logs customer telemetry to **InfluxDB** for historical reporting.

## Architecture

```
RTSP Camera
    │
    ▼
GStreamer Pipeline (Hailo-8 HW inference)
    │  YOLOv8 pose estimation @ full frame rate
    ▼
camera.py  ──  VideoCamera
    │  • Person detection + tracking
    │  • Zone hit-testing (cv2.pointPolygonTest)
    │  • Dwell-time classification
    │  • Staff auto-detection (>15 min)
    │  • ReID via spatial proximity merge
    │  • JPEG encoding @ ~15 fps (throttled)
    │  • Batched InfluxDB writes
    ▼
app.py  ──  Flask Web Server (port 5000)
    │  • /video_feed   — MJPEG stream
    │  • /get_stats    — live zone stats JSON
    │  • /get_config   — zone polygon config
    │  • /save_config  — update zones at runtime
    │  • /get_report   — 30-day InfluxDB report
    ▼
index.html  ──  Web Dashboard
    • Live video with zone overlays
    • Draw / edit / delete zones interactively
    • Real-time stats panel (customers + staff per zone)
    • Historical report table from InfluxDB
```

## Requirements

### Hardware
- Raspberry Pi 5
- Hailo-8 AI accelerator (M.2 / HAT)
- RTSP IP camera

### Software
- Python 3.11+
- GStreamer 1.0 with GI bindings
- Hailo RT + hailo-rpi5-examples SDK
- InfluxDB 2.x (local or remote)

### Python Packages
```
flask
opencv-python
numpy
influxdb-client
```

## Setup

### 1. Configure

All settings are in [`settings.py`](settings.py) with environment-variable overrides:

| Variable | Default | Description |
|---|---|---|
| `RTSP_URL` | `rtsp://admin:...@192.168.0.65:554/...` | Camera RTSP stream URL |
| `INFLUX_URL` | `http://localhost:8086` | InfluxDB server URL |
| `INFLUX_TOKEN` | *(built-in)* | InfluxDB API token |
| `INFLUX_ORG` | `MT` | InfluxDB organization |
| `INFLUX_BUCKET` | `booth` | InfluxDB bucket |
| `CAMERA_ID` | `FarmBest` | Camera identifier tag |
| `HEF_PATH` | `/home/pi/hailo-rpi5-examples/.../yolov8m_pose.hef` | Hailo model path |

Override via environment:
```bash
export RTSP_URL="rtsp://user:pass@10.0.0.1:554/stream"
export CAMERA_ID="BoothA"
```

### 2. Run

```bash
python3 app.py
```

The dashboard is served at **http://\<pi-ip\>:5000**.

### 3. Configure Zones

1. Open the dashboard in a browser.
2. Click **"+ Create New Zone"** and name it (e.g., "Booth A", "Entrance").
3. Select the zone and click **"Draw Selected Zone"**.
4. Click points on the video to draw the polygon boundary.
5. Click **"Finish Drawing"** then **"Save AI Config"**.

Zones are saved to [`config.json`](config.json) and applied immediately — no restart needed.

## Staff Detection

Staff are **automatically identified** — no manual tagging required.

### How It Works

1. Every person entering a zone starts as a **customer** (Crowd 1).
2. As they dwell longer, they progress through Crowd 2 → Crowd 3.
3. If any single zone dwell exceeds **15 minutes**, the person is promoted to **Staff**.
4. Once promoted:
   - Their bounding box turns **orange** with a `STAFF:` label.
   - They are **removed from customer counts** (current, total, crowd_1/2/3).
   - They appear under the **"Staff in Zone"** counter in the dashboard.
   - Their departure is **not logged to InfluxDB** (only customer visits are recorded).
5. If the tracker loses a staff member and re-acquires them with a new ID,
   the **ReID merge** detects spatial proximity (<150px) and carries over:
   - Accumulated dwell times
   - Staff flag (`is_staff = True`)

### Tuning the Threshold

Edit `STAFF_DWELL_THRESHOLD` in [`camera.py`](camera.py) (default: 900 seconds = 15 minutes):

```python
STAFF_DWELL_THRESHOLD = 900  # seconds
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web dashboard |
| GET | `/video_feed` | MJPEG video stream |
| GET | `/get_stats` | Live zone statistics (JSON) |
| GET | `/get_config` | Current zone polygon config (JSON) |
| POST | `/save_config` | Save zone config (body: `{"zones": {...}}`) |
| GET | `/get_report` | 30-day daily report from InfluxDB (JSON) |

## File Structure

```
crowd/
├── app.py              # Flask web server + API routes
├── camera.py           # VideoCamera: detection, tracking, staff logic
├── settings.py         # Centralized config (env-var overrides)
├── config.json         # Zone polygon definitions (auto-saved by UI)
├── templates/
│   └── index.html      # Dashboard: video + stats + zone editor + report
├── yolov8m_pose.hef    # Hailo model (in yolov8m_hailo_model/)
├── make_metadata.py    # Utility: generate YOLO model metadata
└── test.py             # Legacy prototype (unused)
```

## Troubleshooting

### Person not detected when sitting
- The detection uses **bbox bottom-center** as the foot position. For seated people,
  ensure the zone polygon extends to cover the seating area (chairs, desks).
- If confidence is too low, seated postures may be filtered. The threshold is 0.3
  in `camera.py` — lowering it may help but increases false positives.

### Staff promoted too quickly / slowly
- Adjust `STAFF_DWELL_THRESHOLD` in `camera.py`. Default is 900s (15 minutes).

### Trails drawing incorrectly
- Trail path length is capped at 45 points. If the trail appears jumpy, the
  tracker may be losing and re-acquiring the person. The ReID merge (150px radius)
  should handle this, but busy scenes may cause mismatches.

### InfluxDB connection errors
- Verify InfluxDB is running: `systemctl status influxdb`
- Check credentials in `settings.py` or environment variables.
- Writes are batched (50 points / 5s flush) — errors appear in console output.

