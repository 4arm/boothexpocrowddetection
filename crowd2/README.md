# Crowd Analytics — Booth AI Tracking System

Real-time crowd analytics for trade-show / retail booths, powered by
**Raspberry Pi 5 + Hailo-8 AI accelerator** with YOLOv8 pose estimation.

## What It Does

- Detects and tracks people via an RTSP camera feed using a GStreamer + Hailo AI pipeline.
- Classifies visitors into configurable polygon **zones** drawn through the web UI.
- Measures **dwell time** per person per zone and classifies engagement level:

  | Classification | Dwell Time | Meaning |
  |---|---|---|
  | Crowd 1 | < 5 seconds | Passing by |
  | Crowd 2 | 5 – 15 seconds | Glancing |
  | Crowd 3 | 15 seconds – 15 minutes | Engaged |
  | Crowd 4 | > 15 minutes | Highly engaged / long stay |

- Streams annotated video (bounding boxes, trails, ID labels) to a web dashboard.
- Logs customer telemetry to **InfluxDB** for historical reporting.
- **ReID by spatial proximity**: when the tracker assigns a new ID to someone who was
  just lost, dwell times are merged so engagement stats stay accurate.

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
    │  • Dwell-time classification (Crowd 1–4)
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
    • Real-time stats panel per zone
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

## Dwell-Time Tracking

Each person entering a zone is tracked with a unique ID. Their dwell time in each
zone is accumulated continuously. The **dominant zone** (longest total dwell) determines
their classification:

- **Crowd 1** (< 5s) — walked through, no engagement
- **Crowd 2** (5–15s) — brief pause, possible interest
- **Crowd 3** (15s–15m) — meaningful engagement
- **Crowd 4** (> 15m) — extended stay (loyal customer, deep engagement, or staff)

When a person leaves (not seen for >3 seconds), their final dwell classification is
written to InfluxDB for historical reporting.

## ReID (Re-Identification)

The tracker may assign a new ID to someone who was briefly lost (e.g., occlusion, camera glitch, or bounding box changing from head-only to full-body). The system handles this in two stages:

1. **Spatial Merge**: If a new track appears within **150 pixels** of a recently lost track, it is immediately merged.
2. **Appearance-Based ReID**: For each person, the system extracts a 16x16 HSV color histogram from their upper body (clothing color). When a track is lost, its appearance is saved to a gallery for up to 30 seconds. If a new ID appears in a zone, its clothing color is compared against the gallery. If it matches (>55% similarity), the tracks are merged.

When merged:
- All accumulated dwell times are carried over.
- The duplicate counter increments are rolled back so the person is only counted once.
- The InfluxDB write for the old ID is canceled.

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
├── camera.py           # VideoCamera: detection, tracking, dwell logic
├── settings.py         # Centralized config (env-var overrides)
├── config.json         # Zone polygon definitions (auto-saved by UI)
├── templates/
│   └── index.html      # Dashboard: video + stats + zone editor + report
├── yolov8m_hailo_model/# Hailo model files
├── make_metadata.py    # Utility: generate YOLO model metadata
└── test.py             # Legacy prototype (unused)
```

## Troubleshooting

### Person not detected when sitting
- The detection uses **bbox bottom-center** as the foot position. For seated people,
  ensure the zone polygon extends to cover the seating area (chairs, desks).
- If confidence is too low, seated postures may be filtered. The threshold is 0.3
  in `camera.py` — lowering it may help but increases false positives.

### Trails drawing incorrectly
- Trail path length is capped at 45 points. If the trail appears jumpy, the
  tracker may be losing and re-acquiring the person. The ReID merge (150px radius)
  should handle this, but busy scenes may cause mismatches.

### InfluxDB connection errors
- Verify InfluxDB is running: `systemctl status influxdb`
- Check credentials in `settings.py` or environment variables.
- Writes are batched (50 points / 5s flush) — errors appear in console output.
