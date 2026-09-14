# crowd2 Project Context

## Purpose

`crowd2` is a Raspberry Pi 5 crowd analytics application for a booth or retail
area. It consumes an RTSP camera stream, runs YOLOv8 pose inference through a
Hailo accelerator, tracks people inside configurable polygon zones, measures
dwell time, and exposes live and historical analytics through a Flask dashboard.

## Runtime Architecture

```text
RTSP camera
    -> Hailo GStreamer pose pipeline
    -> camera.VideoCamera
       - person filtering and Hailo tracker IDs
       - ankle/bounding-box zone hit testing
       - dwell time and Crowd 1-4 classification
       - spatial and appearance-based re-identification
       - annotated JPEG frame generation
       - batched InfluxDB telemetry
    -> app.py Flask server
       - dashboard and MJPEG stream
       - zone configuration API
       - live statistics API
       - 30-day report API
    -> templates/index.html dashboard
```

The Flask process creates one global `VideoCamera` at import time. The Hailo
pipeline runs in a daemon thread, while `/video_feed` waits for the latest
encoded JPEG frame and serves it as an MJPEG response.

## Main Files

- `app.py`: Flask application and HTTP routes.
- `camera.py`: Hailo/GStreamer integration, detection processing, tracking,
  ReID, zone statistics, JPEG streaming, and InfluxDB writes.
- `settings.py`: Environment-variable based runtime configuration.
- `config.json`: Polygon zone definitions in 1280x720 coordinates.
- `templates/index.html`: Browser dashboard, zone editor, live statistics, and
  report modal.
- `yolov8m_hailo_model/yolov8m.hef`: Repository-local Hailo model artifact.
- `yolov8m_hailo_model/metadata.yaml`: Model metadata used by Hailo tooling.
- `make_metadata.py`: Utility that generates metadata from `yolov8m.pt`.
- `test.py`: Older prototype using OpenCV, Shapely, and a placeholder detector;
  it is not the active application path.

## Detection and Tracking Behavior

- Only detections labeled `person` with confidence at least `0.15` are used.
- The current zone position is determined from ankle landmarks when available.
  If landmarks are unavailable, the bottom-center and bottom corners of the
  bounding box are tested instead.
- A person is tracked only while their position is inside a configured zone.
- Frames are processed at the pipeline rate, but annotated JPEG output is
  throttled to approximately 15 frames per second.
- Track trails are limited to 45 points.
- A track missing for more than 3 seconds is moved to the ReID gallery.
- Spatial ReID merges a newly assigned ID with a nearby prior track when the
  last position is within 150 pixels.
- Appearance ReID compares a 16x16 HSV histogram from the upper half of the
  person crop. Gallery entries remain available for 30 seconds and require a
  correlation score of at least `0.55`, with an ambiguity guard.
- A finalized track is written to InfluxDB only after its ReID gallery entry
  expires. This prevents a brief ID change from creating duplicate counts.

## Dwell Classification

Classification is based on the longest accumulated dwell time across zones:

| Class | Dwell time |
| --- | --- |
| `crowd_1` | Less than 5 seconds |
| `crowd_2` | 5 to less than 15 seconds |
| `crowd_3` | 15 seconds to 15 minutes |
| `crowd_4` | More than 15 minutes |

The live per-zone state tracks `current`, `total`, and all four crowd classes.
The report UI currently renders Crowd 1, Crowd 2, Crowd 3, and the combined
total; the backend still records Crowd 4 values.

## Configuration

`settings.py` supports these environment variables:

| Variable | Purpose |
| --- | --- |
| `RTSP_URL` | Camera stream URL |
| `INFLUX_URL` | InfluxDB server URL |
| `INFLUX_TOKEN` | InfluxDB authentication token |
| `INFLUX_ORG` | InfluxDB organization |
| `INFLUX_BUCKET` | InfluxDB bucket |
| `CAMERA_ID` | Camera tag used in telemetry |
| `HEF_PATH` | Hailo model path |

Do not commit new credentials. Prefer environment variables for deployments and
replace any exposed credentials before sharing the repository.

The active defaults currently expect a local InfluxDB instance and a Hailo
runtime/model installation under `/home/pi/hailo-rpi5-examples`. The RTSP
camera and InfluxDB must be reachable from the Raspberry Pi.

## Current Zones

`config.json` currently contains these zones:

- `Zone AA`: configured polygon.
- `Zone BB`: configured polygon.
- `Zone CC`: configured polygon.
- `Zone DD`: empty polygon and therefore inactive for hit testing.

The dashboard edits coordinates in a fixed 1280x720 logical canvas and saves
the result through `POST /save_config`. The backend reloads the zones without a
process restart.

## HTTP API

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/` | Serves the dashboard. |
| GET | `/video_feed` | Serves the annotated MJPEG stream. |
| GET | `/get_stats` | Returns current in-memory zone statistics. |
| GET | `/get_config` | Returns `{ "zones": ... }` from `config.json`. |
| POST | `/save_config` | Replaces `config.json` and reloads zones. |
| GET | `/get_report` | Queries the last 30 days from InfluxDB. |

## Running

From this directory, with the Hailo SDK, GStreamer GI bindings, camera access,
and Python dependencies installed:

```bash
python3 app.py
```

Open `http://<raspberry-pi-ip>:5000/` in a browser. Required Python packages
include Flask, OpenCV, NumPy, and `influxdb-client`; Hailo and GStreamer
packages are supplied by the Raspberry Pi/Hailo environment.

## Operational Notes

- `camera.py` imports Hailo and GI modules directly, so the application is not
  expected to run on a normal desktop Python environment without the Hailo SDK.
- The GStreamer wrapper disables the local video sink, uses RTSP over TCP with
  low latency, enables NTP/reference timestamps, and attempts to reconnect on
  RTSP EOS or pipeline errors.
- Zone statistics are protected by a thread lock because the pipeline callback
  and Flask request handlers access them concurrently.
- The dashboard polls `/get_stats` once per second.
- The report query uses InfluxDB measurement `zone_total` with `camera_id` and
  `zone` tags and a 30-day range.
- Empty or malformed zone configuration is treated as no active zones.