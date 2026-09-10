import threading
import time
import queue
import os
import sys
import json
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# -- Config -------------------------------------------------------------------
RTSP_URL = "rtsp://admin:Mt10ma18@192.168.0.65:554/Streaming/channels/101"
HEF_PATH = "/home/pi/hailo-rpi5-examples/resources/models/hailo8/yolov8m_pose.hef"
CAMERA_ID = "directional_1"
CONFIDENCE = 0.3
FLASK_PORT = 5000

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "c12Exx1gA58AxCrVrPWXP-abTyDb3HV3UTA8MLEhnrQFYrElQX99gvIh95TQY0g1MSI1IE8soYzPYT5bWpUcFA=="
INFLUX_ORG = "MT"
INFLUX_BUCKET = "booth"
CONFIG_FILE = "config.json"
# -----------------------------------------------------------------------------

os.environ["GST_DEBUG"] = "0"

# -- Source Selection State ---------------------------------------------------
VIDEO_SOURCE = None
SOURCE_SELECTED = threading.Event()
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# -- InfluxDB & Tracking State ------------------------------------------------
write_queue = queue.Queue()
try:
    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)
    query_api = influx_client.query_api()
    print("[startup] Successfully connected to InfluxDB.")
except Exception as e:
    print(f"[startup] Error connecting to InfluxDB: {e}")

directional_stats = {"in": 0, "out": 0}
track_state = {} 
count_lock = threading.Lock()

zone_a = np.array([], np.int32)
zone_b = np.array([], np.int32)

def load_config():
    global zone_a, zone_b
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            zone_a = np.array(config.get("zone_a", []), np.int32)
            zone_b = np.array(config.get("zone_b", []), np.int32)
    except Exception:
        zone_a = np.array([], np.int32)
        zone_b = np.array([], np.int32)

def load_baseline():
    global directional_stats
    flux = f"""
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: today())
  |> filter(fn: (r) => r._measurement == "directional_total" and r.camera_id == "{CAMERA_ID}")
  |> sum()
"""
    try:
        tables = query_api.query(flux, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                field = record.get_field()
                if field in directional_stats:
                    directional_stats[field] = int(record.get_value() or 0)
        print(f"[startup] loaded baseline total: IN={directional_stats['in']}, OUT={directional_stats['out']}")
    except Exception as e:
        print(f"[startup] could not load baseline: {e}")

def influx_writer():
    while True:
        point = write_queue.get()
        try:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        except Exception as e:
            write_queue.put(point)
            time.sleep(1)

# -- Flask Server -------------------------------------------------------------
app_stream = Flask(__name__)
frame_cond = threading.Condition()
canvas_lock = threading.Lock() 
latest_jpeg = None
latest_frame_id = 0

PREVIEW_MAX_FPS = 15.0
PREVIEW_MIN_INTERVAL = 1.0 / PREVIEW_MAX_FPS
_last_preview_publish = 0.0

@app_stream.route('/')
def index():
    return render_template('index.html')

@app_stream.route('/get_status', methods=['GET'])
def get_status():
    return jsonify({"source_selected": SOURCE_SELECTED.is_set()})

@app_stream.route('/set_source', methods=['POST'])
def set_source():
    global VIDEO_SOURCE
    if SOURCE_SELECTED.is_set():
        return jsonify({"status": "error", "message": "Source is already running. Please restart the app to change source."})

    # Option 1: RTSP stream chosen
    if request.is_json:
        data = request.json
        if data.get('type') == 'rtsp':
            VIDEO_SOURCE = RTSP_URL
            SOURCE_SELECTED.set()
            return jsonify({"status": "success"})
    
    # Option 2: Video file uploaded
    if 'video' in request.files:
        file = request.files['video']
        if file.filename != '':
            filepath = os.path.join(UPLOAD_DIR, "uploaded_video.mp4")
            file.save(filepath)
            VIDEO_SOURCE = filepath
            SOURCE_SELECTED.set()
            return jsonify({"status": "success"})
            
    return jsonify({"status": "error", "message": "Invalid source submission."})

@app_stream.route("/video_feed")
def video_feed():
    def generate():
        last_id = -1
        while True:
            with frame_cond:
                frame_cond.wait_for(lambda: latest_jpeg is not None and latest_frame_id != last_id, timeout=5.0)
                jpg = latest_jpeg
                last_id = latest_frame_id
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app_stream.route('/get_stats', methods=['GET'])
def get_stats():
    with count_lock:
        return jsonify(directional_stats)

@app_stream.route('/get_config', methods=['GET'])
def get_config_route():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"zone_a": [], "zone_b": []})

@app_stream.route('/save_config', methods=['POST'])
def save_config_route():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(request.json, f)
        load_config()
        return jsonify({"status": "success", "message": "Zones dynamically updated!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def start_flask():
    app_stream.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)

# -- Pipeline -----------------------------------------------------------------
class DirectionalApp(GStreamerPoseEstimationApp):
    def get_pipeline_string(self):
        pipeline = super().get_pipeline_string()
        pipeline = pipeline.replace("vdevice-group-id=1", "vdevice-group-id=1 multi-process-service=true")
        pipeline = pipeline.replace("rtspsrc", "rtspsrc ntp-sync=true add-reference-timestamp-meta=true protocols=tcp latency=100")
        pipeline = pipeline.replace('caps="video/x-raw, framerate=30/1"', 'caps="video/x-raw"', 1)
        pipeline = pipeline.replace("video-sink=autovideosink", "video-sink=fakesink", 1)
        pipeline = pipeline.replace("keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2", "keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=10", 1)
        return pipeline

class UserData(app_callback_class):
    def __init__(self):
        super().__init__()

def point_in_polygon(point, polygon):
    if len(polygon) < 3: return False
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def app_callback(pad, info, user_data):
    global directional_stats, track_state, _last_preview_publish, latest_jpeg, latest_frame_id

    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK

    format, width, height = get_caps_from_pad(pad)
    if not format or not width or not height: return Gst.PadProbeReturn.OK

    now_ns = time.time_ns()
    now_mon = time.monotonic()

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    current_tracks = {}

    for det in detections:
        if det.get_label() != "person" or det.get_confidence() < CONFIDENCE: continue
        uid = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(uid) != 1: continue

        bbox = det.get_bbox()
        x1, y1 = int(bbox.xmin() * width), int(bbox.ymin() * height)
        x2, y2 = int((bbox.xmin() + bbox.width()) * width), int((bbox.ymin() + bbox.height()) * height)
        
        foot_x, foot_y = int((x1 + x2) / 2), y2
        landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmarks:
            pts = landmarks[0].get_points()
            if len(pts) >= 17:
                la, ra = pts[15], pts[16]
                visible = []
                if la.confidence() < 0.3: visible.append((int(la.x() * width), int(la.y() * height)))
                if ra.confidence() < 0.3: visible.append((int(ra.x() * width), int(ra.y() * height)))
                if visible:
                    foot_x = int(sum(p[0] for p in visible) / len(visible))
                    foot_y = int(sum(p[1] for p in visible) / len(visible))

        current_tracks[uid[0].get_id()] = (x1, y1, x2, y2, foot_x, foot_y)

    with count_lock:
        for tid, (x1, y1, x2, y2, foot_x, foot_y) in current_tracks.items():
            curr_zone = None
            if point_in_polygon((foot_x, foot_y), zone_a): curr_zone = "A"
            elif point_in_polygon((foot_x, foot_y), zone_b): curr_zone = "B"

            if tid not in track_state:
                track_state[tid] = {
                    "history": [], 
                    "last_seen": now_mon, 
                    "counted": False,
                    "pending_zone": None,
                    "zone_hits": 0,
                    "path": [(foot_x, foot_y)]
                }
            
            track = track_state[tid]
            track["last_seen"] = now_mon
            track["path"].append((foot_x, foot_y))
            if len(track["path"]) > 45: track["path"].pop(0)
            
            if curr_zone and not track["counted"]:
                if track["pending_zone"] == curr_zone:
                    track["zone_hits"] += 1
                else:
                    track["pending_zone"] = curr_zone
                    track["zone_hits"] = 1
                
                if track["zone_hits"] >= 3:
                    confirmed_zone = curr_zone
                    history = track["history"]
                    
                    if not history or history[-1] != confirmed_zone:
                        history.append(confirmed_zone)
                    
                        if len(history) >= 2:
                            origin = history[0]
                            if origin == "A" and confirmed_zone == "B":
                                directional_stats["out"] += 1
                                track["counted"] = True
                                write_queue.put(Point("directional_total").tag("camera_id", CAMERA_ID).field("out", 1).time(now_ns))
                            elif origin == "B" and confirmed_zone == "A":
                                directional_stats["in"] += 1
                                track["counted"] = True
                                write_queue.put(Point("directional_total").tag("camera_id", CAMERA_ID).field("in", 1).time(now_ns))

        ids_to_remove = [tid for tid, track in track_state.items() if now_mon - track["last_seen"] > 3.0]
        for tid in ids_to_remove: del track_state[tid]

    if now_mon - _last_preview_publish < PREVIEW_MIN_INTERVAL:
        return Gst.PadProbeReturn.OK
    _last_preview_publish = now_mon

    frame = get_numpy_from_buffer(buffer, format, width, height)
    if frame is None: return Gst.PadProbeReturn.OK
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    for tid, (x1, y1, x2, y2, foot_x, foot_y) in current_tracks.items():
        state = track_state.get(tid, {})
        color = (0, 255, 0) if state.get("counted") else (0, 200, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.putText(canvas, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.circle(canvas, (foot_x, foot_y), 5, color, -1) 

    with canvas_lock:
        ok, jpeg = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ok:
            with frame_cond:
                global latest_jpeg, latest_frame_id
                latest_jpeg = jpeg.tobytes()
                latest_frame_id += 1
                frame_cond.notify_all()

    return Gst.PadProbeReturn.OK

# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    load_config()
    load_baseline()
    user_data = UserData()
    
    # Start web server first to serve the UI
    threading.Thread(target=start_flask, daemon=True).start()
    threading.Thread(target=influx_writer, daemon=True).start()
    
    # Wait for user input from Web UI before initializing the Hailo Pipeline
    print("[startup] Waiting for video source selection from the Web Interface (localhost:5000)...")
    SOURCE_SELECTED.wait()
    print(f"[startup] Selected source: {VIDEO_SOURCE}. Initializing Hailo Pipeline now...")

    sys.argv = [sys.argv[0], "--input", VIDEO_SOURCE, "--hef-path", HEF_PATH, "--disable-sync"]
    app = DirectionalApp(app_callback, user_data)
    app.run()
