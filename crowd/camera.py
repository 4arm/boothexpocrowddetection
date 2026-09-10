import cv2
import json
import time
import numpy as np
import threading
import copy
import os
import sys

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import WriteOptions

from settings import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, CAMERA_ID, HEF_PATH

os.environ["GST_DEBUG"] = "0"

# JPEG encode throttle – minimum seconds between encodes (~15 fps).
# Detection/tracking still runs at full pipeline rate.
_JPEG_MIN_INTERVAL = 1.0 / 15

# Staff classification threshold (seconds).  Any person dwelling in a zone
# longer than this is automatically classified as staff and excluded from
# customer crowd metrics.
STAFF_DWELL_THRESHOLD = 900  # 15 minutes


class HailoVideoApp(GStreamerPoseEstimationApp):
    def __init__(self, callback, user_data, source_url):
        self.source_url = source_url
        super().__init__(callback, user_data)

    def _restart_live_pipeline(self):
        print("[rtsp] restarting live pipeline")
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.PLAYING)
        return False

    def on_eos(self):
        if self.source_type == "rtsp":
            print("[rtsp] EOS received, attempting reconnect")
            self.pipeline.set_state(Gst.State.READY)
            GLib.timeout_add_seconds(2, self._restart_live_pipeline)
            return
        super().on_eos()

    def on_error(self, bus, message):
        if self.source_type == "rtsp":
            err, debug = message.parse_error()
            print(f"[rtsp] pipeline error: {err.message} -- attempting reconnect")
            GLib.timeout_add_seconds(2, self._restart_live_pipeline)
            return
        super().on_error(bus, message)

    def get_pipeline_string(self):
        pipeline = super().get_pipeline_string()
        pipeline = pipeline.replace("vdevice-group-id=1", "vdevice-group-id=1 multi-process-service=true")
        pipeline = pipeline.replace("rtspsrc", "rtspsrc ntp-sync=true add-reference-timestamp-meta=true protocols=tcp latency=100")
        pipeline = pipeline.replace('caps="video/x-raw, framerate=30/1"', 'caps="video/x-raw"', 1)
        pipeline = pipeline.replace("video-sink=autovideosink", "video-sink=fakesink", 1)
        return pipeline


class UserData(app_callback_class):
    def __init__(self):
        super().__init__()


class VideoCamera:
    def __init__(self, source):
        self.source = source
        self.width = 1280
        self.height = 720
        self.config_file = "config.json"

        # Dynamic Zones and State Tracking
        self.zones = {}
        self.zone_stats = {}
        self.track_history = {}
        self.last_active_ids = set()
        self._stats_lock = threading.Lock()

        self.load_config()

        # Shared frame buffers for Flask streaming
        self.latest_jpeg = None
        self.frame_cond = threading.Condition()
        self.stopped = False
        self._last_encode_time = 0.0

        # Initialize InfluxDB with batched writes
        try:
            self.influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            self.write_api = self.influx_client.write_api(
                write_options=WriteOptions(batch_size=50, flush_interval=5_000)
            )
            print("Successfully connected to InfluxDB.")
        except Exception as e:
            print(f"Error connecting to InfluxDB: {e}")

        # Configure and start Hailo GStreamer Pipeline in Background Thread
        sys.argv = [
            sys.argv[0],
            "--input", self.source,
            "--hef-path", HEF_PATH,
            "--disable-sync",
        ]

        self.user_data = UserData()
        self.app = HailoVideoApp(self.app_callback, self.user_data, self.source)

        self.pipeline_thread = threading.Thread(target=self.app.run, daemon=True)
        self.pipeline_thread.start()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                # Pre-compute contour format (N,1,2) for cv2.pointPolygonTest
                self.zones = {
                    k: np.array(v, np.int32).reshape((-1, 1, 2))
                    for k, v in config.get("zones", {}).items()
                    if len(v) > 0
                }
        except Exception as e:
            print(f"Error loading config: {e}")
            self.zones = {}

        with self._stats_lock:
            for z_name in self.zones:
                if z_name not in self.zone_stats:
                    self.zone_stats[z_name] = {
                        "current": 0, "total": 0,
                        "crowd_1": 0, "crowd_2": 0, "crowd_3": 0, "crowd_4": 0,
                        "staff": 0
                    }

    def classify_dwell(self, seconds):
        """Classify customer dwell time. Staff threshold is handled separately."""
        if seconds >= 15: return "crowd_3"
        if seconds >= 5:  return "crowd_2"
        return "crowd_1"

    def _write_total(self, zone, cls):
        """Queue a dwell-classification point to InfluxDB (batched, non-blocking)."""
        p = Point("zone_total") \
            .tag("camera_id", CAMERA_ID) \
            .tag("zone", zone) \
            .field(cls, 1)
        try:
            self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
        except Exception as e:
            print(f"InfluxDB Write Error: {e}")

    # ------------------------------------------------------------------
    # GStreamer callback – runs on pipeline thread at full frame rate
    # ------------------------------------------------------------------
    def app_callback(self, pad, info, user_data):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        format, width, height = get_caps_from_pad(pad)
        if not format or not width or not height:
            return Gst.PadProbeReturn.OK

        frame = get_numpy_from_buffer(buffer, format, width, height)
        if frame is None:
            return Gst.PadProbeReturn.OK

        current_time = time.time()

        # Decide whether to render (draw + JPEG encode) this frame
        should_render = (current_time - self._last_encode_time) >= _JPEG_MIN_INTERVAL

        if should_render:
            canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            canvas = cv2.resize(canvas, (self.width, self.height))

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        current_frame_data = {}

        for det in detections:
            if det.get_label() != "person":
                continue
            if det.get_confidence() < 0.3:
                continue
            uid = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(uid) != 1:
                continue

            bbox = det.get_bbox()
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int((bbox.xmin() + bbox.width()) * self.width)
            y2 = int((bbox.ymin() + bbox.height()) * self.height)

            # Use bbox bottom-center as foot position — stable for overhead
            # cameras and avoids false zone hits from unreliable keypoints.
            foot_x, foot_y = int((x1 + x2) / 2), y2

            current_phys_zone = None
            for z_name, contour in self.zones.items():
                if cv2.pointPolygonTest(contour, (foot_x, foot_y), False) >= 0:
                    current_phys_zone = z_name
                    break

            if current_phys_zone is not None:
                current_frame_data[uid[0].get_id()] = (foot_x, foot_y, current_phys_zone, x1, y1, x2, y2)

        current_active_ids = set(current_frame_data.keys())

        # ----------------------------------------------------------
        # ID Merging / ReID by spatial proximity
        # When the tracker assigns a new ID to someone who was just
        # lost, merge dwell times AND staff status from the old track.
        # ----------------------------------------------------------
        revised_ids = [tid for tid in current_active_ids if tid not in self.last_active_ids and tid in self.track_history]
        lost_ids = [tid for tid in self.track_history if tid not in current_active_ids]

        # --- Critical section: protect zone_stats from Flask reader ---
        with self._stats_lock:
            for rev_id in revised_ids:
                rev_cx, rev_cy, _, _, _, _, _ = current_frame_data[rev_id]
                for lost_id in lost_ids:
                    last_x, last_y = self.track_history[lost_id]["path"][-1]
                    if np.sqrt((rev_cx - last_x)**2 + (rev_cy - last_y)**2) < 150:
                        lost_track = self.track_history[lost_id]
                        lz = lost_track["phys_zone"]
                        was_staff = lost_track["is_staff"]

                        if was_staff:
                            # Undo staff counter for lost track
                            if lz in self.zone_stats:
                                self.zone_stats[lz]["staff"] = max(0, self.zone_stats[lz]["staff"] - 1)
                        else:
                            # Undo customer counters for lost track
                            dz, dc = lost_track["dom_zone"], lost_track["dom_class"]
                            if lz in self.zone_stats:
                                self.zone_stats[lz]["current"] = max(0, self.zone_stats[lz]["current"] - 1)
                            if dz in self.zone_stats:
                                self.zone_stats[dz]["total"] = max(0, self.zone_stats[dz]["total"] - 1)
                                self.zone_stats[dz][dc] = max(0, self.zone_stats[dz][dc] - 1)

                        # Merge dwell times and staff flag into revised track
                        for z, d_time in lost_track["dwells"].items():
                            self.track_history[rev_id]["dwells"][z] = self.track_history[rev_id]["dwells"].get(z, 0) + d_time
                        if was_staff:
                            self.track_history[rev_id]["is_staff"] = True

                        del self.track_history[lost_id]
                        lost_ids.remove(lost_id)
                        break

            self.last_active_ids = current_active_ids

            # ----------------------------------------------------------
            # Normal Tracking Loop
            # ----------------------------------------------------------
            for track_id, (cx, cy, current_phys_zone, x1, y1, x2, y2) in current_frame_data.items():
                if track_id not in self.track_history:
                    # --- New Track ---
                    self.track_history[track_id] = {
                        "last_update": current_time,
                        "last_seen": current_time,
                        "dwells": {current_phys_zone: 0.1},
                        "phys_zone": current_phys_zone,
                        "dom_zone": current_phys_zone,
                        "dom_class": "crowd_1",
                        "is_staff": False,
                        "path": [(cx, cy)]
                    }
                    self.zone_stats[current_phys_zone]["current"] += 1
                    self.zone_stats[current_phys_zone]["total"] += 1
                    self.zone_stats[current_phys_zone]["crowd_1"] += 1
                else:
                    # --- Existing Track ---
                    track = self.track_history[track_id]
                    dt = current_time - track["last_update"]
                    track["last_update"] = current_time
                    track["last_seen"] = current_time
                    track["path"].append((cx, cy))

                    if len(track["path"]) > 45:
                        track["path"].pop(0)

                    # --- Zone transition ---
                    if track["phys_zone"] != current_phys_zone:
                        old_phys = track["phys_zone"]
                        if track["is_staff"]:
                            # Staff moving between zones
                            if old_phys in self.zone_stats:
                                self.zone_stats[old_phys]["staff"] = max(0, self.zone_stats[old_phys]["staff"] - 1)
                            self.zone_stats[current_phys_zone]["staff"] += 1
                        else:
                            # Customer moving between zones
                            if old_phys in self.zone_stats:
                                self.zone_stats[old_phys]["current"] -= 1
                            self.zone_stats[current_phys_zone]["current"] += 1
                        track["phys_zone"] = current_phys_zone

                    # --- Dwell accumulation ---
                    if current_phys_zone not in track["dwells"]:
                        track["dwells"][current_phys_zone] = 0
                    track["dwells"][current_phys_zone] += dt

                    # --- Staff promotion check ---
                    if not track["is_staff"]:
                        max_dwell = max(track["dwells"].values())
                        if max_dwell >= STAFF_DWELL_THRESHOLD:
                            # *** Promote to staff ***
                            track["is_staff"] = True
                            old_dom = track["dom_zone"]
                            old_cls = track["dom_class"]
                            # Remove from customer counters
                            if old_dom in self.zone_stats:
                                self.zone_stats[old_dom][old_cls] = max(0, self.zone_stats[old_dom][old_cls] - 1)
                                self.zone_stats[old_dom]["total"] = max(0, self.zone_stats[old_dom]["total"] - 1)
                            if current_phys_zone in self.zone_stats:
                                self.zone_stats[current_phys_zone]["current"] = max(0, self.zone_stats[current_phys_zone]["current"] - 1)
                            # Add to staff counter
                            self.zone_stats[current_phys_zone]["staff"] += 1
                            print(f"[staff] Track {track_id} promoted to STAFF (dwell {max_dwell:.0f}s)")
                        else:
                            # Normal customer dwell classification
                            new_dom_zone = max(track["dwells"], key=track["dwells"].get)
                            new_dom_class = self.classify_dwell(track["dwells"][new_dom_zone])

                            old_dom_zone = track["dom_zone"]
                            old_dom_class = track["dom_class"]

                            if new_dom_zone != old_dom_zone or new_dom_class != old_dom_class:
                                if old_dom_zone in self.zone_stats:
                                    self.zone_stats[old_dom_zone][old_dom_class] -= 1
                                    if new_dom_zone != old_dom_zone:
                                        self.zone_stats[old_dom_zone]["total"] -= 1

                                track["dom_zone"] = new_dom_zone
                                track["dom_class"] = new_dom_class
                                self.zone_stats[new_dom_zone][new_dom_class] += 1
                                if new_dom_zone != old_dom_zone:
                                    self.zone_stats[new_dom_zone]["total"] += 1

            # ----------------------------------------------------------
            # Cleanup lost tracks (not seen for >3 seconds)
            # ----------------------------------------------------------
            ids_to_remove = []
            for track_id, track in self.track_history.items():
                if current_time - track["last_seen"] > 3.0:
                    if track["is_staff"]:
                        # Staff leaving — decrement staff counter, no InfluxDB write
                        if track["phys_zone"] in self.zone_stats:
                            self.zone_stats[track["phys_zone"]]["staff"] = max(0, self.zone_stats[track["phys_zone"]]["staff"] - 1)
                    else:
                        # Customer leaving — write dwell class to InfluxDB
                        self._write_total(track["dom_zone"], track["dom_class"])
                        if track["phys_zone"] in self.zone_stats:
                            self.zone_stats[track["phys_zone"]]["current"] = max(0, self.zone_stats[track["phys_zone"]]["current"] - 1)
                    ids_to_remove.append(track_id)

            for track_id in ids_to_remove:
                del self.track_history[track_id]

        # --- End critical section ---

        # Draw overlays and encode JPEG only at throttled rate
        if should_render:
            for track_id, (cx, cy, _, x1, y1, x2, y2) in current_frame_data.items():
                is_staff = self.track_history.get(track_id, {}).get("is_staff", False)

                if is_staff:
                    color = (255, 165, 0)   # Orange for staff
                    label = f"STAFF: {track_id}"
                else:
                    color = (0, 255, 0)     # Green for customers
                    label = f"ID: {track_id}"

                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)

                if track_id in self.track_history and len(self.track_history[track_id]["path"]) > 1:
                    points = np.array(self.track_history[track_id]["path"], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [points], isClosed=False, color=(0, 255, 255), thickness=3)

            ok, jpeg = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self.frame_cond:
                    self.latest_jpeg = jpeg.tobytes()
                    self.frame_cond.notify_all()
                self._last_encode_time = current_time

        return Gst.PadProbeReturn.OK

    def process_frame(self):
        with self.frame_cond:
            self.frame_cond.wait(timeout=1.0)
            return self.latest_jpeg

    def get_frontend_stats(self):
        with self._stats_lock:
            return copy.deepcopy(self.zone_stats)

    def __del__(self):
        self.stopped = True
        if hasattr(self, 'write_api'):
            self.write_api.close()
        if hasattr(self, 'influx_client'):
            self.influx_client.close()
