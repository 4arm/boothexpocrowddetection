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
_JPEG_MIN_INTERVAL = 1.0 / 15


# ==================================================================
# Appearance-Based ReID (Color Histogram)
# ==================================================================
class AppearanceReID:
    """Lightweight re-identification using upper-body HSV color histograms.

    When the tracker assigns a new ID to someone who was recently lost,
    this module compares clothing color to a gallery of lost tracks and
    merges them if the appearance matches — even if the bounding box
    changed from head-only to full-body or the person moved.

    Flow:
        track_history  ──(lost >3s)──►  gallery  ──(expired >30s)──►  InfluxDB
                                           ▲ match
                             new detection ─┘
    """

    def __init__(self, match_threshold=0.55, gallery_ttl=30.0):
        self.match_threshold = match_threshold
        self.gallery_ttl = gallery_ttl          # seconds to keep lost tracks
        self.gallery = {}                       # old_id -> {hist, track, time}

    # ----- feature extraction -----

    def extract(self, frame_rgb, x1, y1, x2, y2, frame_w, frame_h, target_w, target_h):
        """Extract a 16×16 HSV histogram from the upper-body crop.

        Coordinates (x1 … y2) are in *target* space (1280×720).  The crop is
        taken from the original-resolution RGB frame for higher quality.
        """
        sx = frame_w / target_w
        sy = frame_h / target_h
        ox1 = max(0, int(x1 * sx))
        oy1 = max(0, int(y1 * sy))
        ox2 = min(frame_w, int(x2 * sx))
        oy2 = min(frame_h, int(y2 * sy))

        h = oy2 - oy1
        w = ox2 - ox1
        if h < 8 or w < 8:
            return None

        # Upper 50 % of bbox = head + torso (most colour-distinctive)
        crop = frame_rgb[oy1:oy1 + int(h * 0.5), ox1:ox2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def blend(old_hist, new_hist, alpha=0.15):
        """Exponential moving average so the feature adapts slowly."""
        if old_hist is None:
            return new_hist
        if new_hist is None:
            return old_hist
        return (1.0 - alpha) * old_hist + alpha * new_hist

    # ----- gallery management -----

    def save(self, track_id, histogram, track_data, timestamp):
        """Add a lost track to the gallery for future matching."""
        if histogram is None:
            return
        self.gallery[track_id] = {
            "hist": histogram.copy(),
            "track": {
                "dwells":     track_data["dwells"].copy(),
                "dom_zone":   track_data["dom_zone"],
                "dom_class":  track_data["dom_class"],
            },
            "time": timestamp,
        }

    def match(self, histogram, current_time):
        """Find the best gallery match for *histogram*.

        Returns (gallery_id, score) or (None, 0.0).
        Rejects ambiguous matches where 2nd-best is ≥ 85 % of 1st-best.
        """
        if histogram is None:
            return None, 0.0

        candidates = []
        for gid, entry in self.gallery.items():
            if current_time - entry["time"] > self.gallery_ttl:
                continue
            score = cv2.compareHist(histogram, entry["hist"], cv2.HISTCMP_CORREL)
            candidates.append((gid, score))

        if not candidates:
            return None, 0.0

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = candidates[0]

        if best_score < self.match_threshold:
            return None, 0.0

        # Ambiguity guard: reject if two gallery entries look very similar
        if len(candidates) > 1 and candidates[1][1] >= best_score * 0.85:
            return None, 0.0

        return best_id, best_score

    def pop(self, track_id):
        """Remove and return a gallery entry."""
        return self.gallery.pop(track_id, None)

    def flush_expired(self, current_time):
        """Remove expired entries; return their track dicts for InfluxDB."""
        expired_tracks = []
        to_remove = []
        for gid, entry in self.gallery.items():
            if current_time - entry["time"] > self.gallery_ttl:
                expired_tracks.append(entry["track"])
                to_remove.append(gid)
        for gid in to_remove:
            del self.gallery[gid]
        return expired_tracks


# ==================================================================
# Hailo GStreamer Pipeline Wrapper
# ==================================================================
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
        
        # Enhance tracking robustness: remember lost people for 30 frames (1 second) 
        # instead of the default 2 frames, to prevent ID flickering.
        pipeline = pipeline.replace("keep-lost-frames=2", "keep-lost-frames=30")
        
        # Lower the extremely strict Intersection-over-Union thresholds (0.9 -> 0.5) 
        # so the tracker doesn't drop the ID when overlapping people change shape slightly
        pipeline = pipeline.replace("iou-thr=0.9", "iou-thr=0.5")
        pipeline = pipeline.replace("init-iou-thr=0.7", "init-iou-thr=0.5")
        
        pipeline = pipeline.replace("rtspsrc", "rtspsrc ntp-sync=true add-reference-timestamp-meta=true protocols=tcp latency=100")
        pipeline = pipeline.replace('caps="video/x-raw, framerate=30/1"', 'caps="video/x-raw"', 1)
        pipeline = pipeline.replace("video-sink=autovideosink", "video-sink=fakesink", 1)
        return pipeline


class UserData(app_callback_class):
    def __init__(self):
        super().__init__()


# ==================================================================
# Main Video Camera / Tracker
# ==================================================================
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

        # Appearance-based ReID
        self.reid = AppearanceReID(match_threshold=0.55, gallery_ttl=30.0)

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
                        "crowd_1": 0, "crowd_2": 0, "crowd_3": 0, "crowd_4": 0
                    }

    def classify_dwell(self, seconds):
        if seconds > 900: return "crowd_4"
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

        # current_frame_data values:
        #   (foot_x, foot_y, zone, x1, y1, x2, y2, histogram)
        current_frame_data = {}

        for det in detections:
            if det.get_label() != "person": continue
            if det.get_confidence() < 0.15: continue
            uid = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(uid) != 1:
                continue

            bbox = det.get_bbox()
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int((bbox.xmin() + bbox.width()) * self.width)
            y2 = int((bbox.ymin() + bbox.height()) * self.height)

            # --- IMPROVED: Skeletal Detection for Zone Placement ---
            # By using Pose Keypoints, the model can predict where the feet are 
            # even if the person is standing behind a chair or table (occluded).
            test_points = []
            has_ankles = False
            
            landmarks = det.get_objects_typed(hailo.HAILO_LANDMARKS)
            if landmarks:
                pts = landmarks[0].get_points()
                if len(pts) >= 17:
                    la, ra = pts[15], pts[16]
                    ax1 = int((la.x() * bbox.width() + bbox.xmin()) * self.width)
                    ay1 = int((la.y() * bbox.height() + bbox.ymin()) * self.height)
                    ax2 = int((ra.x() * bbox.width() + bbox.xmin()) * self.width)
                    ay2 = int((ra.y() * bbox.height() + bbox.ymin()) * self.height)
                    
                    # Test Left Ankle, Right Ankle, and Midpoint
                    test_points = [
                        (ax1, ay1),
                        (ax2, ay2),
                        (int((ax1 + ax2) / 2), int((ay1 + ay2) / 2))
                    ]
                    foot_x, foot_y = int((ax1 + ax2) / 2), int((ay1 + ay2) / 2)
                    has_ankles = True

            if not has_ankles:
                # Fallback to Bounding Box Spatial Smoothing
                foot_x, foot_y = int((x1 + x2) / 2), y2
                test_points = [
                    (x1, y2),                  # Bottom Left
                    (foot_x, foot_y),          # Bottom Center
                    (x2, y2)                   # Bottom Right
                ]

            current_phys_zone = None
            for z_name, contour in self.zones.items():
                for pt in test_points:
                    if cv2.pointPolygonTest(contour, pt, False) >= 0:
                        current_phys_zone = z_name
                        break
                if current_phys_zone: break

            if current_phys_zone is not None:
                # Extract appearance feature for ReID
                hist = self.reid.extract(
                    frame, x1, y1, x2, y2,
                    width, height, self.width, self.height
                )
                current_frame_data[uid[0].get_id()] = (
                    foot_x, foot_y, current_phys_zone,
                    x1, y1, x2, y2, hist
                )

        current_active_ids = set(current_frame_data.keys())

        # ----------------------------------------------------------
        # Stage 1: Spatial merge (fast, for nearby ID re-assignments)
        # ----------------------------------------------------------
        revised_ids = [tid for tid in current_active_ids
                       if tid not in self.last_active_ids and tid in self.track_history]
        lost_ids = [tid for tid in self.track_history if tid not in current_active_ids]

        with self._stats_lock:
            for rev_id in revised_ids:
                rev_cx, rev_cy, _, _, _, _, _, _ = current_frame_data[rev_id]
                for lost_id in lost_ids:
                    last_x, last_y = self.track_history[lost_id]["path"][-1]
                    if np.sqrt((rev_cx - last_x)**2 + (rev_cy - last_y)**2) < 150:
                        lost_track = self.track_history[lost_id]
                        lz, dz, dc = lost_track["phys_zone"], lost_track["dom_zone"], lost_track["dom_class"]

                        if lz in self.zone_stats:
                            self.zone_stats[lz]["current"] = max(0, self.zone_stats[lz]["current"] - 1)
                        if dz in self.zone_stats:
                            self.zone_stats[dz]["total"] = max(0, self.zone_stats[dz]["total"] - 1)
                            self.zone_stats[dz][dc] = max(0, self.zone_stats[dz][dc] - 1)

                        for z, d_time in lost_track["dwells"].items():
                            self.track_history[rev_id]["dwells"][z] = self.track_history[rev_id]["dwells"].get(z, 0) + d_time

                        # Carry over appearance feature
                        if "histogram" in lost_track and lost_track["histogram"] is not None:
                            self.track_history[rev_id]["histogram"] = lost_track["histogram"]

                        del self.track_history[lost_id]
                        lost_ids.remove(lost_id)
                        break

            self.last_active_ids = current_active_ids

            # ----------------------------------------------------------
            # Stage 2: Normal Tracking Loop + ReID gallery matching
            # ----------------------------------------------------------
            for track_id, (cx, cy, current_phys_zone, x1, y1, x2, y2, hist) in current_frame_data.items():
                if track_id not in self.track_history:
                    # --- New ID detected in a zone ---

                    # Try appearance-based ReID against gallery
                    match_id, match_score = self.reid.match(hist, current_time)

                    if match_id is not None:
                        # *** ReID match found — restore lost track ***
                        gallery_entry = self.reid.pop(match_id)
                        gallery_track = gallery_entry["track"]

                        merged_dwells = gallery_track["dwells"].copy()
                        if current_phys_zone in merged_dwells:
                            merged_dwells[current_phys_zone] += 0.1
                        else:
                            merged_dwells[current_phys_zone] = 0.1

                        self.track_history[track_id] = {
                            "last_update": current_time,
                            "last_seen": current_time,
                            "dwells": merged_dwells,
                            "phys_zone": current_phys_zone,
                            "dom_zone": gallery_track["dom_zone"],
                            "dom_class": gallery_track["dom_class"],
                            "histogram": hist,
                            "path": [(cx, cy)]
                        }
                        # Only restore live count (total & crowd_X already counted)
                        self.zone_stats[current_phys_zone]["current"] += 1
                        print(f"[reid] Matched new ID {track_id} → lost gallery (score={match_score:.2f})")

                    else:
                        # Genuinely new person
                        self.track_history[track_id] = {
                            "last_update": current_time,
                            "last_seen": current_time,
                            "dwells": {current_phys_zone: 0.1},
                            "phys_zone": current_phys_zone,
                            "dom_zone": current_phys_zone,
                            "dom_class": "crowd_1",
                            "histogram": hist,
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

                    # Update appearance feature (EMA)
                    track["histogram"] = self.reid.blend(track.get("histogram"), hist)

                    if track["phys_zone"] != current_phys_zone:
                        if track["phys_zone"] in self.zone_stats:
                            self.zone_stats[track["phys_zone"]]["current"] -= 1
                        self.zone_stats[current_phys_zone]["current"] += 1
                        track["phys_zone"] = current_phys_zone

                    if current_phys_zone not in track["dwells"]:
                        track["dwells"][current_phys_zone] = 0
                    track["dwells"][current_phys_zone] += dt

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
            # Cleanup: lost tracks → save to ReID gallery (not InfluxDB yet)
            # ----------------------------------------------------------
            ids_to_remove = []
            for track_id, track in self.track_history.items():
                if current_time - track["last_seen"] > 3.0:
                    # Save appearance to gallery for future ReID matching.
                    # InfluxDB write is deferred until the gallery entry expires.
                    self.reid.save(
                        track_id, track.get("histogram"),
                        track, track["last_seen"]
                    )
                    if track["phys_zone"] in self.zone_stats:
                        self.zone_stats[track["phys_zone"]]["current"] = max(0, self.zone_stats[track["phys_zone"]]["current"] - 1)
                    ids_to_remove.append(track_id)

            for track_id in ids_to_remove:
                del self.track_history[track_id]

            # Flush expired gallery entries → finalize their InfluxDB records
            for expired_track in self.reid.flush_expired(current_time):
                self._write_total(expired_track["dom_zone"], expired_track["dom_class"])

        # --- End critical section ---

        # Draw overlays and encode JPEG only at throttled rate
        if should_render:
            for track_id, (cx, cy, _, x1, y1, x2, y2, _hist) in current_frame_data.items():
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
            result = copy.deepcopy(self.zone_stats)
            
            # Compute live per-tier counts from active tracks
            now_counts = {}
            for z_name in result:
                now_counts[z_name] = {"now_crowd_1": 0, "now_crowd_2": 0, "now_crowd_3": 0, "now_crowd_4": 0}
            
            for track_id, track in self.track_history.items():
                zone = track.get("phys_zone")
                if zone and zone in now_counts:
                    dom_zone = track.get("dom_zone", zone)
                    max_dwell = track.get("dwells", {}).get(dom_zone, 0)
                    tier = self.classify_dwell(max_dwell)
                    now_counts[zone]["now_" + tier] += 1
            
            for z_name in result:
                result[z_name].update(now_counts.get(z_name, {}))
            
            return result

    def __del__(self):
        self.stopped = True
        if hasattr(self, 'write_api'):
            self.write_api.close()
        if hasattr(self, 'influx_client'):
            self.influx_client.close()
