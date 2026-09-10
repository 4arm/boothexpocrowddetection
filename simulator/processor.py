import cv2
import json
import time
import numpy as np
import threading
import copy
import os
import sys

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'crowd')))
from settings import HEF_PATH

os.environ["GST_DEBUG"] = "0"
_JPEG_MIN_INTERVAL = 1.0 / 30

class HailoVideoApp(GStreamerPoseEstimationApp):
    def __init__(self, callback, user_data, source_url):
        self.source_url = source_url
        super().__init__(callback, user_data)
        
    def get_pipeline_string(self):
        pipeline = super().get_pipeline_string()
        pipeline = pipeline.replace("vdevice-group-id=1", "vdevice-group-id=1 multi-process-service=true")
        pipeline = pipeline.replace("keep-lost-frames=2", "keep-lost-frames=30")
        pipeline = pipeline.replace("iou-thr=0.9", "iou-thr=0.5")
        pipeline = pipeline.replace("init-iou-thr=0.7", "init-iou-thr=0.5")
        pipeline = pipeline.replace("video-sink=autovideosink", "video-sink=fakesink", 1)
        return pipeline

class UserData(app_callback_class):
    def __init__(self):
        super().__init__()

class SimulatorProcessor:
    def __init__(self, video_path):
        self.source = video_path
        self.width = 1280
        self.height = 720
        self.config_file = "/home/pi/booth/crowd/config.json"
        
        self.zones = {}
        self.zone_stats = {}
        self._stats_lock = threading.Lock()

        self.load_config()

        self.latest_jpeg = None
        self.frame_cond = threading.Condition()
        self.stopped = False
        self._last_encode_time = 0.0
        self.is_finished = False

        sys.argv = [
            sys.argv[0],
            "--input", self.source,
            "--hef-path", HEF_PATH,
        ]

        self.user_data = UserData()
        self.app = HailoVideoApp(self.app_callback, self.user_data, self.source)

        self.original_on_eos = self.app.on_eos
        self.app.on_eos = self.on_eos

        self.pipeline_thread = threading.Thread(target=self.app.run, daemon=True)
        self.pipeline_thread.start()

    def on_eos(self):
        print("[video] End of stream reached.")
        self.is_finished = True
        self.original_on_eos()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.zones = {
                    k: np.array(v, np.int32).reshape((-1, 1, 2))
                    for k, v in config.get("zones", {}).items()
                    if len(v) > 0
                }
        except Exception as e:
            self.zones = {}

        with self._stats_lock:
            for z_name in self.zones:
                if z_name not in self.zone_stats:
                    self.zone_stats[z_name] = {"current": 0, "total": 0}

    def app_callback(self, pad, info, user_data):
        buffer = info.get_buffer()
        if buffer is None: return Gst.PadProbeReturn.OK

        format, width, height = get_caps_from_pad(pad)
        if not format or not width or not height: return Gst.PadProbeReturn.OK

        frame = get_numpy_from_buffer(buffer, format, width, height)
        if frame is None: return Gst.PadProbeReturn.OK

        import hailo
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        current_time = time.time()
        should_render = (current_time - self._last_encode_time) >= _JPEG_MIN_INTERVAL

        live_counts = {z: 0 for z in self.zones}
        detected_points = []
        
        for det in detections:
            if det.get_label() != "person": continue
            if det.get_confidence() < 0.2: continue
            uid = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            
            bbox = det.get_bbox()
            x1 = int(bbox.xmin() * self.width)
            y1 = int(bbox.ymin() * self.height)
            x2 = int((bbox.xmin() + bbox.width()) * self.width)
            y2 = int((bbox.ymin() + bbox.height()) * self.height)
            foot_x, foot_y = int((x1 + x2) / 2), y2
            
            track_id = uid[0].get_id() if len(uid) == 1 else "?"
            detected_points.append((x1, y1, x2, y2, foot_x, foot_y, track_id))

            for z_name, contour in self.zones.items():
                if cv2.pointPolygonTest(contour, (foot_x, foot_y), False) >= 0:
                    live_counts[z_name] += 1
                    break

        with self._stats_lock:
            for z, count in live_counts.items():
                self.zone_stats[z]["current"] = count
                if count > self.zone_stats[z]["total"]:
                    self.zone_stats[z]["total"] = count

        if should_render:
            canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            canvas = cv2.resize(canvas, (self.width, self.height))
            
            for z_name, contour in self.zones.items():
                cv2.polylines(canvas, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
                x, y = contour[0][0]
                cv2.putText(canvas, z_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
            for (x1, y1, x2, y2, cx, cy, tid) in detected_points:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)

            ok, jpeg = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self.frame_cond:
                    self.latest_jpeg = jpeg.tobytes()
                    self.frame_cond.notify_all()
                self._last_encode_time = current_time

        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        return Gst.PadProbeReturn.OK

    def process_frame(self):
        with self.frame_cond:
            self.frame_cond.wait(timeout=0.1)
            return self.latest_jpeg

    def get_frontend_stats(self):
        with self._stats_lock:
            return copy.deepcopy(self.zone_stats)

    def stop(self):
        self.stopped = True
        self.pipeline_thread.join(timeout=1.0)

