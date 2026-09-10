"""Centralized configuration with environment-variable overrides.

All credentials and shared constants live here. Current values are kept as
defaults so existing deployments work without reconfiguration.  To override,
set the corresponding environment variable before starting the app.
"""

import os

INFLUX_URL    = os.environ.get("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.environ.get("INFLUX_TOKEN",  "7T_yxGlTGix-ytg4dkEB5Po_faLfDQ9rIYl-Kp6fhovWPX7IywlIAKp4lZSaQtpMaL8yNikgqZhOF8nzJ6SYnw==")
INFLUX_ORG    = os.environ.get("INFLUX_ORG",    "MT")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "booth")
CAMERA_ID     = os.environ.get("CAMERA_ID",     "FarmBest")
RTSP_URL      = os.environ.get("RTSP_URL",      "rtsp://admin:Mt10ma18@192.168.0.65:554/Streaming/channels/101")
HEF_PATH      = os.environ.get("HEF_PATH",      "/home/pi/booth/crowd/yolov8m_hailo_model/yolov8m.hef")

