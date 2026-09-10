import cv2
import numpy as np
import time
import math
from shapely.geometry import Point, Polygon
from influxdb_client import InfluxDBClient, Point as InfluxPoint
from influxdb_client.client.write_api import SYNCHRONOUS

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
RTSP_URL = "rtsp://admin:Mt10ma18@192.168.0.65:554/Streaming/channels/101" 

# Shared Influx Settings
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "7T_yxGlTGix-ytg4dkEB5Po_faLfDQ9rIYl-Kp6fhovWPX7IywlIAKp4lZSaQtpMaL8yNikgqZhOF8nzJ6SYnw=="
INFLUX_ORG    = "MT"
INFLUX_BUCKET = "booth"
CAMERA_ID     = "FarmBest"



ZONE_C_COORDS = [(100, 300), (400, 300), (500, 700), (50, 700)]
ZONE_C_POLYGON = Polygon(ZONE_C_COORDS)

ZONE_A_COORDS = [(450, 300), (900, 300), (950, 700), (550, 700)]
ZONE_A_POLYGON = Polygon(ZONE_A_COORDS)

TIME_GATE_SECONDS = 4.0 # Time-Gate Filtering threshold

# ---------------------------------------------------------
# 2. HAILO INFERENCE WRAPPER (PLACEHOLDER)
# ---------------------------------------------------------
class HailoObjectDetector:
    def __init__(self, hef_path):
        print(f"Loading Hailo model from {hef_path}...")
        self.model_loaded = True 
        
    def infer(self, frame):
        # Returns [x_min, y_min, x_max, y_max, confidence, class_id]
        return [[500, 400, 550, 500, 0.85, 0]] 

# ---------------------------------------------------------
# 3. CENTROID TRACKER & ANALYTICS ENGINE
# ---------------------------------------------------------
class TrackerAndAnalyzer:
    def __init__(self):
        self.active_tracks = {} # {id: {centroid, zone, first_seen_zone_A, state}}
        self.next_id = 1
        
    def determine_raw_zone(self, x, y):
        pt = Point(x, y)
        if ZONE_A_POLYGON.contains(pt):
            return "Zone A"
        elif ZONE_C_POLYGON.contains(pt):
            return "Zone C"
        return "Outside"

    def update_tracks(self, detections):
        current_centroids = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if cls == 0 and conf > 0.5:
                # Use bottom-center for foot placement
                cx, cy = (x1 + x2) / 2, y2
                current_centroids.append((cx, cy, det))

        updated_tracks = {}
        
        # Simple Euclidean Distance matching for tracking IDs
        for cx, cy, det in current_centroids:
            matched_id = None
            min_dist = float('inf')
            
            for track_id, track_data in self.active_tracks.items():
                prev_cx, prev_cy = track_data['centroid']
                dist = math.hypot(cx - prev_cx, cy - prev_cy)
                
                # If distance is small enough, it's the same person
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    matched_id = track_id
            
            raw_zone = self.determine_raw_zone(cx, cy)
            now = time.time()

            if matched_id is None:
                # New Person detected
                matched_id = self.next_id
                self.next_id += 1
                state = "Zone C" # Hierarchical Zone Promotion: Default to Walkway
                first_seen_A = now if raw_zone == "Zone A" else None
            else:
                # Existing Person
                state = self.active_tracks[matched_id]['state']
                first_seen_A = self.active_tracks[matched_id]['first_seen_zone_A']
                
                # Time-Gate Logic
                if raw_zone == "Zone A":
                    if first_seen_A is None:
                        first_seen_A = now
                    elif (now - first_seen_A) >= TIME_GATE_SECONDS:
                        state = "Zone A" # Promoted after 4 seconds
                else:
                    first_seen_A = None
                    state = "Zone C" # Revert to Walkway if they leave the booth

            updated_tracks[matched_id] = {
                'centroid': (cx, cy),
                'bbox': det,
                'raw_zone': raw_zone,
                'first_seen_zone_A': first_seen_A,
                'state': state
            }
            
            # Remove track from active to prevent double matching
            if matched_id in self.active_tracks:
                del self.active_tracks[matched_id]

        self.active_tracks = updated_tracks
        return self.active_tracks

# ---------------------------------------------------------
# 4. MAIN APPLICATION LOOP
# ---------------------------------------------------------
def main():
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    detector = HailoObjectDetector(hef_path="yolov5m_person.hef")
    analyzer = TrackerAndAnalyzer()
    
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Cannot connect to RTSP stream.")
        return

    print("Starting AI Analytics with Time-Gate Filtering...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.infer(frame)
        tracked_objects = analyzer.update_tracks(detections)
        
        counts = {"Zone A": 0, "Zone C": 0}

        for track_id, data in tracked_objects.items():
            x1, y1, x2, y2, conf, cls = data['bbox']
            cx, cy = data['centroid']
            state = data['state']
            
            if state in counts:
                counts[state] += 1

            # Visual overlay mapping
            color = (0, 0, 255) if state == "Zone A" else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
            cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Polygons
        cv2.polylines(frame, [np.array(ZONE_C_COORDS, np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(ZONE_A_COORDS, np.int32)], True, (0, 0, 255), 2)

        # Write Telemetry
        point = (
            InfluxPoint("spatial_engagement")
            .tag("camera_id", "CM5_Cam_01")
            .field("engaged_leads", counts["Zone A"])
            .field("passing_traffic", counts["Zone C"])
        )
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

        # Dashboard UI
        cv2.putText(frame, f"Engaged (Zone A >4s): {counts['Zone A']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Walkway (Zone C): {counts['Zone C']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Hailo AI Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.close()

if __name__ == "__main__":
    main()
