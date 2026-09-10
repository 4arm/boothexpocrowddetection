import json
import logging

from flask import Flask, render_template, Response, jsonify, request
from influxdb_client import InfluxDBClient

from settings import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, CAMERA_ID, RTSP_URL
from camera import VideoCamera

app = Flask(__name__)

video_camera = VideoCamera(RTSP_URL)


@app.route('/')
def index():
    return render_template('index.html')


def generate_frames(camera):
    while True:
        frame = camera.process_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(video_camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_stats', methods=['GET'])
def get_stats():
    return jsonify(video_camera.get_frontend_stats())


@app.route('/get_config', methods=['GET'])
def get_config():
    try:
        with open('config.json', 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        logging.warning("Failed to load config: %s", e)
        return jsonify({"zones": {}})


@app.route('/save_config', methods=['POST'])
def save_config():
    data = request.json
    try:
        with open('config.json', 'w') as f:
            json.dump(data, f)
        video_camera.load_config()
        return jsonify({"status": "success", "message": "Configuration saved & AI regions dynamically updated."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_report', methods=['GET'])
def get_report():
    """Queries InfluxDB to generate the dynamic daily report table."""
    try:
        # Context manager ensures the client is closed even on errors
        with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
            query_api = client.query_api()

            flux_query = f"""
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: -30d)
              |> filter(fn: (r) => r["_measurement"] == "zone_total")
              |> filter(fn: (r) => r["camera_id"] == "{CAMERA_ID}")
              |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
            """
            tables = query_api.query(flux_query)

            report = {}
            all_zones_found = set()

            for table in tables:
                for record in table.records:
                    date_str = record.get_time().strftime('%Y-%m-%d')
                    zone = record.values.get("zone")
                    field = record.get_field()
                    val = int(record.get_value() or 0)

                    all_zones_found.add(zone)

                    if date_str not in report:
                        report[date_str] = {}

                    if zone not in report[date_str]:
                        report[date_str][zone] = {"crowd_1": 0, "crowd_2": 0, "crowd_3": 0, "total": 0}

                    if field in ["crowd_1", "crowd_2", "crowd_3", "crowd_4"]:
                        report[date_str][zone][field] += val
                        report[date_str][zone]["total"] += val

            return jsonify({
                "dates": report,
                "zones": list(all_zones_found)
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
