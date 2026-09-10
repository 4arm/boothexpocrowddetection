import os
import time
import signal
import cv2
import numpy as np

# Monkey patch signal to avoid GStreamer crashing in threads
_original_signal = signal.signal
def fake_signal(sig, handler): pass
signal.signal = fake_signal

from flask import Flask, render_template, request, Response, jsonify
from processor import SimulatorProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
processor = None

@app.route('/')
def simulator():
    return render_template('simulator.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    global processor
    data = request.json
    people = data.get('people', [])
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bg_path = os.path.join(base_dir, 'static', 'bg.jpg')
    person_path = os.path.join(base_dir, 'static', 'human.png')
    
    bg = cv2.imread(bg_path)
    if bg is None:
        return jsonify({"success": False, "error": f"Background not found at {bg_path}"})
    
    person_img = cv2.imread(person_path, cv2.IMREAD_UNCHANGED)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'simulated.mp4')
    out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
    
    # Generate 5 seconds of video (150 frames)
    # Add slight wobble to people so tracker detects them as moving objects
    for frame_idx in range(150):
        frame = bg.copy()
        wobble_x = int(np.sin(frame_idx * 0.5) * 2)
        wobble_y = int(np.cos(frame_idx * 0.5) * 1)
        
        for p in people:
            w = int(p['w'])
            h = int(p['h'])
            x = int(p['x']) + wobble_x
            y = int(p['y']) + wobble_y
            
            if w <= 0 or h <= 0: continue
            
            p_resized = cv2.resize(person_img, (w, h))
            
            # Alpha blending
            y1, y2 = max(0, y), min(720, y + h)
            x1, x2 = max(0, x), min(1280, x + w)
            
            py1 = 0 if y >= 0 else -y
            py2 = h if y + h <= 720 else h - ((y + h) - 720)
            px1 = 0 if x >= 0 else -x
            px2 = w if x + w <= 1280 else w - ((x + w) - 1280)
            
            if y1 >= y2 or x1 >= x2: continue
            
            # Check if image has an alpha channel
            if p_resized.shape[2] == 4:
                alpha_s = p_resized[py1:py2, px1:px2, 3] / 255.0
                alpha_s = np.expand_dims(alpha_s, axis=2)
            else:
                alpha_s = np.ones((py2-py1, px2-px1, 1), dtype=np.float32)
                
            alpha_l = 1.0 - alpha_s
            
            frame[y1:y2, x1:x2, :3] = (alpha_s * p_resized[py1:py2, px1:px2, :3] +
                                       alpha_l * frame[y1:y2, x1:x2, :3])
                
        out.write(frame)
    out.release()
    
    if processor is not None:
        processor.stop()
    processor = SimulatorProcessor(filepath)
    
    return jsonify({"success": True})

@app.route('/player')
def player():
    return render_template('player.html')

def gen_frames():
    global processor
    while True:
        if processor is None or processor.is_finished:
            if processor and processor.is_finished:
                break
            time.sleep(0.1)
            continue
            
        frame = processor.process_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    global processor
    if processor is None:
        return jsonify({"status": "no_video"})
    if processor.is_finished:
        return jsonify({"status": "finished", "stats": processor.get_frontend_stats()})
    return jsonify({"status": "playing", "stats": processor.get_frontend_stats()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)

