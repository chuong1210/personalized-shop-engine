"""
🚦 BACKEND API CHO HỆ THỐNG GIÁM SÁT GIAO THÔNG
Flask API Server với YOLOv8 + DeepSORT + Real-time SSE
"""

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import numpy as np
import json
import base64
import time
from collections import deque, defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading

app = Flask(__name__)
CORS(app)

# ==================== CẤU HÌNH ====================
YOLO_MODEL = "bestV3.pt"  # Hoặc "yolov8n.pt"
CUSTOM_CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]

CLASS_CONF_THRESHOLDS = {
    "Motorcycle": 0.2,
    "Car": 0.35,
    "Truck": 0.35,
    "Ambulance": 0.35,
    "Bus": 0.35
}

# ==================== LOAD MODEL ====================
print("📦 Đang tải mô hình YOLOv8...")
try:
    model = YOLO(YOLO_MODEL)
    print("✅ YOLOv8 đã sẵn sàng!")
except Exception as e:
    print(f" Lỗi tải model: {e}")
    print("️  Sử dụng model mặc định yolov8n.pt")
    model = YOLO("yolov8n.pt")

# ==================== TRACKER ====================
print("📦 Đang tải DeepSORT tracker...")
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=None,
    embedder="mobilenet",
    half=True,
    embedder_gpu=False
)
print("✅ DeepSORT đã sẵn sàng!")

# ==================== GLOBAL TRACKING DATA ====================
track_history = defaultdict(lambda: deque(maxlen=30))
track_smoothed_speed = {}

# Jam detection state
jam_start_time = None
jam_duration = 0.0
is_currently_jammed = False

# ==================== HELPER FUNCTIONS ====================

def reset_tracking_data():
    """Reset tracking data khi bắt đầu video mới"""
    global track_history, track_smoothed_speed, jam_start_time, jam_duration, is_currently_jammed
    track_history.clear()
    track_smoothed_speed.clear()
    jam_start_time = None
    jam_duration = 0.0
    is_currently_jammed = False

def point_in_polygon(point, polygon):
    """Check if point is inside polygon"""
    x, y = point
    result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
    return result >= 0

def calculate_speed(history, meter_per_pixel, roi_length_m, roi_polygon):
    """Calculate vehicle speed"""
    if len(history) >= 2:
        pt1 = history[-2][:2]
        pt2 = history[-1][:2]
        dt = history[-1][2] - history[-2][2]
        
        if dt > 0.001:
            dist_pixels = np.linalg.norm(np.array(pt2) - np.array(pt1))
            
            if meter_per_pixel:
                dist_meters = dist_pixels * meter_per_pixel
            else:
                roi_length_pixels = cv2.arcLength(roi_polygon, True)
                meter_per_pixel_est = roi_length_m / roi_length_pixels
                dist_meters = dist_pixels * meter_per_pixel_est
            
            speed_kmh = (dist_meters / dt) * 3.6
            return min(max(speed_kmh, 0.0), 120.0)
    
    return 0.0

def update_jam_status(is_jam_condition, current_time):
    """
    Cập nhật trạng thái kẹt xe dựa trên thời gian liên tục
    Chỉ báo kẹt xe khi điều kiện kẹt xe kéo dài > ngưỡng
    """
    global jam_start_time, jam_duration, is_currently_jammed
    
    if is_jam_condition:
        if jam_start_time is None:
            # Bắt đầu đếm thời gian kẹt xe
            jam_start_time = current_time
            jam_duration = 0.0
            is_currently_jammed = False
        else:
            # Tính thời gian kẹt liên tục
            jam_duration = current_time - jam_start_time
            # Chỉ set is_jammed = True khi vượt ngưỡng
            # (ngưỡng sẽ check ở frontend)
    else:
        # Reset khi không còn kẹt
        jam_start_time = None
        jam_duration = 0.0
        is_currently_jammed = False
    
    return jam_duration

def encode_image(frame, quality=85):
    """Encode frame to base64 with quality control"""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f" Encode error: {e}")
        return None

def process_frame_detection(frame, roi_points, settings, meter_per_pixel=None):
    """Process single frame with detection and tracking"""
    global track_history, track_smoothed_speed
    
    t_now = time.time()
    
    # Parse ROI
    roi_polygon = np.array(roi_points, np.int32) if roi_points else None
    vehicle_equiv = settings.get('vehicleEquiv', {
        'Motorcycle': 1, 'Car': 5, 'Truck': 19, 'Bus': 17, 'Ambulance': 10
    })
    
    # YOLOv8 detection
    results = model.predict(
        frame,
        conf=0.25,
        iou=0.5,
        imgsz=640,
        verbose=False
    )
    
    # Parse detections
    detections = []
    r = results[0]
    
    if hasattr(r, 'boxes'):
        for box in r.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]
            
            if cls_name in CUSTOM_CLASSES:
                conf = float(box.conf[0].cpu().numpy())
                threshold = CLASS_CONF_THRESHOLDS.get(cls_name, 0.35)
                
                if conf >= threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.astype(int)
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_name))
    
    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Process tracks
    in_roi_ids = set()
    in_roi_classes = []
    speeds_kmh = []
    
    for tr in tracks:
        if not tr.is_confirmed():
            continue
        
        track_id = tr.track_id
        left, top, right, bottom = map(int, tr.to_ltrb())
        cx, cy = int((left + right) / 2), int((top + bottom) / 2)
        
        # Update history
        track_history[track_id].append((cx, cy, t_now))
        
        # Check if in ROI
        in_roi = False
        if roi_polygon is not None:
            in_roi = point_in_polygon((cx, cy), roi_polygon)
        
        if in_roi:
            in_roi_ids.add(track_id)
            cls_name = tr.get_det_class() if hasattr(tr, 'get_det_class') else "Unknown"
            in_roi_classes.append(cls_name)
            
            # Calculate speed
            speed_kmh = calculate_speed(
                track_history[track_id],
                meter_per_pixel,
                settings.get('roiLength', 200),
                roi_polygon
            )
            
            # Smooth speed with exponential moving average
            alpha = 0.3
            prev = track_smoothed_speed.get(track_id, speed_kmh)
            smooth = alpha * speed_kmh + (1 - alpha) * prev
            track_smoothed_speed[track_id] = smooth
            
            if smooth > 0.5:
                speeds_kmh.append(smooth)
            
            # Draw bounding box (green for in-ROI)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label with background
            label = f"ID:{track_id} {smooth:.1f}km/h"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Background rectangle
            cv2.rectangle(frame, (left, top - text_h - 8), (left + text_w + 4, top), (0, 255, 0), -1)
            cv2.putText(frame, label, (left + 2, top - 4), font, font_scale, (0, 0, 0), thickness)
        else:
            # Draw gray box for out-of-ROI vehicles
            cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 1)
    
    # Draw ROI polygon
    if roi_polygon is not None:
        # Fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], (0, 255, 255))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Border
        cv2.polylines(frame, [roi_polygon], True, (0, 255, 255), 3)
        
        # Points
        for i, pt in enumerate(roi_polygon):
            cv2.circle(frame, tuple(pt), 8, (0, 255, 255), -1)
            cv2.circle(frame, tuple(pt), 10, (255, 255, 255), 2)
            
            # Number label with background
            label = str(i + 1)
            cv2.circle(frame, (pt[0] + 20, pt[1] - 20), 15, (0, 255, 255), -1)
            cv2.circle(frame, (pt[0] + 20, pt[1] - 20), 15, (255, 255, 255), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
            cv2.putText(frame, label, (pt[0] + 20 - text_w//2, pt[1] - 15), 
                       font, 0.6, (0, 0, 0), 2)
    
    # Calculate stats
    count_in_roi = len(in_roi_ids)
    equiv_count = sum(vehicle_equiv.get(cls, 1.0) for cls in in_roi_classes)
    avg_speed_kmh = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0
    
    # Check jam condition
    threshold_count = settings.get('thresholdCount', 30)
    threshold_speed = settings.get('thresholdSpeed', 5)
    is_jam_condition = (equiv_count > threshold_count and avg_speed_kmh < threshold_speed)
    
    # Update jam status with duration tracking
    jam_dur = update_jam_status(is_jam_condition, t_now)
    
    # Draw info overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (10, 10), (450, 230), (0, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 40
    line_height = 35
    
    cv2.putText(frame, f"Xe trong ROI: {count_in_roi}", (20, y_offset),
                font, 0.7, (0, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(frame, f"Tuong duong: {equiv_count:.1f} xe may", (20, y_offset),
                font, 0.7, (0, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(frame, f"Van toc TB: {avg_speed_kmh:.2f} km/h", (20, y_offset),
                font, 0.7, (0, 255, 255), 2)
    y_offset += line_height
    
    if is_jam_condition:
        cv2.putText(frame, f"Thoi gian ket: {jam_dur:.1f}s", (20, y_offset),
                    font, 0.7, (255, 165, 0), 2)
        y_offset += line_height
    
    # Status message
    jam_threshold = settings.get('jamDurationThreshold', 10)
    if is_jam_condition and jam_dur >= jam_threshold:
        cv2.putText(frame, "! CANH BAO KET XE !", (20, y_offset),
                    font, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "Hoat dong binh thuong", (20, y_offset),
                    font, 0.8, (0, 255, 0), 2)
    
    return {
        'vehicle_count': count_in_roi,
        'equivalent_count': equiv_count,
        'avg_speed': avg_speed_kmh,
        'is_jammed': is_jam_condition,
        'jam_duration': jam_dur,
        'frame': encode_image(frame, quality=85)
    }

# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'tracker_loaded': tracker is not None
    })

@app.route('/process_video', methods=['POST'])
def process_video():
    """Process entire video (for local mode) - Real-time SSE streaming"""
    try:
        # Reset tracking data for new video
        reset_tracking_data()
        
        # Get video file
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        video_file = request.files['video']
        
        # Save to temp file
        temp_path = '/tmp/upload_video.mp4'
        video_file.save(temp_path)
        
        # Get parameters
        roi_points_str = request.form.get('roi_points', '[]')
        roi_points = json.loads(roi_points_str)
        roi_points = [(int(p['x']), int(p['y'])) for p in roi_points]
        
        meter_per_pixel_str = request.form.get('meter_per_pixel', 'null')
        meter_per_pixel = None if meter_per_pixel_str == 'null' else float(meter_per_pixel_str)
        
        settings_str = request.form.get('settings', '{}')
        settings = json.loads(settings_str)
        
        # Generator for SSE
        @stream_with_context
        def generate():
            cap = cv2.VideoCapture(temp_path)
            frame_count = 0
            fps_start = time.time()
            fps_counter = 0
            current_fps = 0.0
            
            print(f"\n🎬 Bắt đầu xử lý video: {temp_path}")
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("✅ Đã xử lý xong video!")
                        break
                    
                    frame_count += 1
                    fps_counter += 1
                    
                    # Process frame
                    start_time = time.time()
                    result = process_frame_detection(frame, roi_points, settings, meter_per_pixel)
                    process_time = time.time() - start_time
                    
                    # Calculate FPS every 10 frames
                    if fps_counter >= 10:
                        current_fps = 10.0 / (time.time() - fps_start)
                        fps_start = time.time()
                        fps_counter = 0
                    
                    result['fps'] = current_fps
                    result['frame_number'] = frame_count
                    result['process_time'] = process_time
                    
                    # Send as SSE
                    yield f"data: {json.dumps(result)}\n\n"
                    
                    # Log progress
                    if frame_count % 30 == 0:
                        print(f"⏳ Frame {frame_count} | FPS: {current_fps:.1f} | Process: {process_time*1000:.1f}ms")
            
            finally:
                cap.release()
                print(f"🏁 Đã xử lý {frame_count} frames")
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        print(f" Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process single frame (for stream mode)"""
    try:
        # Get frame
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        nparr = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400
        
        # Get parameters
        roi_points_str = request.form.get('roi_points', '[]')
        roi_points = json.loads(roi_points_str)
        roi_points = [(int(p['x']), int(p['y'])) for p in roi_points]
        
        meter_per_pixel_str = request.form.get('meter_per_pixel', 'null')
        meter_per_pixel = None if meter_per_pixel_str == 'null' else float(meter_per_pixel_str)
        
        settings_str = request.form.get('settings', '{}')
        settings = json.loads(settings_str)
        
        # Process frame
        start_time = time.time()
        result = process_frame_detection(frame, roi_points, settings, meter_per_pixel)
        fps = 1.0 / (time.time() - start_time)
        result['fps'] = fps
        
        return jsonify(result)
        
    except Exception as e:
        print(f" Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚦 BACKEND API SERVER - HỆ THỐNG GIÁM SÁT GIAO THÔNG")
    print("="*70)
    print("📡 Server đang chạy tại: http://localhost:5000")
    print("🔧 Endpoints:")
    print("   - GET  /health         : Health check")
    print("   - POST /process_frame  : Xử lý frame đơn lẻ (stream mode)")
    print("   - POST /process_video  : Xử lý video real-time (SSE streaming)")
    print("="*70)
    print("✨ Tính năng mới:")
    print("   - Real-time SSE streaming (không đợi xử lý xong)")
    print("   - Tracking thời gian kẹt xe liên tục")
    print("   - Tối ưu FPS và hiển thị mượt mà")
    print("   - Vẽ ROI và calibration đẹp hơn")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)