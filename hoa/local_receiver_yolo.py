"""
🚦 HỆ THỐNG PHÁT HIỆN KẸT XE REAL-TIME
Tác giả: Traffic Monitor System
Mô tả: Hỗ trợ 2 chế độ:
       1. Import video từ máy local
       2. Nhận stream trực tiếp từ phone qua WebSocket
"""

import cv2
import numpy as np
import websocket
import threading
import time
from collections import deque, defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# ==================== CẤU HÌNH ====================
VPS_URL = "wss://traffic058.io.vn/receiver"

# Model YOLOv8
YOLO_MODEL = "yolov8n.pt"

# Các lớp xe cần nhận dạng
CUSTOM_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Bảng quy đổi xe máy tương đương
VEHICLE_EQUIV = {
    "motorcycle": 1,
    "car": 5, 
    "truck": 19,
    "bus": 17
}

# Ngưỡng confidence
CLASS_CONF_THRESHOLDS = {
    "motorcycle": 0.2,
    "car": 0.35,
    "truck": 0.35,
    "bus": 0.35
}

# Ngưỡng cảnh báo kẹt xe
THRESHOLD_COUNT = 15
THRESHOLD_SPEED = 5.0

# ==================== LỚP CHÍNH ====================
class TrafficMonitor:
    def __init__(self, mode="local", video_path=None):
        self.mode = mode  # "local" hoặc "stream"
        self.video_path = video_path
        self.frame_queue = []
        self.running = True
        self.cap = None
        
        # Load YOLOv8
        print("📦 Đang tải mô hình YOLOv8...")
        self.model = YOLO(YOLO_MODEL)
        print("✅ YOLOv8 đã sẵn sàng!")
        
        # Initialize DeepSORT
        print("📦 Đang tải DeepSORT tracker...")
        self.tracker = DeepSort(
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
        
        # ROI
        self.roi = None
        self.roi_length_m = 20.0
        self.meter_per_pixel = None
        
        # Tracking history
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_smoothed_speed = {}
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.total_frames = 0
        
        # UI
        self.instruction_text = ""
        self.instruction_color = (255, 255, 255)
        
        # Setup source
        if self.mode == "local":
            self.setup_local_video()
        else:
            self.setup_stream()
    
    # ==================== SETUP VIDEO LOCAL ====================
    def setup_local_video(self):
        """Mở video từ file local"""
        if not self.video_path or not os.path.exists(self.video_path):
            print(" File video không tồn tại!")
            self.running = False
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(" Không thể mở video!")
            self.running = False
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print("\n" + "="*70)
        print("✅ ĐÃ TẢI VIDEO THÀNH CÔNG!")
        print("="*70)
        print(f"📁 File: {os.path.basename(self.video_path)}")
        print(f"🎬 Tổng frames: {self.total_frames}")
        print(f"⏱️  FPS: {fps:.2f}")
        print(f"⏰ Thời lượng: {self.total_frames/fps:.2f}s")
        print("="*70 + "\n")
    
    # ==================== SETUP STREAM ====================
    def setup_stream(self):
        """Kết nối WebSocket stream"""
        print(f"🔌 Đang kết nối đến VPS: {VPS_URL}")
    
    # ==================== WEBSOCKET CALLBACKS ====================
    def on_message(self, ws, message):
        try:
            if len(message) < 10:
                return
            
            if message[0] != 0xFF or message[1] != 0xD8:
                return
            
            nparr = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.frame_queue.append(frame)
                if len(self.frame_queue) > 3:
                    self.frame_queue.pop(0)
        except Exception as e:
            print(f"️  Lỗi giải mã: {e}")
    
    def on_error(self, ws, error):
        print(f" Lỗi WebSocket: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("👋 WebSocket đã đóng")
        self.running = False
    
    def on_open(self, ws):
        print("\n" + "="*70)
        print("✅ ĐÃ KẾT NỐI THÀNH CÔNG VỚI VPS!")
        print("="*70)
        print("📱 Bây giờ hãy bắt đầu stream video từ điện thoại")
        print("="*70)
        self.print_instructions()
    
    # ==================== PRINT INSTRUCTIONS ====================
    def print_instructions(self):
        print("\n🎮 HƯỚNG DẪN SỬ DỤNG:")
        print("   ├─ [R] Chọn vùng quan sát (ROI)")
        print("   ├─ [C] Hiệu chỉnh tỉ lệ thực (Calibrate)")
        if self.mode == "local":
            print("   ├─ [SPACE] Tạm dừng/Tiếp tục")
            print("   ├─ [←/→] Tua lùi/tua tới 5 giây")
        print("   └─ [Q] Thoát chương trình")
        print("="*70 + "\n")
    
    # ==================== VẼ THÔNG BÁO ====================
    def draw_instruction(self, frame):
        """Vẽ thông báo hướng dẫn"""
        if self.instruction_text:
            overlay = frame.copy()
            h, w = frame.shape[:2]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            lines = self.instruction_text.split('\n')
            y_offset = 50
            max_width = 0
            
            for line in lines:
                (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_width = max(max_width, text_w)
            
            padding = 20
            box_h = len(lines) * 35 + padding * 2
            box_w = max_width + padding * 2
            box_x = (w - box_w) // 2
            box_y = y_offset - padding
            
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                         self.instruction_color, 2)
            
            for i, line in enumerate(lines):
                (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                text_x = (w - text_w) // 2
                text_y = y_offset + i * 35 + text_h
                
                cv2.putText(frame, line, (text_x + 2, text_y + 2), font, 
                           font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(frame, line, (text_x, text_y), font, 
                           font_scale, self.instruction_color, thickness)
    
    # ==================== ROI SELECTION ====================
    def select_roi(self, frame):
        """Chọn vùng quan sát"""
        print("\n" + "="*70)
        print("🎯 CHỌN VÙNG QUAN SÁT (ROI)")
        print("="*70)
        print("📍 Click chuột TRÁI để chọn các đỉnh")
        print("📍 Click chuột PHẢI để đóng polygon")
        print("✅ Nhấn ENTER để xác nhận |  ESC để hủy")
        print("="*70 + "\n")
        
        tmp = frame.copy()
        roi_points = []
        instruction = "Click chuot TRAI de chon cac dinh\nClick chuot PHAI de dong polygon\nENTER: Xac nhan | ESC: Huy"
        point_count = 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_points, tmp, point_count, instruction
            
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
                point_count += 1
                
                cv2.circle(tmp, (x, y), 7, (0, 255, 0), -1)
                cv2.circle(tmp, (x, y), 9, (255, 255, 255), 2)
                cv2.putText(tmp, str(point_count), (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if len(roi_points) > 1:
                    cv2.line(tmp, roi_points[-2], roi_points[-1], (0, 255, 255), 3)
                
                if len(roi_points) == 1:
                    instruction = f"Da chon {len(roi_points)} dinh\nChon them cac dinh khac"
                    print(f"✅ Đã chọn điểm {point_count}")
                elif len(roi_points) == 2:
                    instruction = f"Da chon {len(roi_points)} dinh\nChon it nhat 1 dinh nua"
                    print(f"✅ Đã chọn điểm {point_count}")
                else:
                    instruction = f"Da chon {len(roi_points)} dinh\nClick PHAI de dong, hoac chon them"
                    print(f"✅ Đã chọn điểm {point_count}")
                
            elif event == cv2.EVENT_RBUTTONDOWN and len(roi_points) >= 3:
                cv2.line(tmp, roi_points[-1], roi_points[0], (0, 255, 255), 3)
                cv2.polylines(tmp, [np.array(roi_points, np.int32)], True, (0, 0, 255), 3)
                instruction = f"Polygon da dong ({len(roi_points)} dinh)\nNhan ENTER de xac nhan"
                print(f"✅ Đã đóng polygon với {len(roi_points)} đỉnh")
            
            self.instruction_text = instruction
            self.instruction_color = (0, 255, 255)
            display = tmp.copy()
            self.draw_instruction(display)
            cv2.imshow("Chon ROI", display)
        
        cv2.namedWindow("Chon ROI")
        cv2.setMouseCallback("Chon ROI", mouse_callback)
        
        self.instruction_text = instruction
        self.instruction_color = (0, 255, 255)
        display = tmp.copy()
        self.draw_instruction(display)
        cv2.imshow("Chon ROI", display)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                if len(roi_points) >= 3:
                    print("✅ Đã xác nhận ROI")
                    break
                else:
                    print("️  Cần ít nhất 3 đỉnh!")
                    self.instruction_text = "CAN IT NHAT 3 DINH!\nChon them cac dinh"
                    self.instruction_color = (0, 0, 255)
                    display = tmp.copy()
                    self.draw_instruction(display)
                    cv2.imshow("Chon ROI", display)
            elif key == 27:
                roi_points = []
                print(" Đã hủy chọn ROI")
                break
        
        cv2.destroyWindow("Chon ROI")
        self.instruction_text = ""
        
        if len(roi_points) >= 3:
            self.roi = np.array(roi_points, np.int32)
            print(f"\n🎯 ROI đã được chọn với {len(self.roi)} đỉnh\n")
    
    # ==================== CALIBRATE ====================
    def calibrate_scale(self, frame):
        """Hiệu chỉnh tỉ lệ"""
        print("\n" + "="*70)
        print("📏 HIỆU CHỈNH TỈ LỆ THỰC")
        print("="*70)
        print("📍 Click 2 điểm có khoảng cách thực biết trước")
        print("✅ Nhập khoảng cách (mét) |  ESC để hủy")
        print("="*70 + "\n")
        
        tmp = frame.copy()
        pts = []
        instruction = "Click 2 diem co khoang cach thuc biet truoc"
        
        def mouse_cb(event, x, y, flags, param):
            nonlocal pts, tmp, instruction
            
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
                
                cv2.circle(tmp, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(tmp, (x, y), 10, (255, 255, 255), 2)
                
                label = "DIEM 1" if len(pts) == 1 else "DIEM 2"
                cv2.putText(tmp, label, (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(pts) == 1:
                    instruction = "Da chon DIEM 1\nChon DIEM 2"
                    print("✅ Đã chọn điểm 1")
                elif len(pts) == 2:
                    cv2.line(tmp, pts[0], pts[1], (0, 255, 255), 3)
                    
                    dist_px = np.linalg.norm(np.array(pts[1]) - np.array(pts[0]))
                    
                    mid_x = (pts[0][0] + pts[1][0]) // 2
                    mid_y = (pts[0][1] + pts[1][1]) // 2
                    cv2.putText(tmp, f"{dist_px:.1f} pixels", (mid_x, mid_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    instruction = f"Khoang cach: {dist_px:.1f} pixels\nNhap khoang cach thuc vao console"
                    print(f"✅ Đã chọn điểm 2")
                    print(f"📏 Khoảng cách: {dist_px:.1f} pixels")
                
                self.instruction_text = instruction
                self.instruction_color = (0, 255, 255)
                display = tmp.copy()
                self.draw_instruction(display)
                cv2.imshow("Hieu chinh ti le", display)
        
        cv2.namedWindow("Hieu chinh ti le")
        cv2.setMouseCallback("Hieu chinh ti le", mouse_cb)
        
        self.instruction_text = instruction
        self.instruction_color = (0, 255, 255)
        display = tmp.copy()
        self.draw_instruction(display)
        cv2.imshow("Hieu chinh ti le", display)
        
        while len(pts) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                pts = []
                print(" Đã hủy hiệu chỉnh")
                break
        
        cv2.destroyWindow("Hieu chinh ti le")
        self.instruction_text = ""
        
        if len(pts) == 2:
            p1, p2 = np.array(pts[0], float), np.array(pts[1], float)
            pixel_dist = float(np.linalg.norm(p2 - p1))
            
            print(f"📏 Khoảng cách: {pixel_dist:.2f} pixels")
            
            try:
                real_dist = float(input("➡️  Nhập khoảng cách thực (mét): "))
                
                if real_dist > 0:
                    self.meter_per_pixel = real_dist / pixel_dist
                    print("="*70)
                    print(f"✅ ĐÃ HIỆU CHỈNH!")
                    print(f"   1 pixel = {self.meter_per_pixel:.6f} mét")
                    print("="*70 + "\n")
            except:
                print(" Giá trị không hợp lệ!\n")
    
    # ==================== TÍNH VẬN TỐC ====================
    def calculate_speed(self, history):
        if len(history) >= 2:
            pt1 = history[-2][:2]
            pt2 = history[-1][:2]
            dt = history[-1][2] - history[-2][2]
            
            if dt > 0.001:
                dist_pixels = np.linalg.norm(np.array(pt2) - np.array(pt1))
                
                if self.meter_per_pixel:
                    dist_meters = dist_pixels * self.meter_per_pixel
                else:
                    if self.roi is not None:
                        roi_length_pixels = cv2.arcLength(self.roi, True)
                        self.meter_per_pixel = self.roi_length_m / roi_length_pixels
                        dist_meters = dist_pixels * self.meter_per_pixel
                    else:
                        dist_meters = dist_pixels * 0.05
                
                speed_kmh = (dist_meters / dt) * 3.6
                return min(max(speed_kmh, 0.0), 120.0)
        
        return 0.0
    
    # ==================== XỬ LÝ FRAME ====================
    def process_frame(self, frame):
        t_now = time.time()
        
        # Detection
        results = self.model.predict(frame, conf=0.25, iou=0.5, imgsz=640, verbose=False)
        
        detections = []
        r = results[0]
        
        if hasattr(r, 'boxes'):
            for box in r.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                if cls_name in CUSTOM_CLASSES:
                    conf = float(box.conf[0].cpu().numpy())
                    threshold = CLASS_CONF_THRESHOLDS.get(cls_name, 0.35)
                    
                    if conf >= threshold:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy.astype(int)
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls_name))
        
        # Tracking
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        in_roi_ids = set()
        in_roi_classes = []
        speeds_kmh = []
        
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            
            track_id = tr.track_id
            left, top, right, bottom = map(int, tr.to_ltrb())
            cx, cy = int((left + right) / 2), int((top + bottom) / 2)
            
            self.track_history[track_id].append((cx, cy, t_now))
            
            in_roi = (
                self.roi is not None and
                cv2.pointPolygonTest(self.roi, (cx, cy), False) >= 0
            )
            
            if in_roi:
                in_roi_ids.add(track_id)
                cls_name = tr.get_det_class() if hasattr(tr, 'get_det_class') else "unknown"
                in_roi_classes.append(cls_name)
                
                speed_kmh = self.calculate_speed(self.track_history[track_id])
                alpha = 0.3
                prev = self.track_smoothed_speed.get(track_id, speed_kmh)
                smooth = alpha * speed_kmh + (1 - alpha) * prev
                self.track_smoothed_speed[track_id] = smooth
                
                if smooth > 0.5:
                    speeds_kmh.append(smooth)
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id} {smooth:.1f}km/h",
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 1)
        
        # Stats
        count_in_roi = len(in_roi_ids)
        equiv_count = sum(VEHICLE_EQUIV.get(cls, 1.0) for cls in in_roi_classes)
        avg_speed_kmh = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0
        
        # Vẽ ROI
        if self.roi is not None:
            cv2.polylines(frame, [self.roi], True, (0, 0, 255), 3)
            for pt in self.roi:
                cv2.circle(frame, tuple(pt), 6, (0, 255, 255), -1)
                cv2.circle(frame, tuple(pt), 8, (255, 255, 255), 2)
        
        # Cảnh báo
        is_congested = (equiv_count > THRESHOLD_COUNT and avg_speed_kmh < THRESHOLD_SPEED)
        
        # Vẽ UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (5, 5), (450, 200), (255, 255, 255), 2)
        
        y_offset = 30
        cv2.putText(frame, f"Frame: {self.frame_count}/{self.total_frames if self.mode=='local' else '∞'}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Xe trong ROI: {count_in_roi}", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Tuong duong: {equiv_count:.1f} xe may", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"Van toc TB: {avg_speed_kmh:.2f} km/h", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        if is_congested:
            cv2.putText(frame, "! CANH BAO KET XE !", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "Hoat dong binh thuong", (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Help
        h, w = frame.shape[:2]
        help_y = h - 80
        cv2.rectangle(frame, (w - 250, help_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 250, help_y - 10), (w - 10, h - 10), (255, 255, 255), 1)
        
        cv2.putText(frame, "[R] Chon ROI", (w - 240, help_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "[C] Hieu chinh", (w - 240, help_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "[Q] Thoat", (w - 240, help_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    # ==================== RUN LOCAL VIDEO ====================
    def run_local(self):
        """Chạy xử lý video local"""
        cv2.namedWindow('Traffic Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Traffic Monitor', 1280, 720)
        
        paused = False
        
        print("🎥 Đang xử lý video...")
        self.print_instructions()
        
        while self.running:
            try:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("\n✅ Đã xử lý hết video!")
                        break
                    
                    self.frame_count += 1
                    
                    # Process
                    processed = self.process_frame(frame)
                    
                    # FPS
                    self.fps_counter += 1
                    if self.fps_counter % 10 == 0:
                        fps_end = time.time()
                        self.fps = 10 / (fps_end - self.fps_start_time)
                        self.fps_start_time = fps_end
                    
                    cv2.imshow('Traffic Monitor', processed)
                    
                    if self.frame_count % 30 == 0:
                        progress = (self.frame_count / self.total_frames) * 100
                        print(f"✅ {self.frame_count}/{self.total_frames} ({progress:.1f}%) | FPS: {self.fps:.1f}")
                else:
                    # Paused - chỉ hiển thị frame hiện tại
                    cv2.imshow('Traffic Monitor', processed)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n🛑 Đang dừng...")
                    break
                elif key == ord('r') or key == ord('R'):
                    ret, frame = self.cap.read()
                    if ret:
                        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                        print("\n🎯 Bắt đầu chọn ROI...")
                        self.select_roi(frame)
                elif key == ord('c') or key == ord('C'):
                    ret, frame = self.cap.read()
                    if ret:
                        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                        print("\n📏 Bắt đầu hiệu chỉnh...")
                        self.calibrate_scale(frame)
                elif key == 32:  # SPACE
                    paused = not paused
                    status = "⏸️  TẠM DỪNG" if paused else "▶️  TIẾP TỤC"
                    print(f"\n{status}")
                elif key == 81:  # Left arrow
                    current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    new_pos = max(0, current - fps * 5)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    print(f"⏪ Tua lùi 5 giây")
                elif key == 83:  # Right arrow
                    current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    new_pos = min(self.total_frames, current + fps * 5)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    print(f"⏩ Tua tới 5 giây")
                    
            except KeyboardInterrupt:
                print("\n🛑 Nhận Ctrl+C...")
                break
            except Exception as e:
                print(f" Lỗi: {e}")
                import traceback
                traceback.print_exc()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("👋 HOÀN TẤT XỬ LÝ VIDEO")
        print(f"📊 Tổng frames: {self.frame_count}")
        print("="*70 + "\n")
    
    # ==================== RUN STREAM ====================
    def run_stream(self):
        """Chạy xử lý stream từ phone"""
        ws = websocket.WebSocketApp(
            VPS_URL,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        cv2.namedWindow('Traffic Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Traffic Monitor', 1280, 720)
        
        print("🎥 Đang chờ video stream...")
        
        while self.running:
            try:
                if len(self.frame_queue) > 0:
                    frame = self.frame_queue.pop(0)
                    self.frame_count += 1
                    
                    processed = self.process_frame(frame)
                    
                    self.fps_counter += 1
                    if self.fps_counter % 10 == 0:
                        fps_end = time.time()
                        self.fps = 10 / (fps_end - self.fps_start_time)
                        self.fps_start_time = fps_end
                    
                    cv2.imshow('Traffic Monitor', processed)
                    
                    if self.frame_count % 30 == 0:
                        print(f"✅ Đã xử lý {self.frame_count} frames | FPS: {self.fps:.1f}")
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n🛑 Đang dừng...")
                    break
                elif key == ord('r') or key == ord('R'):
                    if len(self.frame_queue) > 0:
                        print("\n🎯 Bắt đầu chọn ROI...")
                        self.select_roi(self.frame_queue[0])
                    else:
                        print("\n️  Chưa có video stream!")
                elif key == ord('c') or key == ord('C'):
                    if len(self.frame_queue) > 0:
                        print("\n📏 Bắt đầu hiệu chỉnh...")
                        self.calibrate_scale(self.frame_queue[0])
                    else:
                        print("\n️  Chưa có video stream!")
                    
            except KeyboardInterrupt:
                print("\n🛑 Nhận Ctrl+C...")
                break
            except Exception as e:
                print(f" Lỗi: {e}")
                import traceback
                traceback.print_exc()
        
        ws.close()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print("👋 STREAM ĐÃ DỪNG")
        print(f"📊 Tổng frames: {self.frame_count}")
        print("="*70 + "\n")
    
    # ==================== RUN ====================
    def run(self):
        """Chạy chương trình"""
        if self.mode == "local":
            self.run_local()
        else:
            self.run_stream()


# ==================== GUI CHỌN CHỂ ĐỘ ====================
class ModeSelectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚦 Traffic Monitor - Chọn chế độ")
        self.root.geometry("500x300")
        self.root.resizable(False, False)
        
        self.mode = None
        self.video_path = None
        
        # Title
        title = tk.Label(
            self.root,
            text="🚦 HỆ THỐNG GIÁM SÁT GIAO THÔNG",
            font=("Arial", 16, "bold"),
            fg="#2C3E50"
        )
        title.pack(pady=20)
        
        subtitle = tk.Label(
            self.root,
            text="Phát hiện kẹt xe bằng YOLOv8 + DeepSORT",
            font=("Arial", 10),
            fg="#7F8C8D"
        )
        subtitle.pack(pady=5)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=30)
        
        # Button 1: Import Video
        btn1 = tk.Button(
            btn_frame,
            text="📁 Import Video từ Máy",
            font=("Arial", 12, "bold"),
            bg="#3498DB",
            fg="white",
            width=25,
            height=2,
            command=self.select_local_video,
            cursor="hand2"
        )
        btn1.pack(pady=10)
        
        # Button 2: Stream
        btn2 = tk.Button(
            btn_frame,
            text="📱 Stream từ Điện thoại",
            font=("Arial", 12, "bold"),
            bg="#2ECC71",
            fg="white",
            width=25,
            height=2,
            command=self.select_stream,
            cursor="hand2"
        )
        btn2.pack(pady=10)
        
        # Footer
        footer = tk.Label(
            self.root,
            text="Powered by YOLOv8 + DeepSORT",
            font=("Arial", 8),
            fg="#95A5A6"
        )
        footer.pack(side=tk.BOTTOM, pady=10)
        
        self.root.mainloop()
    
    def select_local_video(self):
        """Chọn video từ máy"""
        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.mode = "local"
            self.video_path = file_path
            self.root.destroy()
        else:
            messagebox.showwarning("Cảnh báo", "Bạn chưa chọn file video!")
    
    def select_stream(self):
        """Chọn chế độ stream"""
        result = messagebox.askyesno(
            "Xác nhận",
            "Bạn sẽ stream video từ điện thoại.\n\n"
            "Đảm bảo:\n"
            "1. VPS server đang chạy\n"
            "2. Caddy HTTPS đang chạy\n"
            "3. Điện thoại sẽ truy cập: https://traffic058.io.vn\n\n"
            "Tiếp tục?"
        )
        
        if result:
            self.mode = "stream"
            self.video_path = None
            self.root.destroy()
    
    def get_selection(self):
        """Lấy lựa chọn"""
        return self.mode, self.video_path


# ==================== MAIN ====================
def main():
    print("\n" + "="*70)
    print("🚦 HỆ THỐNG GIÁM SÁT GIAO THÔNG - PHÁT HIỆN KẸT XE")
    print("="*70)
    print("📡 Hỗ trợ 2 chế độ:")
    print("   1. Import video từ máy local")
    print("   2. Stream trực tiếp từ điện thoại")
    print("="*70 + "\n")
    
    try:
        # Hiển thị GUI chọn chế độ
        gui = ModeSelectionGUI()
        mode, video_path = gui.get_selection()
        
        if mode is None:
            print("👋 Người dùng hủy chương trình")
            return
        
        # Khởi tạo và chạy
        monitor = TrafficMonitor(mode=mode, video_path=video_path)
        
        if monitor.running:
            monitor.run()
        
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
    except Exception as e:
        print(f"\n Lỗi nghiêm trọng: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()