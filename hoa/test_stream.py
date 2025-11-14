
import cv2
import time
import numpy as np
from datetime import datetime
import requests
import subprocess
import platform
import os
import sys

class StreamTester:
    def __init__(self, stream_url):
        """
        Args:
            stream_url: HLS stream URL (http://VPS_IP/hls/stream.m3u8)
                       hoặc RTMP URL (rtmp://VPS_IP:1935/live/stream)
        """
        self.stream_url = stream_url
        self.is_hls = stream_url.endswith('.m3u8')
        self.is_rtmp = stream_url.startswith('rtmp://')
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None
        self.latency_samples = []
        self.fps_samples = []
        
    def check_ffmpeg_installed(self):
        """Kiểm tra FFmpeg có được cài đặt không"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(" FFmpeg đã cài đặt")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        print(" FFmpeg chưa cài đặt!")
        print("\nHướng dẫn cài đặt FFmpeg:")
        
        if platform.system() == 'Windows':
            print("1. Download: https://www.gyan.dev/ffmpeg/builds/")
            print("2. Extract và thêm vào PATH")
            print("3. Hoặc: choco install ffmpeg")
        elif platform.system() == 'Darwin':  # macOS
            print("brew install ffmpeg")
        else:  # Linux
            print("sudo apt install ffmpeg")
        
        return False
    
    def test_vps_connection(self):
        """Test kết nối đến VPS"""
        print("\n" + "="*60)
        print("KIỂM TRA KẾT NỐI VPS")
        print("="*60)
        
        # Extract VPS IP/domain
        if self.is_hls:
            base_url = self.stream_url.split('/hls')[0]
        elif self.is_rtmp:
            # rtmp://ip:port/app/stream -> http://ip
            parts = self.stream_url.replace('rtmp://', '').split('/')
            base_url = f"http://{parts[0].split(':')[0]}"
        else:
            print(" URL không hợp lệ")
            return False
        
        # Test HTTP connection
        try:
            print(f"Đang test kết nối đến: {base_url}")
            response = requests.get(f"{base_url}/health", timeout=5)
            
            if response.status_code == 200:
                print(f" VPS phản hồi: {response.text.strip()}")
                print(f" Response time: {response.elapsed.total_seconds():.3f}s")
                return True
            else:
                print(f" VPS trả về status code: {response.status_code}")
                return False
                
        except requests.ConnectionError:
            print(f" Không thể kết nối đến VPS")
            print(f"  Kiểm tra:")
            print(f"  - VPS có đang chạy không?")
            print(f"  - IP address đúng chưa?")
            print(f"  - Firewall có chặn không?")
            return False
        except requests.Timeout:
            print(f" Timeout khi kết nối (>5s)")
            return False
        except Exception as e:
            print(f" Lỗi: {e}")
            return False
    
    def test_stream_availability(self):
        """Test xem stream có sẵn không"""
        print("\n" + "="*60)
        print("KIỂM TRA STREAM")
        print("="*60)
        
        if self.is_hls:
            try:
                print(f"Đang kiểm tra HLS playlist: {self.stream_url}")
                response = requests.get(self.stream_url, timeout=10)
                
                if response.status_code == 200:
                    print(" HLS playlist có sẵn")
                    
                    # Parse playlist
                    lines = response.text.split('\n')
                    ts_files = [l for l in lines if l.endswith('.ts')]
                    
                    if ts_files:
                        print(f" Tìm thấy {len(ts_files)} segments")
                        print(f"  First segment: {ts_files[0]}")
                        return True
                    else:
                        print(" Playlist không có segments (chưa có stream)")
                        print("  → Bật Flutter app và bắt đầu streaming")
                        return False
                else:
                    print(f" Không tìm thấy playlist (HTTP {response.status_code})")
                    return False
                    
            except Exception as e:
                print(f" Lỗi khi kiểm tra HLS: {e}")
                return False
        
        elif self.is_rtmp:
            print("RTMP stream - sẽ test khi mở video")
            return True
        
        return False
    
    def measure_latency(self, frame):
        """
        Đo độ trễ bằng cách so sánh timestamp
        (Chỉ ước lượng - cần sync clock giữa điện thoại và PC để chính xác)
        """
        # Placeholder - trong thực tế cần embed timestamp trong stream
        return 0
    
    def calculate_fps(self):
        """Tính FPS hiện tại"""
        if self.start_time is None:
            return 0
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0
    
    def draw_stats(self, frame):
        """Vẽ thống kê lên frame"""
        h, w = frame.shape[:2]
        
        # Background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Calculate stats
        fps = self.calculate_fps()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Draw text
        y_pos = 40
        stats = [
            f"Stream: {self.stream_url.split('/')[-1]}",
            f"Resolution: {w}x{h}",
            f"FPS: {fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Dropped: {self.dropped_frames}",
            f"Time: {elapsed:.1f}s",
        ]
        
        for stat in stats:
            cv2.putText(frame, stat, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Local Time: {current_time}", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def test_stream_playback(self):
        """Test phát video từ stream"""
        print("\n" + "="*60)
        print("BẮT ĐẦU PHÁT STREAM")
        print("="*60)
        print("Nhấn 'q' để thoát")
        print("Nhấn 's' để screenshot")
        print("Nhấn 'r' để reset statistics")
        print("="*60 + "\n")
        
        # Open stream
        cap = cv2.VideoCapture(self.stream_url)
        
        if not cap.isOpened():
            print(" Không thể mở stream!")
            print("\nGợi ý khắc phục:")
            print("1. Kiểm tra Flutter app đã bắt đầu streaming chưa")
            print("2. Kiểm tra URL có đúng không")
            print("3. Thử với VLC: vlc", self.stream_url)
            return False
        
        print(" Đã mở stream thành công!")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Stream properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {stream_fps}")
        
        self.start_time = time.time()
        last_frame_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                current_time = time.time()
                
                if not ret:
                    self.dropped_frames += 1
                    
                    # Check if stream ended or network issue
                    if self.dropped_frames > 30:
                        print("\n Quá nhiều frames bị drop - stream có thể đã dừng")
                        break
                    
                    continue
                
                self.frame_count += 1
                
                # Calculate frame interval
                frame_interval = current_time - last_frame_time
                last_frame_time = current_time
                
                # Draw statistics
                frame_with_stats = self.draw_stats(frame)
                
                # Show frame
                cv2.imshow('Stream Test - Press Q to quit', frame_with_stats)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n Đã dừng bởi người dùng")
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f" Đã lưu screenshot: {filename}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.dropped_frames = 0
                    self.start_time = time.time()
                    print(" Đã reset statistics")
                
                # Warning for low FPS
                actual_fps = self.calculate_fps()
                if actual_fps < 15 and self.frame_count > 30:
                    print(f" FPS thấp: {actual_fps:.1f} - kiểm tra băng thông mạng")
        
        except KeyboardInterrupt:
            print("\n Đã dừng bởi Ctrl+C")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats()
        
        return True
    
    def print_final_stats(self):
        """In thống kê cuối cùng"""
        print("\n" + "="*60)
        print("THỐNG KÊ CUỐI CÙNG")
        print("="*60)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        drop_rate = (self.dropped_frames / (self.frame_count + self.dropped_frames) * 100) if (self.frame_count + self.dropped_frames) > 0 else 0
        
        print(f"Tổng thời gian: {elapsed:.1f}s")
        print(f"Tổng frames: {self.frame_count}")
        print(f"Frames dropped: {self.dropped_frames} ({drop_rate:.2f}%)")
        print(f"FPS trung bình: {avg_fps:.1f}")
        
        # Quality assessment
        print("\nĐánh giá:")
        if avg_fps >= 25 and drop_rate < 5:
            print(" XUẤT SẮC - Stream rất ổn định")
        elif avg_fps >= 20 and drop_rate < 10:
            print(" TỐT - Stream khá ổn định")
        elif avg_fps >= 15 and drop_rate < 20:
            print(" TRUNG BÌNH - Có thể cải thiện")
        else:
            print(" KÉM - Cần kiểm tra lại cấu hình")
        
        print("="*60)
    
    def run_full_test(self):
        """Chạy toàn bộ test"""
        print("\n" + "="*60)
        print("STREAM TESTING TOOL")
        print("="*60)
        print(f"Stream URL: {self.stream_url}")
        print(f"Stream type: {'HLS' if self.is_hls else 'RTMP' if self.is_rtmp else 'Unknown'}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print("="*60)
        
        # Test 1: Check FFmpeg
        if not self.check_ffmpeg_installed():
            print("\n Cảnh báo: Không có FFmpeg, một số stream có thể không hoạt động")
            response = input("Tiếp tục? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Test 2: VPS connection
        if not self.test_vps_connection():
            print("\n Không thể kết nối đến VPS")
            return
        
        # Test 3: Stream availability
        if not self.test_stream_availability():
            print("\n Stream chưa sẵn sàng")
            response = input("Thử mở stream anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Test 4: Playback
        self.test_stream_playback()

def main():
    print("="*60)
    print("STREAM TESTING TOOL")
    print("Test stream từ Flutter app qua VPS về máy local")
    print("="*60)
    
    # Prompt for stream URL
    print("\nNhập URL của stream:")
    print("  HLS: http://VPS_IP/hls/stream.m3u8")
    print("  RTMP: rtmp://VPS_IP:1935/live/stream")
    
    stream_url = input("\nStream URL: ").strip()
    
    if not stream_url:
        print(" URL không được để trống!")
        return
    
    # Validate URL
    if not (stream_url.startswith('http://') or 
            stream_url.startswith('https://') or 
            stream_url.startswith('rtmp://')):
        print(" URL không hợp lệ! Phải bắt đầu với http://, https://, hoặc rtmp://")
        return
    
    # Create tester and run
    tester = StreamTester(stream_url)
    tester.run_full_test()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Đã thoát chương trình")
    except Exception as e:
        print(f"\n Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()