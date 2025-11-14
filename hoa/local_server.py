import cv2
import numpy as np
import websocket
import threading

VPS_URL = "wss://traffic058.io.vn/receiver"  # Hoặc ws://130.94.7.8:8080/receiver nếu không dùng HTTPS

class VideoReceiver:
    def __init__(self):
        self.frame_queue = []
        self.frame_count = 0
        self.running = True
        print(f"🔌 Connecting to VPS: {VPS_URL}")
        
    def on_message(self, ws, message):
        try:
            # Debug: Kiểm tra JPEG header
            if len(message) < 10:
                print(f"️  Message too short: {len(message)} bytes")
                return
                
            # Kiểm tra JPEG signature (FF D8 FF)
            if message[0] != 0xFF or message[1] != 0xD8:
                print(f"️  Invalid JPEG header: {message[:4].hex()}")
                return
            
            # Decode JPEG từ WebSocket binary message
            nparr = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.frame_queue.append(frame)
                
                # Giữ tối đa 30 frames trong queue
                if len(self.frame_queue) > 30:
                    self.frame_queue.pop(0)
            else:
                print(f"️  Failed to decode JPEG ({len(message)} bytes)")
                    
        except Exception as e:
            print(f"️  Decode error: {e}")
    
    def on_error(self, ws, error):
        print(f" WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("👋 WebSocket closed")
        self.running = False
    
    def on_open(self, ws):
        print("✅ Connected to VPS!")
        print("📱 Now start streaming from your phone")
        
    def receive_and_display(self):
        # Kết nối WebSocket trong thread riêng
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
        
        # Hiển thị video trong main thread
        cv2.namedWindow('WebRTC Stream', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('WebRTC Stream', 640, 480)
        
        print("🎥 Waiting for video frames...")
        
        while self.running:
            try:
                if len(self.frame_queue) > 0:
                    frame = self.frame_queue.pop(0)
                    self.frame_count += 1
                    
                    # Thêm overlay
                    cv2.putText(frame, f'Frame: {self.frame_count}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    
                    cv2.putText(frame, f'Queue: {len(self.frame_queue)}', 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    cv2.putText(frame, 'Press Q to quit', 
                               (10, frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('WebRTC Stream', frame)
                    
                    if self.frame_count % 30 == 0:
                        print(f"✅ Displayed {self.frame_count} frames")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break
            except Exception as e:
                print(f" Display error: {e}")
                continue
        
        ws.close()
        cv2.destroyAllWindows()
        print("👋 Receiver stopped")

def main():
    print("="*60)
    print("🚀 WebRTC Stream Receiver (WebSocket Client)")
    print("="*60)
    print("️  Make sure VPS server is running first!")
    print()
    
    try:
        receiver = VideoReceiver()
        receiver.receive_and_display()
    except Exception as e:
        print(f" Fatal error: {e}")

if __name__ == "__main__":
    main()