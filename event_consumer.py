import json
import logging
import time
import signal
import sys
from datetime import datetime
from typing import List, Dict, Any

from kafka import KafkaConsumer
from kafka.errors import KafkaError
import psycopg2.extras  # Thư viện quan trọng để bulk insert

from database import Database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cấu hình điểm số cho các hành động
ACTION_SCORES = {
    'view': 1.0,
    'search': 0.5,
    'cart_add': 3.0,
    'cart_remove': -1.0,
    'wishlist': 2.0,
    'purchase': 10.0,
    'click': 0.5
}

class BulkEventConsumer:
    """
    Consumer xử lý event theo lô (Batch Processing)
    - Gom nhiều events lại rồi insert 1 lần vào DB (Tối ưu hiệu năng DB)
    - Commit offset thủ công (Đảm bảo không mất dữ liệu nếu crash)
    """
    
    def __init__(self, kafka_config: dict, db: Database):
        self.db = db
        
        # Cấu hình Batch
        self.BATCH_SIZE = 1000       # Gom đủ 1000 events thì ghi
        self.FLUSH_INTERVAL = 5.0    # Hoặc tối đa 5 giây thì ghi
        self.buffer: List[Dict] = [] # Vùng nhớ tạm
        self.last_flush_time = time.time()
        
        # Khởi tạo Kafka Consumer
        self.consumer = KafkaConsumer(
            'user-events',  # Topic name
            bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
            group_id=kafka_config.get('consumer_group', 'recommendation-engine-bulk'),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            # QUAN TRỌNG: Tắt auto commit để kiểm soát việc mất dữ liệu
            enable_auto_commit=False,
            # Tối ưu fetch
            fetch_min_bytes=1024, 
            fetch_max_wait_ms=500
        )
        
        # Xử lý tín hiệu tắt (Ctrl+C) để flush dữ liệu còn lại
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        self.running = True
        
        logger.info(f"BulkEventConsumer initialized. Batch Size: {self.BATCH_SIZE}, Interval: {self.FLUSH_INTERVAL}s")

    def start(self):
        """Bắt đầu vòng lặp tiêu thụ tin nhắn"""
        logger.info("Started consuming events...")
        
        try:
            # Vòng lặp chính đọc message từ Kafka
            for message in self.consumer:
                if not self.running:
                    break
                
                # 1. Thêm vào bộ đệm
                event = message.value
                self.buffer.append(event)
                
                # 2. Kiểm tra điều kiện để ghi xuống DB
                current_time = time.time()
                is_batch_full = len(self.buffer) >= self.BATCH_SIZE
                is_time_up = (current_time - self.last_flush_time) >= self.FLUSH_INTERVAL
                
                if is_batch_full or is_time_up:
                    success = self.flush_buffer()
                    if success:
                        # 3. Chỉ commit Kafka khi đã ghi DB thành công
                        self.consumer.commit()
                        self.last_flush_time = time.time()
                    else:
                        # Nếu ghi lỗi, break loop hoặc xử lý retry
                        # Ở đây ta chọn break để restart container/service nhằm tránh mất data
                        logger.error("Critical Error: Failed to flush to DB. Stopping consumer.")
                        break
                        
        except Exception as e:
            logger.error(f"Consumer loop crashed: {e}", exc_info=True)
        finally:
            self.close()

    def flush_buffer(self) -> bool:
        """Ghi dữ liệu từ bộ đệm vào Database"""
        if not self.buffer:
            return True

        row_count = len(self.buffer)
        logger.info(f"Flushing {row_count} events to Database...")
        
        try:
            # Chuẩn bị dữ liệu cho execute_values
            values = []
            for event in self.buffer:
                # Tính toán điểm và map dữ liệu
                action_type = event.get('event_type', 'view')
                score = event.get('score')
                
                # Nếu client không gửi score, tự tính dựa trên config
                if score is None:
                    score = ACTION_SCORES.get(action_type, 1.0)
                
                # Xử lý thời gian
                created_at = self._parse_timestamp(event.get('server_timestamp') or event.get('timestamp'))
                
                # Tạo tuple tương ứng với các cột trong bảng user_interactions
                values.append((
                    event.get('user_id'),
                    event.get('product_id'),
                    event.get('shop_id', 'unknown'),
                    action_type,
                    float(score),
                    int(event.get('quantity', 1)),
                    float(event.get('price', 0)) if event.get('price') else None,
                    json.dumps(event.get('metadata', {})),
                    created_at
                ))

            # Thực hiện Bulk Insert
            conn = self.db.connect()
            with conn.cursor() as cur:
                query = """
                    INSERT INTO user_interactions 
                    (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at)
                    VALUES %s
                """
                # execute_values cực nhanh cho việc insert nhiều dòng
                psycopg2.extras.execute_values(cur, query, values)
            
            conn.commit()
            logger.info(f"Successfully inserted {row_count} rows.")
            
            # Xóa bộ đệm sau khi ghi thành công
            self.buffer.clear()
            return True

        except Exception as e:
            logger.error(f"Flush failed: {e}")
            if 'conn' in locals():
                conn.rollback()
            return False

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Helper để parse thời gian từ nhiều định dạng"""
        if ts is None:
            return datetime.now()
        try:
            # Nếu là số (timestamp từ time.time())
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(ts)
            # Nếu là string
            if isinstance(ts, str):
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except:
            pass
        return datetime.now()

    def shutdown(self, signum, frame):
        """Xử lý tắt graceful"""
        logger.info("Received shutdown signal. Flushing remaining events...")
        self.running = False
        # Cố gắng flush lần cuối
        if self.buffer:
            if self.flush_buffer():
                self.consumer.commit()
        sys.exit(0)

    def close(self):
        self.consumer.close()
        logger.info("Consumer closed.")

# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    import yaml
    import os
    
    # Load Config
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        logger.error("Config file not found!")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Init Database
    try:
        db = Database(config['database'])
        # Test connection
        db.connect()
    except Exception as e:
        logger.error(f"Cannot connect to Database: {e}")
        return

    # Start Consumer
    consumer = BulkEventConsumer(config['kafka'], db)
    consumer.start()

if __name__ == '__main__':
    main()