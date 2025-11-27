"""
Chat Analyzer - Phân tích batch các tin nhắn từ MySQL events table
Tối ưu token bằng cách gộp nhiều tin nhắn vào 1 lần gọi LLM
"""

import json
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# ========================================
# 🔧 CẤU HÌNH
# ========================================

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '101204',
    'database': 'agent_ai_db',
    'charset': 'utf8mb4'
}

# API Key - Thay bằng key của bạn hoặc dùng biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCzsBMZaslKg5xnuRjEm8L7-D2bgHRPZIk")

# Cấu hình batch processing
BATCH_SIZE = 10  # Số tin nhắn xử lý cùng lúc (tăng/giảm tùy quota)
DELAY_BETWEEN_BATCHES = 65  # Giây chờ giữa các batch (tránh rate limit)

# ========================================
# 🧠 KHỞI TẠO LLM VỚI RETRY
# ========================================

# Dùng model có quota cao hơn
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Model này có quota cao hơn gemini-2.0-flash-exp
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY,
    max_retries=3,
    timeout=120
)

# ========================================
# 📋 SYSTEM PROMPT CHO BATCH ANALYSIS
# ========================================

BATCH_ANALYSIS_PROMPT = """Bạn là chuyên gia phân tích chat e-commerce. 

Nhiệm vụ: Phân tích TẤT CẢ các tin nhắn dưới đây và trả về kết quả dạng JSON array.

Các chủ đề (topic):
- TRA_CUU_DON_HANG: Hỏi về đơn hàng
- HOI_PHI_SHIP: Hỏi về phí ship
- HOI_CHINH_SACH: Hỏi về đổi trả, bảo hành
- TIM_KIEM_SAN_PHAM: Tìm sản phẩm
- TU_VAN_SAN_PHAM: Tư vấn, so sánh sản phẩm
- HOI_KHUYEN_MAI: Hỏi về voucher, giảm giá
- KHIEU_NAI_SAN_PHAM: Khiếu nại sản phẩm
- KHIEU_NAI_GIAO_HANG: Khiếu nại giao hàng
- LOI_HE_THONG: Lỗi hệ thống
- UNKNOWN: Không xác định

Cảm xúc (sentiment):
POSITIVE, NEUTRAL, NEGATIVE, FRUSTRATED, CONFUSED, URGENT

Ý định mua (purchase_intent):
HIGH, MEDIUM, LOW, NONE

Entities cần trích xuất:
- product_id: Mã SP (p001, SP123...)
- sku_id: Mã SKU
- order_code: Mã đơn hàng
- shop_id: Mã shop
- brand_name: Thương hiệu
- category_name: Danh mục
- voucher_code: Mã giảm giá

DANH SÁCH TIN NHẮN CẦN PHÂN TÍCH:
{messages}

QUAN TRỌNG: Trả về JSON array với format:
[
  {{
    "message_index": 0,
    "topic": "TIM_KIEM_SAN_PHAM",
    "sentiment": "POSITIVE",
    "purchase_intent": "HIGH",
    "entities": [
      {{"type": "category_name", "context": "giày thể thao"}}
    ]
  }},
  {{
    "message_index": 1,
    ...
  }}
]

CHỈ TRẢ VỀ JSON ARRAY, KHÔNG GIẢI THÍCH THÊM!
"""

# ========================================
# 🗄️ DATABASE FUNCTIONS
# ========================================

def get_db_connection():
    """Kết nối MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            print(" Kết nối MySQL thành công!")
            return connection
    except Error as e:
        print(f" Lỗi kết nối MySQL: {e}")
        return None

def parse_content_field(content_str: str) -> Optional[str]:
    """Parse cột content để lấy text message"""
    try:
        content = json.loads(content_str)
        if 'parts' in content and len(content['parts']) > 0:
            return content['parts'][0].get('text', '')
        return None
    except:
        return None

def fetch_unprocessed_messages(connection, limit: int = BATCH_SIZE) -> List[Dict[str, Any]]:
    """
    Lấy các tin nhắn chưa xử lý từ DB
    - author = 'user'
    - custom_metadata IS NULL (chưa phân tích)
    """
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT id, content, timestamp
    FROM events
    WHERE author = 'user' 
      AND custom_metadata IS NULL
      AND content IS NOT NULL
    ORDER BY timestamp ASC
    LIMIT %s
    """
    
    cursor.execute(query, (limit,))
    results = cursor.fetchall()
    cursor.close()
    
    # Parse content
    messages = []
    for row in results:
        text = parse_content_field(row['content'])
        if text and text.strip():
            messages.append({
                'id': row['id'],
                'text': text.strip(),
                'timestamp': row['timestamp']
            })
    
    return messages

def update_custom_metadata_batch(connection, updates: List[Dict[str, Any]]):
    """
    Cập nhật custom_metadata cho nhiều records cùng lúc
    updates: [{"id": "...", "metadata": {...}}, ...]
    """
    cursor = connection.cursor()
    
    # Prepare batch update
    for update in updates:
        metadata_json = json.dumps(update['metadata'], ensure_ascii=False)
        
        query = """
        UPDATE events
        SET custom_metadata = %s
        WHERE id = %s
        """
        
        cursor.execute(query, (metadata_json, update['id']))
    
    connection.commit()
    cursor.close()
    print(f" Đã cập nhật {len(updates)} records vào DB")

# ========================================
# 🤖 LLM ANALYSIS WITH RETRY
# ========================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    retry=retry_if_exception_type(Exception)
)
def analyze_batch_with_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gọi LLM để phân tích batch messages
    Có retry mechanism với exponential backoff
    """
    # Chuẩn bị input cho LLM
    messages_text = ""
    for idx, msg in enumerate(messages):
        messages_text += f"\n--- Tin nhắn {idx} ---\n{msg['text']}\n"
    
    prompt = BATCH_ANALYSIS_PROMPT.format(messages=messages_text)
    
    print(f"🔄 Đang gọi LLM phân tích {len(messages)} tin nhắn...")
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse JSON response
        response_text = response.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        results = json.loads(response_text.strip())
        
        print(f" LLM trả về {len(results)} kết quả")
        return results
        
    except Exception as e:
        print(f" Lỗi khi gọi LLM: {e}")
        raise

# ========================================
# 🔄 MAIN PROCESSING LOGIC
# ========================================

def process_batch(connection, messages: List[Dict[str, Any]]):
    """Xử lý một batch messages"""
    if not messages:
        print("️ Không có tin nhắn để xử lý")
        return
    
    print(f"\n{'='*60}")
    print(f"📦 Xử lý batch: {len(messages)} tin nhắn")
    print(f"{'='*60}")
    
    try:
        # Gọi LLM phân tích batch
        analysis_results = analyze_batch_with_llm(messages)
        
        # Chuẩn bị dữ liệu update
        updates = []
        for result in analysis_results:
            idx = result.get('message_index')
            if idx is None or idx >= len(messages):
                continue
            
            msg = messages[idx]
            
            # Tạo metadata theo yêu cầu (bỏ confidence, reasoning)
            # Chỉ giữ type và context cho entities
            clean_entities = [
                {
                    "type": e.get("type"),
                    "context": e.get("context")
                }
                for e in result.get('entities', [])
            ]
            
            metadata = {
                "topic": result.get('topic', 'UNKNOWN'),
                "sentiment": result.get('sentiment', 'NEUTRAL'),
                "purchase_intent": result.get('purchase_intent', 'NONE'),
                "entities": clean_entities,
                "analyzed_at": datetime.now().isoformat()
            }
            
            updates.append({
                'id': msg['id'],
                'metadata': metadata
            })
            
            # Log kết quả
            print(f"\n📝 ID: {msg['id'][:8]}...")
            print(f"   Message: {msg['text'][:50]}...")
            print(f"   Topic: {metadata['topic']}")
            print(f"   Sentiment: {metadata['sentiment']}")
            print(f"   Intent: {metadata['purchase_intent']}")
            print(f"   Entities: {len(clean_entities)}")
        
        # Batch update vào DB
        if updates:
            update_custom_metadata_batch(connection, updates)
            print(f"\n Hoàn thành batch: {len(updates)}/{len(messages)} tin nhắn")
        
    except Exception as e:
        print(f"\n Lỗi xử lý batch: {e}")
        print("⏭️ Bỏ qua batch này và tiếp tục...")

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 CHAT ANALYZER - BATCH PROCESSING")
    print("="*60)
    
    # Kết nối DB
    connection = get_db_connection()
    if not connection:
        print(" Không thể kết nối DB. Thoát!")
        return
    
    try:
        total_processed = 0
        batch_count = 0
        
        while True:
            # Lấy batch messages chưa xử lý
            messages = fetch_unprocessed_messages(connection, BATCH_SIZE)
            
            if not messages:
                print("\n Đã xử lý xong tất cả tin nhắn!")
                break
            
            batch_count += 1
            print(f"\n{'='*60}")
            print(f"🔄 BATCH #{batch_count}: Tìm thấy {len(messages)} tin nhắn chưa xử lý")
            print(f"{'='*60}")
            
            # Xử lý batch
            process_batch(connection, messages)
            total_processed += len(messages)
            
            # Chờ giữa các batch để tránh rate limit
            if len(messages) == BATCH_SIZE:  # Còn batch tiếp theo
                print(f"\n⏳ Chờ {DELAY_BETWEEN_BATCHES}s trước batch tiếp theo (tránh rate limit)...")
                time.sleep(DELAY_BETWEEN_BATCHES)
            
        # Tổng kết
        print("\n" + "="*60)
        print(f"🎉 HOÀN THÀNH!")
        print(f"📊 Tổng số tin nhắn đã xử lý: {total_processed}")
        print(f"📦 Tổng số batch: {batch_count}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n️ Người dùng dừng chương trình")
    except Exception as e:
        print(f"\n Lỗi không mong muốn: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("🔌 Đã đóng kết nối MySQL")

# ========================================
# 🏃 RUN SCRIPT
# ========================================

if __name__ == "__main__":
    main()