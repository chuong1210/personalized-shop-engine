import mysql.connector
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

class MySQLConnection:
    """Quản lý kết nối MySQL"""
    
    def __init__(self, host: str, user: str, password: str, database: str):
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'use_unicode': True
        }
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Kết nối đến MySQL"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            print("✅ Kết nối MySQL thành công!")
        except mysql.connector.Error as err:
            print(f" Lỗi kết nối MySQL: {err}")
            raise
    
    def close(self):
        """Đóng kết nối"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("🔒 Đã đóng kết nối MySQL")
    
    def get_unprocessed_events(self, batch_size: int = 50) -> List[Dict[str, Any]]:
        """
        Lấy các events chưa được xử lý (custom_metadata = NULL hoặc empty)
        và author = 'user'
        """
        query = """
        SELECT 
            id,
            app_name,
            user_id,
            session_id,
            content,
            custom_metadata
        FROM events
        WHERE author = 'user'
          AND content IS NOT NULL
          AND (custom_metadata IS NULL OR custom_metadata = '')
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        self.cursor.execute(query, (batch_size,))
        results = self.cursor.fetchall()
        
        # Parse JSON content
        events = []
        for row in results:
            try:
                content_json = json.loads(row['content'])
                # Lấy text từ parts
                text = ""
                if 'parts' in content_json and len(content_json['parts']) > 0:
                    text = content_json['parts'][0].get('text', '')
                
                if text.strip():  # Chỉ lấy nếu có text
                    events.append({
                        'id': row['id'],
                        'app_name': row['app_name'],
                        'user_id': row['user_id'],
                        'session_id': row['session_id'],
                        'message': text,
                        'original_content': row['content']
                    })
            except json.JSONDecodeError:
                print(f"️ Không parse được content của event {row['id']}")
                continue
        
        return events
    
    def batch_update_custom_metadata(self, updates: List[Dict[str, Any]]) -> int:
        """
        Batch update custom_metadata cho nhiều events
        
        Args:
            updates: List of {'id': event_id, 'custom_metadata': json_string}
        
        Returns:
            Số bản ghi được cập nhật
        """
        if not updates:
            return 0
        
        query = """
        UPDATE events
        SET custom_metadata = %s
        WHERE id = %s
        """
        
        # Chuẩn bị data cho batch update
        data = [(update['custom_metadata'], update['id']) for update in updates]
        
        try:
            self.cursor.executemany(query, data)
            self.connection.commit()
            updated_count = self.cursor.rowcount
            print(f"✅ Đã cập nhật {updated_count} bản ghi")
            return updated_count
        except mysql.connector.Error as err:
            print(f" Lỗi khi cập nhật: {err}")
            self.connection.rollback()
            return 0
    
    def get_stats(self) -> Dict[str, int]:
        """Lấy thống kê"""
        stats = {}
        
        # Tổng số events của user
        self.cursor.execute("SELECT COUNT(*) as total FROM events WHERE author = 'user'")
        stats['total_user_events'] = self.cursor.fetchone()['total']
        
        # Số events đã xử lý
        self.cursor.execute("""
            SELECT COUNT(*) as processed 
            FROM events 
            WHERE author = 'user' 
              AND custom_metadata IS NOT NULL 
              AND custom_metadata != ''
        """)
        stats['processed_events'] = self.cursor.fetchone()['processed']
        
        # Số events chưa xử lý
        stats['unprocessed_events'] = stats['total_user_events'] - stats['processed_events']
        
        return stats