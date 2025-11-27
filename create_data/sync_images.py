"""
sync_images.py
Đồng bộ và Vector hóa hình ảnh sản phẩm
"""
import mysql.connector
import psycopg2
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import io
import logging
from psycopg2.extras import execute_values

# Cấu hình DB (Dùng lại config cũ của bạn)
MYSQL_PRODUCT_CONFIG = {
    'host': 'localhost', 'port': 3306, 'user': 'root', 'password': '101204',
    'database': 'ecommerce_product_db', 'charset': 'utf8mb4'
}
PG_CONFIG = {
    'host': 'localhost', 'port': 5432, 'user': 'postgres', 'password': '101204',
    'database': 'shop_service'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageEmbeddingEngine:
    def __init__(self):
        logger.info("⏳ Loading CLIP Model...")
        # Sử dụng model CLIP đa ngôn ngữ hoặc bản chuẩn
        self.model = SentenceTransformer('clip-ViT-B-32')
        logger.info(" CLIP Model Loaded.")

    def image_to_vector(self, image_source):
        """
        Input: URL string hoặc PIL Image object
        Output: List[float] (512 dims)
        """
        try:
            img = None
            if isinstance(image_source, str): # Nếu là URL
                # Giả sử ảnh lưu local hoặc url public
                # Nếu là URL:
                if image_source.startswith('http'):
                    response = requests.get(image_source, stream=True, timeout=5)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content))
                else:
                    # Nếu là đường dẫn file local
                    img = Image.open(image_source)
            else:
                img = image_source # Đã là PIL Image

            # Encode
            vector = self.model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
            return vector.tolist()
            
        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            return None
import json

def sync_product_images():
    engine = ImageEmbeddingEngine()
    
    mysql_conn = mysql.connector.connect(**MYSQL_PRODUCT_CONFIG)
    pg_conn = psycopg2.connect(**PG_CONFIG)
    
    try:
        # 1. Lấy dữ liệu: ID, Image chính, Media
        with mysql_conn.cursor(dictionary=True) as cur:
            cur.execute("""
                SELECT id, image, media 
                FROM product 
                WHERE delete_status = 'Active'
            """)
            products = cur.fetchall()
            
        logger.info(f"Found {len(products)} products. Processing images...")
        
        # Xóa dữ liệu cũ để sync lại từ đầu (cho sạch)
        with pg_conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE product_image_embeddings")
        pg_conn.commit()

        batch_data = []
        total_vectors = 0
        
        for p in products:
            product_id = p['id']
            all_urls = []
            
            # A. Lấy ảnh chính
            if p['image']:
                all_urls.append(p['image'])
            
            # B. Lấy ảnh Media (JSON Array)
            if p['media']:
                try:
                    # Parse chuỗi JSON thành List Python
                    # Input: '["url1", "url2"]' -> Output: ['url1', 'url2']
                    media_urls = json.loads(p['media'])
                    
                    if isinstance(media_urls, list):
                        all_urls.extend(media_urls)
                except Exception as e:
                    # Nếu JSON lỗi thì bỏ qua media, chỉ dùng ảnh chính
                    pass
            
            # C. Deduplicate (Loại bỏ ảnh trùng nhau nếu ảnh chính lặp lại trong media)
            unique_urls = list(set(all_urls))
            
            # D. Vector hóa từng ảnh
            for url in unique_urls:
                # Bỏ qua url rỗng hoặc quá ngắn
                if not url or len(url) < 10: 
                    continue

                vector = engine.image_to_vector(url)
                
                if vector:
                    # Thêm vào batch: (product_id, url, vector)
                    batch_data.append((product_id, url, vector))
            
            # Insert mỗi khi gom đủ 50 vector (để tiết kiệm RAM)
            if len(batch_data) >= 50:
                _insert_batch(pg_conn, batch_data)
                total_vectors += len(batch_data)
                batch_data = [] # Reset batch
                print(f"Processed vectors: {total_vectors}...", end='\r')

        # Insert số còn lại
        if batch_data:
            _insert_batch(pg_conn, batch_data)
            total_vectors += len(batch_data)
            
        logger.info(f"\n DONE! Total vectors created: {total_vectors}")
        
    finally:
        mysql_conn.close()
        pg_conn.close()

def _insert_batch(conn, data):
    with conn.cursor() as cur:
        # Insert vào bảng mới product_image_embeddings
        execute_values(cur, """
            INSERT INTO product_image_embeddings (product_id, image_url, embedding)
            VALUES %s
        """, data)
    conn.commit()

if __name__ == "__main__":
    sync_product_images()