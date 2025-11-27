"""
data_pipeline.py
-----------------------
Quy trình ETL toàn diện cho AI Recommendation System:
1. Sync Reviews từ MySQL -> PostgreSQL (Có phân tích cảm xúc).
2. Sync Product Features (Embedding mô tả sản phẩm).
3. Transform Orders & Comments thật thành User Interactions (Train Data).
4. Tính toán User Profiles dựa trên lịch sử mua hàng thật.
"""

import psycopg2
import mysql.connector
import logging
import sys
import json
import random
import numpy as np
import uuid
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer
from bs4 import BeautifulSoup
import re

# Import AI Engine
try:
    from create_data.sentiment_engine import VietnameseSentimentAnalyzer
except ImportError:
    print("❌ Error: Missing sentiment_engine.py. Please create it first.")
    sys.exit(1)

import sys
import os
import time
import json
import random
import logging
import uuid
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import mysql.connector
import psycopg2
from psycopg2.extras import execute_values
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# Load từ biến môi trường hoặc file config trong thực tế
MYSQL_PRODUCT_CONFIG = {
    'host': 'localhost', 'port': 3306, 'user': 'root', 'password': '101204',
    'database': 'ecommerce_product_db', 'charset': 'utf8mb4'
}

MYSQL_ORDER_CONFIG = {
    'host': 'localhost', 'port': 3306, 'user': 'root', 'password': '101204',
    'database': 'ecommerce_order_db', 'charset': 'utf8mb4'
}

PG_CONFIG = {
    'host': 'localhost', 'port': 5432, 'user': 'postgres', 'password': '101204',
    'database': 'shop_service'
}

# AI Models
SENTIMENT_MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"
EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-embedding"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- HELPERS ---
def get_pg_conn():
    return psycopg2.connect(**PG_CONFIG)

def get_mysql_product_conn():
    return mysql.connector.connect(**MYSQL_PRODUCT_CONFIG)

def get_mysql_order_conn():
    return mysql.connector.connect(**MYSQL_ORDER_CONFIG)

# ==============================================================================
# MODULE 1: SENTIMENT ANALYSIS ENGINE (INTEGRATED)
# ==============================================================================
# class SentimentEngine:
#     def __init__(self):
#         try:
#             from transformers import pipeline
#             logger.info("⏳ Loading Sentiment Model (PhoBERT)...")
#             # device=0 nếu có GPU, -1 là CPU
#             self.classifier = pipeline(
#                 "sentiment-analysis", 
#                 model=SENTIMENT_MODEL_NAME, 
#                 tokenizer=SENTIMENT_MODEL_NAME,
#                 truncation=True, 
#                 max_length=256,
#                 device=-1 
#             )
#             logger.info(" Sentiment Model Loaded.")
#         except Exception as e:
#             logger.error(f"❌ Failed to load Sentiment Model: {e}")
#             self.classifier = None

#     def analyze_hybrid(self, contents, ratings):
#         results = []
#         # Tách text cần analyze
#         to_analyze_indices = [i for i, txt in enumerate(contents) if txt and len(txt.strip()) > 3]
#         to_analyze_texts = [contents[i] for i in to_analyze_indices]
        
#         ai_preds = []
#         if self.classifier and to_analyze_texts:
#             try:
#                 ai_preds = self.classifier(to_analyze_texts, batch_size=16)
#             except:
#                 ai_preds = [{'label': 'NEU', 'score': 0.5}] * len(to_analyze_texts)

#         ai_cursor = 0
#         for i in range(len(contents)):
#             rating = ratings[i]
#             if i in to_analyze_indices:
#                 pred = ai_preds[ai_cursor]
#                 ai_cursor += 1
#                 label, score = self._convert_ai(pred)
#                 # Fallback nếu AI không chắc chắn
#                 if abs(score) < 0.6:
#                     label, score = self._convert_rating(rating)
#             else:
#                 label, score = self._convert_rating(rating)
#             results.append((label, score))
#         return results

#     def _convert_ai(self, res):
#         label = res['label']
#         score = res['score']
#         if label == 'POS': return 'positive', score
#         elif label == 'NEG': return 'negative', -score
#         else: return 'neutral', 0.0

#     def _convert_rating(self, rating):
#         if rating >= 5: return 'positive', 1.0
#         elif rating == 4: return 'positive', 0.8
#         elif rating == 3: return 'neutral', 0.0
#         elif rating == 2: return 'negative', -0.5
#         else: return 'negative', -1.0

# ==============================================================================
# MODULE 2: EMBEDDING ENGINE
# ==============================================================================
class EmbeddingEngine:
    def __init__(self):
        logger.info("⏳ Loading Embedding Model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        self.model.max_seq_length = 256
        logger.info(" Embedding Model Loaded.")

    def clean_text(self, html_text):
        if not html_text: return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True).lower()
        text = re.sub(r'giá sản phẩm trên tiki.*?(?=\.|$)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1000] # Giới hạn ký tự input

    def encode(self, text):
        if not text or len(text) < 5:
            # Fallback random normalized vector
            vec = np.random.normal(0, 1, 768)
            return (vec / np.linalg.norm(vec)).tolist()
        
        try:
            tokenized = ViTokenizer.tokenize(text)
            # Truncate tokens thủ công nếu cần
            embedding = self.model.encode(tokenized, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            vec = np.random.normal(0, 1, 768)
            return (vec / np.linalg.norm(vec)).tolist()

# ==============================================================================
# TASK 1: SYNC REVIEWS (ETL)
# ==============================================================================
def task_sync_reviews(sentiment_engine):
    logger.info("--- TASK 1: Syncing Reviews from MySQL ---")
    pg_conn = get_pg_conn()
    
    # Check if we need to sync
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM product_reviews")
        count = cur.fetchone()[0]
        if count > 0:
            logger.info(f"⚡ PostgreSQL already has {count} reviews. Skipping full sync (or implement incremental logic).")
            # Trong thực tế, bạn sẽ check timestamp `updated_at` để sync thêm. 
            # Ở đây ta return để tiết kiệm thời gian nếu đã có dữ liệu.
            pg_conn.close()
            return 

    mysql_conn = get_mysql_order_conn()
    try:
        with mysql_conn.cursor(dictionary=True) as cur:
            # Lấy review thật từ bảng product_comment
            cur.execute("""
                SELECT 
                    comment_id as review_id, product_id, sku_id, user_id, 
                    rating, title, content, created_at
                FROM product_comment
                WHERE parent_id IS NULL AND rating IS NOT NULL
            """)
            reviews = cur.fetchall()
        
        logger.info(f"Fetched {len(reviews)} reviews from MySQL.")
        if not reviews: return

        # Process in batches
        BATCH_SIZE = 50
        total_inserted = 0
        
        with pg_conn.cursor() as cur:
            for i in range(0, len(reviews), BATCH_SIZE):
                batch = reviews[i : i + BATCH_SIZE]
                contents = [r['content'] for r in batch]
                ratings = [r['rating'] for r in batch]
                
                # AI Analysis
                sentiments = sentiment_engine.analyze_hybrid(contents, ratings)
                
                values = []
                for idx, row in enumerate(batch):
                    lbl, score = sentiments[idx]
                    values.append((
                        row['review_id'], row['product_id'], row['sku_id'], row['user_id'],
                        row['rating'], row['title'], row['content'], 0,
                        float(score), lbl, row['created_at']
                    ))
                
                query = """
                    INSERT INTO product_reviews 
                    (review_id, product_id, sku_id, user_id, rating, title, content, helpful_count, sentiment_score, sentiment_label, created_at)
                    VALUES %s
                    ON CONFLICT (review_id) DO NOTHING
                """
                execute_values(cur, query, values)
                total_inserted += len(values)
                print(f"   Processed {total_inserted}/{len(reviews)}...", end='\r')
        
        pg_conn.commit()
        print("")
        logger.info(" Reviews synced successfully.")
        
        # Update aggregate metrics in user_profiles
        logger.info("Updating user profile review metrics...")
        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_profiles (user_id, review_count, avg_rating_given)
                SELECT user_id, COUNT(*), AVG(rating) FROM product_reviews GROUP BY user_id
                ON CONFLICT (user_id) DO UPDATE SET 
                review_count = EXCLUDED.review_count, avg_rating_given = EXCLUDED.avg_rating_given
            """)
        pg_conn.commit()

    finally:
        mysql_conn.close()
        pg_conn.close()

# ==============================================================================
# TASK 2: PRODUCT FEATURES & EMBEDDINGS
# ==============================================================================
def task_sync_products(embedding_engine):
    logger.info("--- TASK 2: Syncing Products & Embeddings ---")
    pg_conn = get_pg_conn()
    mysql_conn = get_mysql_product_conn()
    
    try:
        # Lấy sản phẩm từ MySQL Product DB
        with mysql_conn.cursor(dictionary=True) as cur:
            cur.execute("""
                SELECT 
                    p.id as product_id, p.category_id, p.brand_id, p.shop_id,
                    p.name, p.description, p.short_description,
                    (SELECT AVG(price) FROM product_sku WHERE product_id = p.id) as price
                FROM product p
                WHERE p.delete_status = 'Active'
                LIMIT 2000 -- Giới hạn demo
            """)
            products = cur.fetchall()
        
        logger.info(f"Processing {len(products)} products...")
        
        values = []
        for p in products:
            # Tạo text cho embedding
            raw_text = f"{p['name']} {p['short_description']} {p['description']}"
            clean_text = embedding_engine.clean_text(raw_text)
            vector = embedding_engine.encode(clean_text)
            
            price = float(p['price']) if p['price'] else 0
            
            values.append((
                p['product_id'], p['category_id'], p['brand_id'], p['shop_id'],
                price, vector, datetime.now()
            ))
        
        with pg_conn.cursor() as cur:
            # Insert/Update Features
            # Lưu ý: Cột text_embedding phải map với vector type của Postgres
            # Chúng ta dùng execute_values nhưng cần cast vector đúng kiểu
            # Cách đơn giản là loop insert hoặc format string. Ở đây dùng loop cho an toàn với vector.
            for v in values:
                cur.execute("""
                    INSERT INTO product_features 
                    (product_id, category_id, brand_id, shop_id, current_price, text_embedding, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (product_id) DO UPDATE SET
                    text_embedding = EXCLUDED.text_embedding,
                    current_price = EXCLUDED.current_price,
                    last_updated = NOW()
                """, v)
        
        pg_conn.commit()
        logger.info(" Product features & embeddings updated.")

    finally:
        mysql_conn.close()
        pg_conn.close()

# ==============================================================================
# TASK 3: IMPORT REAL ORDERS -> USER INTERACTIONS (CORE LOGIC)
# ==============================================================================
def task_import_real_interactions():
    """
    Đây là phần quan trọng nhất: Chuyển đổi Order thật thành Interaction
    """
    logger.info("--- TASK 3: Importing REAL User Interactions from Orders ---")
    
    pg_conn = get_pg_conn()
    mysql_conn = get_mysql_order_conn()
    
    try:
        # 1. Lấy dữ liệu mua hàng thật từ MySQL (JOIN 3 bảng)
        # Chỉ lấy đơn đã hoàn thành hoặc đang xử lý (không lấy đơn hủy)
        logger.info("Fetching real order history...")
        with mysql_conn.cursor(dictionary=True) as cur:
            cur.execute("""
                SELECT 
                    o.user_id,
                    oi.product_id,
                    so.shop_id,
                    oi.quantity,
                    oi.final_unit_price as price,
                    o.created_at
                FROM order_items oi
                JOIN shop_orders so ON oi.shop_order_id = so.id
                JOIN orders o ON so.order_id = o.id
                WHERE so.status NOT IN ('CANCELLED', 'REFUNDED')
                AND o.created_at >= NOW() - INTERVAL 180 DAY
            """)
            orders = cur.fetchall()
            
        logger.info(f"Found {len(orders)} purchased items to transform.")
        
        interactions_buffer = []
        
        for order in orders:
            # A. Interaction: PURCHASE (Source of Truth)
            interactions_buffer.append((
                order['user_id'], order['product_id'], order['shop_id'],
                'purchase', 10.0, # Điểm cao nhất
                order['quantity'], order['price'], 
                json.dumps({'source': 'real_order'}), order['created_at']
            ))
            
            # B. Interaction: IMPLIED VIEW (Suy luận)
            # Nếu user mua, chắc chắn họ đã xem trước đó.
            # Tạo view event trước khi mua khoảng 5-60 phút
            view_time = order['created_at'] - timedelta(minutes=random.randint(5, 60))
            interactions_buffer.append((
                order['user_id'], order['product_id'], order['shop_id'],
                'view', 1.0, # Điểm view
                1, order['price'],
                json.dumps({'source': 'implied_from_order'}), view_time
            ))
            
            # C. Interaction: IMPLIED CART (Suy luận)
            # Tạo cart event trước khi mua 2-30 phút
            cart_time = order['created_at'] - timedelta(minutes=random.randint(2, 30))
            interactions_buffer.append((
                order['user_id'], order['product_id'], order['shop_id'],
                'cart_add', 3.0,
                order['quantity'], order['price'],
                json.dumps({'source': 'implied_from_order'}), cart_time
            ))

        # 2. Insert vào Postgres
        if interactions_buffer:
            with pg_conn.cursor() as cur:
                # Xóa dữ liệu cũ (chỉ demo, thực tế dùng incremental)
                cur.execute("TRUNCATE TABLE user_interactions")
                
                query = """
                    INSERT INTO user_interactions 
                    (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at)
                    VALUES %s
                """
                execute_values(cur, query, interactions_buffer)
            pg_conn.commit()
            logger.info(f" Imported {len(interactions_buffer)} interactions (Real Purchases + Implied Views).")
        else:
            logger.warning("⚠️ No real orders found! (Please create orders in MySQL first)")

        # 3. Tính toán User Profiles dựa trên Order thật
        logger.info("Computing User Profiles from real orders...")
        with pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_profiles (user_id, total_orders, total_spent, avg_order_value, last_purchase_at, profile_updated_at)
                SELECT 
                    user_id,
                    COUNT(DISTINCT created_at) as total_orders,
                    SUM(price * quantity) as total_spent,
                    AVG(price * quantity) as avg,
                    MAX(created_at) as last_buy,
                    NOW()
                FROM user_interactions
                WHERE action_type = 'purchase'
                GROUP BY user_id
                ON CONFLICT (user_id) DO UPDATE SET
                total_orders = EXCLUDED.total_orders,
                total_spent = EXCLUDED.total_spent,
                last_purchase_at = EXCLUDED.last_purchase_at
            """)
        pg_conn.commit()
        logger.info(" User profiles updated.")

    finally:
        mysql_conn.close()
        pg_conn.close()

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("=======================================================")
    print("   AI DATA PIPELINE (REAL DATA MIGRATION)   ")
    print("=======================================================")
    
    # 1. Init Engines (Load Model tốn thời gian nên init 1 lần)
    sentiment_engine = VietnameseSentimentAnalyzer()
    embedding_engine = EmbeddingEngine()
    
    # 2. Run Tasks
    try:
        # Step 1: Sync Reviews (Lấy User ID thật và sở thích thật từ comment)
        task_sync_reviews(sentiment_engine)
        
        # Step 2: Sync Products (Để có vector tìm kiếm tương đồng)
        task_sync_products(embedding_engine)
        
        # Step 3: Import Orders (Tạo dữ liệu train chính xác nhất)
        task_import_real_interactions()
        
        # Step 4: Refresh Stats
        pg_conn = get_pg_conn()
        with pg_conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW daily_recommendation_stats;")
            pg_conn.commit()
        pg_conn.close()
        
        print("\n=======================================================")
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("   - Real Orders imported.")
        print("   - Real Reviews imported.")
        print("   - Implied Views/Carts generated.")
        print("   - Ready for 'python train.py'")
        print("=======================================================")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()