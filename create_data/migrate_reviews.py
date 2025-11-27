"""
migrate_reviews.py - Sync review data from MySQL to PostgreSQL AI database
Supports Hybrid Sentiment Analysis (AI + Rating fallback)
"""

import mysql.connector
import psycopg2
from datetime import datetime
import logging
import sys

# Import module engine vừa tạo
try:
    from create_data.sentiment_engine import VietnameseSentimentAnalyzer
except ImportError:
    print("❌ Error: Missing sentiment_engine.py file")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION (Khuyên dùng .env cho production) ---
MYSQL_ORDER_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'ecommerce_order_db',
    'user': 'root',
    'password': '101204',
    'charset': 'utf8mb4' # Quan trọng cho tiếng Việt/Emoji
}

PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'shop_service',
    'user': 'postgres',
    'password': '101204'
}

BATCH_SIZE = 64  # Số lượng review xử lý mỗi lần (Tùy chỉnh theo RAM/CPU)

def sync_reviews():
    logger.info("🚀 Starting Advanced Review Sync...")
    logger.info(f"Source: MySQL {MYSQL_ORDER_CONFIG['database']}")
    logger.info(f"Target: PostgreSQL {PG_CONFIG['database']}")
    
    # 1. Khởi tạo AI Engine (Chỉ load model 1 lần)
    try:
        sentiment_analyzer = VietnameseSentimentAnalyzer()
    except Exception as e:
        logger.error("❌ Cannot start AI Engine. Aborting.")
        return

    mysql_conn = None
    pg_conn = None

    try:
        # 2. Kết nối Databases
        mysql_conn = mysql.connector.connect(**MYSQL_ORDER_CONFIG)
        pg_conn = psycopg2.connect(**PG_CONFIG)
        
        # 3. Fetch dữ liệu từ MySQL
        with mysql_conn.cursor(dictionary=True) as cur:
            logger.info(" Fetching reviews from MySQL...")
            cur.execute("""
                SELECT 
                    pc.comment_id as review_id,
                    pc.product_id,
                    pc.sku_id,
                    pc.user_id,
                    pc.rating,
                    pc.title,
                    pc.content,
                    pc.created_at,
                    COUNT(rl.user_id) as helpful_count
                FROM product_comment pc
                LEFT JOIN review_likes rl ON pc.comment_id = rl.review_id
                WHERE pc.parent_id IS NULL 
                AND pc.rating IS NOT NULL
                GROUP BY pc.comment_id, pc.product_id, pc.sku_id, pc.user_id, 
                         pc.rating, pc.title, pc.content, pc.created_at
                ORDER BY pc.created_at DESC
            """)
            reviews = cur.fetchall()
        
        total_reviews = len(reviews)
        logger.info(f" Found {total_reviews} reviews to process.")
        
        if total_reviews == 0:
            return

        # 4. Xóa dữ liệu cũ (Full Sync strategy)
        # Trong thực tế có thể dùng "Upsert" dựa trên timestamp để sync nhanh hơn
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM product_reviews")
        pg_conn.commit()
        logger.info(" Cleared existing PostgreSQL reviews.")

        # 5. Xử lý theo Batch (Quan trọng để tối ưu AI)
        inserted_count = 0
        
        with pg_conn.cursor() as cur:
            for i in range(0, total_reviews, BATCH_SIZE):
                batch = reviews[i : i + BATCH_SIZE]
                
                # Tách list để đưa vào hàm xử lý
                contents = [r.get('content') for r in batch]
                ratings = [r.get('rating', 3) for r in batch] # Default 3 sao nếu null
                
                # --- GỌI HÀM HYBRID ---
                # Hàm này trả về list [(label, score), ...]
                analyzed_results = sentiment_analyzer.analyze_batch_hybrid(contents, ratings)
                
                # Insert từng dòng trong batch
                for idx, review in enumerate(batch):
                    sent_label, sent_score = analyzed_results[idx]
                    
                    try:
                        cur.execute("""
                            INSERT INTO product_reviews 
                            (review_id, product_id, sku_id, user_id, rating, 
                             title, content, helpful_count, sentiment_score, 
                             sentiment_label, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (review_id) DO UPDATE SET
                                rating = EXCLUDED.rating,
                                helpful_count = EXCLUDED.helpful_count,
                                sentiment_score = EXCLUDED.sentiment_score,
                                sentiment_label = EXCLUDED.sentiment_label
                        """, (
                            review['review_id'],
                            review['product_id'],
                            review['sku_id'],
                            review['user_id'],
                            review['rating'],
                            review['title'],
                            review['content'],
                            review['helpful_count'],
                            float(sent_score),
                            sent_label,
                            review['created_at']
                        ))
                        inserted_count += 1
                    except Exception as e:
                        logger.error(f"⚠️ Insert Error ID {review['review_id']}: {e}")

                # Commit mỗi batch hoặc mỗi vài batch để an toàn
                pg_conn.commit()
                
                # Progress log
                progress = min(i + BATCH_SIZE, total_reviews)
                print(f"   Processed {progress}/{total_reviews} reviews...", end='\r')

        print("") # Xuống dòng sau khi chạy xong
        logger.info(f" Successfully inserted {inserted_count} reviews.")

        # 6. Cập nhật Metrics thống kê
        update_product_review_metrics(pg_conn)
        update_user_review_metrics(pg_conn)
        
        logger.info("🎉 All Done!")

    except Exception as e:
        logger.error(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if mysql_conn and mysql_conn.is_connected():
            mysql_conn.close()
        if pg_conn:
            pg_conn.close()

def update_product_review_metrics(pg_conn):
    logger.info("📊 Updating Product metrics (Avg Rating, Counts)...")
    with pg_conn.cursor() as cur:
        cur.execute("""
            UPDATE product_features pf SET
                review_count = COALESCE(r.cnt, 0),
                avg_rating_updated = COALESCE(r.avg_rating, 0),
                rating_distribution = COALESCE(r.distribution, '{}'::jsonb)
            FROM (
                SELECT 
                    product_id,
                    COUNT(*) as cnt,
                    AVG(rating) as avg_rating,
                    jsonb_build_object(
                        '5', COUNT(CASE WHEN rating = 5 THEN 1 END),
                        '4', COUNT(CASE WHEN rating = 4 THEN 1 END),
                        '3', COUNT(CASE WHEN rating = 3 THEN 1 END),
                        '2', COUNT(CASE WHEN rating = 2 THEN 1 END),
                        '1', COUNT(CASE WHEN rating = 1 THEN 1 END)
                    ) as distribution
                FROM product_reviews
                GROUP BY product_id
            ) r
            WHERE pf.product_id = r.product_id
        """)
        pg_conn.commit()
        logger.info(f"   Updated metrics for {cur.rowcount} products.")

def update_user_review_metrics(pg_conn):
    logger.info("👤 Updating User profile metrics...")
    with pg_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO user_profiles (user_id, review_count, avg_rating_given, is_verified_reviewer)
            SELECT 
                user_id,
                COUNT(*) as review_count,
                AVG(rating) as avg_rating_given,
                CASE WHEN COUNT(*) >= 3 THEN TRUE ELSE FALSE END as is_verified
            FROM product_reviews
            GROUP BY user_id
            ON CONFLICT (user_id) DO UPDATE SET
                review_count = EXCLUDED.review_count,
                avg_rating_given = EXCLUDED.avg_rating_given,
                is_verified_reviewer = EXCLUDED.is_verified_reviewer,
                profile_updated_at = NOW()
        """)
        pg_conn.commit()
        logger.info(f"   Updated profiles for {cur.rowcount} users.")

if __name__ == '__main__':
    sync_reviews()