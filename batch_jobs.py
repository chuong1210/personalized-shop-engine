"""
batch_jobs_updated.py - Batch jobs with review sync capability
"""

import logging
from datetime import datetime
from database import Database
from recommend_service import RecommendationService
from train import train_collaborative_filtering, train_content_based, train_hybrid_model, cache_similar_products
# Import engine để tạo embedding cho sản phẩm mới
from  create_data.data_pipeline import EmbeddingEngine 
import traceback
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)


class BatchJobs:
    """Scheduled batch jobs for maintaining recommendation system"""
    
    def __init__(self, db: Database, service: RecommendationService = None):
        self.db = db
        self.service = service
        logger.info("BatchJobs initialized")
    
    # ... (keep all existing methods) ...
    
    def update_user_spending_stats(self):
        """
        Tính toán thói quen chi tiêu của user dựa trên lịch sử MUA HÀNG và THÊM GIỎ
        """
        logger.info("Starting User Spending Profile Update...")
        start_time = datetime.now()

        # 1. Câu Query tính toán
        # Logic:
        # - Chỉ lấy các hành động: purchase (mua), cart_add (thêm giỏ)
        # - Mua hàng (purchase) quan trọng hơn, nên ta ưu tiên lọc theo purchase trước.
        # - Nếu chưa mua gì, ta có thể tính dựa trên cart_add để dự đoán.
        
        query = """
            WITH spending_stats AS (
                SELECT 
                    user_id,
                    COUNT(*) as total_actions,
                    AVG(price) as avg_price,       -- Giá trung bình sản phẩm họ quan tâm
                    MIN(price) as min_price,       -- Giá thấp nhất
                    MAX(price) as max_price,       -- Giá cao nhất
                    MAX(created_at) as last_action
                FROM user_interactions
                WHERE action_type IN ('purchase', 'cart_add') 
                  AND price > 0 -- Loại bỏ lỗi giá = 0
                GROUP BY user_id
            )
            UPDATE user_profiles up
            SET 
                avg_order_value = s.avg_price,
                price_range_min = s.min_price,
                price_range_max = s.max_price,
                last_active_at = s.last_action,
                profile_updated_at = NOW()
            FROM spending_stats s
            WHERE up.user_id = s.user_id;
        """
        
        # 2. Để chắc chắn mọi user trong interactions đều có profile, 
        # ta insert user mới trước khi update
        insert_missing_users = """
            INSERT INTO user_profiles (user_id)
            SELECT DISTINCT user_id FROM user_interactions
            ON CONFLICT (user_id) DO NOTHING;
        """

        try:
            # Bước A: Tạo profile rỗng cho user mới
            self.db.execute(insert_missing_users)
            
            # Bước B: Tính toán và Update số liệu
            self.db.execute(query)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"User spending stats updated successfully in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to update spending stats: {e}")
    
    def sync_reviews_from_mysql(self, mysql_db, batch_size: int = 1000):
        """
        Sync reviews từ MySQL sang PostgreSQL
        
        Args:
            mysql_db: MySQLDatabase instance
            batch_size: Batch size for insert
        """
        logger.info("Starting review sync from MySQL...")
        start_time = datetime.now()
        
        try:
            # Fetch reviews từ MySQL (only last 24 hours for daily sync)
            reviews_df = mysql_db.query("""
                SELECT 
                    pc.comment_id as review_id,
                    pc.product_id,
                    pc.sku_id,
                    pc.user_id,
                    pc.rating,
                    pc.title,
                    pc.content,
                    COALESCE(like_count.count, 0) as helpful_count,
                    pc.created_at,
                    pc.updated_at
                FROM product_comment pc
                LEFT JOIN (
                    SELECT review_id, COUNT(*) as count
                    FROM review_likes
                    GROUP BY review_id
                ) like_count ON pc.comment_id = like_count.review_id
                WHERE 
                    pc.parent_id IS NULL
                    AND pc.rating BETWEEN 1 AND 5
                    AND pc.updated_at >= NOW() - INTERVAL 1 DAY
                ORDER BY pc.updated_at DESC
            """)
            
            if reviews_df.empty:
                logger.info("No new/updated reviews to sync")
                return
            
            logger.info(f"Found {len(reviews_df)} reviews to sync")
            
            # Prepare data
            review_data = []
            for _, row in reviews_df.iterrows():
                review_data.append((
                    row['review_id'],
                    row['product_id'],
                    row['sku_id'],
                    row['user_id'],
                    int(row['rating']),
                    row['title'] if row['title'] else '',
                    row['content'] if row['content'] else '',
                    int(row['helpful_count']),
                    row['created_at'],
                    row['updated_at']
                ))
            
            # Batch insert
            for i in range(0, len(review_data), batch_size):
                batch = review_data[i:i + batch_size]
                
                self.db.execute_many("""
                    INSERT INTO product_reviews 
                    (review_id, product_id, sku_id, user_id, rating, title, content, 
                     helpful_count, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (review_id) DO UPDATE SET
                        rating = EXCLUDED.rating,
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        helpful_count = EXCLUDED.helpful_count,
                        updated_at = EXCLUDED.updated_at
                """, batch)
            
            # Update product rating metrics
            product_ids = reviews_df['product_id'].unique()
            for product_id in product_ids:
                self.db.execute(
                    "SELECT update_product_rating_metrics(%s)",
                    (product_id,)
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Review sync completed: {len(reviews_df)} reviews, "
                       f"{len(product_ids)} products updated in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Review sync failed: {e}")
            raise
    
    # ---------------------------------------------------------
    # 3. UPDATE PRODUCT METRICS (QUAN TRỌNG - THIẾU)
    # ---------------------------------------------------------
    def update_product_metrics(self):
        """
        Tính toán các chỉ số thống kê cho sản phẩm:
        - View/Purchase 7 ngày, 30 ngày.
        - Conversion Rate.
        - Trending Score.
        """
        logger.info("Updating Product Metrics...")
        try:
            # Logic:
            # 1. Tính tổng view/mua trong 7 và 30 ngày qua từ user_interactions
            # 2. Update vào bảng product_features
            
            query = """
                WITH metrics AS (
                    SELECT 
                        product_id,
                        COUNT(CASE WHEN action_type = 'view' AND created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as views_7d,
                        COUNT(CASE WHEN action_type = 'view' AND created_at >= NOW() - INTERVAL '30 days' THEN 1 END) as views_30d,
                        COUNT(CASE WHEN action_type = 'purchase' AND created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as buys_7d,
                        COUNT(CASE WHEN action_type = 'purchase' AND created_at >= NOW() - INTERVAL '30 days' THEN 1 END) as buys_30d
                    FROM user_interactions
                    GROUP BY product_id
                )
                UPDATE product_features pf
                SET 
                    view_count_7d = m.views_7d,
                    view_count_30d = m.views_30d,
                    purchase_count_7d = m.buys_7d,
                    purchase_count_30d = m.buys_30d,
                    -- Tính Conversion Rate (Mua / Xem)
                    conversion_rate = CASE 
                        WHEN m.views_30d > 0 THEN CAST(m.buys_30d AS DECIMAL) / m.views_30d 
                        ELSE 0 
                    END,
                    -- Tính Trending Score (Công thức đơn giản: Mua * 10 + Xem)
                    trending_score = (m.buys_7d * 10) + m.views_7d,
                    last_updated = NOW()
                FROM metrics m
                WHERE pf.product_id = m.product_id;
            """
            self.db.execute(query)
            logger.info("Product metrics updated successfully.")
            
        except Exception as e:
            logger.error(f"Failed to update product metrics: {e}")

    # ---------------------------------------------------------
    # 4. REFRESH MATERIALIZED VIEWS (THIẾU)
    # ---------------------------------------------------------
    def refresh_materialized_views(self):
        """Làm mới các bảng view thống kê cho Dashboard"""
        logger.info("Refreshing Materialized Views...")
        try:
            # CONCURRENTLY giúp không bị lock bảng khi đang refresh (nếu Postgres version hỗ trợ)
            # Nếu lỗi, bỏ chữ CONCURRENTLY
            self.db.execute("REFRESH MATERIALIZED VIEW daily_recommendation_stats;")
            logger.info("Views refreshed.")
        except Exception as e:
            logger.error(f"Failed to refresh views: {e}")

    # ---------------------------------------------------------
    # 5. SYNC PRODUCTS FROM MAIN DB (THIẾU)
    # ---------------------------------------------------------
    def sync_products_from_main_db(self, mysql_db):
        """
        Đồng bộ sản phẩm mới hoặc cập nhật giá từ MySQL -> Postgres
        Chạy hàng ngày để hệ thống AI luôn có dữ liệu mới nhất.
        """
        logger.info("Syncing Products from Main DB...")
        try:
            # 1. Lấy sản phẩm đã thay đổi trong 24h qua (hoặc toàn bộ nếu ít)
            # Ở đây lấy hết status Active để đảm bảo giá đúng
            products_df = mysql_db.query("""
                SELECT 
                    id as product_id, category_id, brand_id, shop_id,
                    name, description, short_description,
                    (SELECT AVG(price) FROM product_sku WHERE product_id = product.id) as price
                FROM product
                WHERE delete_status = 'Active' 
                AND updated_at >= NOW() - INTERVAL 2 DAY
            """)
            
            if products_df.empty:
                logger.info("No product updates found.")
                return

            logger.info(f"Syncing {len(products_df)} products...")
            
            # 2. Update vào Postgres
            # Lưu ý: Chỉ tính lại Embedding nếu chưa có hoặc cần thiết (ở đây update giá là chính)
            for _, row in products_df.iterrows():
                # Logic: Nếu sản phẩm chưa có vector thì tạo, có rồi thì giữ nguyên (trừ khi muốn update lại vector)
                # Để đơn giản và nhanh, ta chỉ tạo vector nếu cần
                
                # Check exist
                exists = self.db.fetchone("SELECT 1 FROM product_features WHERE product_id = %s", (row['product_id'],))
                
                vector = None
                if not exists and self.embedding_engine:
                    # Tạo vector cho sp mới
                    text = f"{row['name']} {row['short_description']} {row['description']}"
                    clean_text = self.embedding_engine.clean_text(text)
                    vector = self.embedding_engine.encode(clean_text)
                
                # Query Upsert
                # Nếu sp đã có, chỉ update giá và category. Nếu chưa có, insert full.
                if exists:
                    self.db.execute("""
                        UPDATE product_features 
                        SET current_price = %s, category_id = %s, last_updated = NOW()
                        WHERE product_id = %s
                    """, (row['price'] or 0, row['category_id'], row['product_id']))
                elif vector:
                    self.db.execute("""
                        INSERT INTO product_features 
                        (product_id, category_id, brand_id, shop_id, current_price, text_embedding, last_updated)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """, (row['product_id'], row['category_id'], row['brand_id'], row['shop_id'], row['price'] or 0, vector))
            
            logger.info("Product sync completed.")

        except Exception as e:
            logger.error(f"Failed to sync products: {e}")
            traceback.print_exc()

    # ---------------------------------------------------------
    # 6. TRAIN MODELS (THIẾU)
    # ---------------------------------------------------------
    def train_models(self):
        """
        Kích hoạt quy trình huấn luyện lại Model AI
        """
        logger.info("Starting Model Retraining Job...")
        try:
            # Sử dụng config đã load
            if not self.config:
                logger.error("No config provided for training.")
                return

            # 1. Train CF
            cf_model = train_collaborative_filtering(self.db, self.config)
            
            # 2. Train Content Based (Nếu cần update vector)
            # cb_model = train_content_based(...) # Thường ít chạy cái này vì tốn time, sync daily đã làm rồi.
            
            # 3. Cache Similar Products (Quan trọng để API chạy nhanh)
            # Cần init lại CB Engine tạm thời để cache
            from cb_engine import ContentBasedEngine
            temp_cb = ContentBasedEngine()
            temp_cb.update_database_embeddings(self.db) # Load vector từ DB vào RAM
            
            cache_similar_products(cf_model, temp_cb, self.db)
            
            logger.info("Model Retraining Completed Successfully.")
            
        except Exception as e:
            logger.error(f"Model Retraining Failed: {e}")
            traceback.print_exc()

    # ---------------------------------------------------------
    # 7. CLEANUP OLD DATA (THIẾU)
    # ---------------------------------------------------------
    def cleanup_old_data(self, days_to_keep=180):
        """
        Xóa dữ liệu log quá cũ để giảm tải DB
        """
        logger.info(f"Cleaning up data older than {days_to_keep} days...")
        try:
            # 1. Xóa interaction cũ (chỉ giữ lại purchase)
            self.db.execute("""
                DELETE FROM user_interactions
                WHERE created_at < NOW() - INTERVAL '%s days'
                AND action_type != 'purchase' -- Giữ lại lịch sử mua hàng vĩnh viễn
            """, (days_to_keep,))
            
            # 2. Xóa recommendation logs cũ
            self.db.execute("""
                DELETE FROM recommendation_logs
                WHERE shown_at < NOW() - INTERVAL '%s days'
            """, (days_to_keep,))
            
            logger.info("Cleanup completed.")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    # ---------------------------------------------------------
    # 8. UPDATE USER PROFILES (Bổ sung logic chung)
    # ---------------------------------------------------------
    def update_user_profiles(self):
        """
        Cập nhật các thông tin chung khác của user profile (ngoài spending)
        Ví dụ: last_active_at chính xác, tổng số đơn hàng...
        """
        logger.info("Updating general User Profiles...")
        try:
            self.db.execute("""
                UPDATE user_profiles up
                SET 
                    total_orders = (SELECT COUNT(*) FROM user_interactions ui WHERE ui.user_id = up.user_id AND ui.action_type = 'purchase'),
                    total_spent = (SELECT SUM(price * quantity) FROM user_interactions ui WHERE ui.user_id = up.user_id AND ui.action_type = 'purchase'),
                    profile_updated_at = NOW()
            """)
            logger.info("User profiles updated.")
        except Exception as e:
             logger.error(f"Update user profile failed: {e}")