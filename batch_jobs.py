"""
batch_jobs_updated.py - Batch jobs with review sync capability
"""

import logging
from datetime import datetime
from database import Database
from recommend_service import RecommendationService

logger = logging.getLogger(__name__)


class BatchJobs:
    """Scheduled batch jobs for maintaining recommendation system"""
    
    def __init__(self, db: Database, service: RecommendationService = None):
        self.db = db
        self.service = service
        logger.info("BatchJobs initialized")
    
    # ... (keep all existing methods) ...
    
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