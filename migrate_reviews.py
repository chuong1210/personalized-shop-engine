"""
migrate_reviews.py - Sync review data from MySQL to PostgreSQL AI database
Supports 2 separate MySQL databases: ecommerce_product_db and ecommerce_order_db
"""

import mysql.connector
import psycopg2
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config MySQL - Product Database
MYSQL_PRODUCT_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'ecommerce_product_db',
    'user': 'root',
    'password': '101204'
}

# Config MySQL - Order Database (có review)
MYSQL_ORDER_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'ecommerce_order_db',
    'user': 'root',
    'password': '101204'
}

# Config Postgres (AI)
PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'shop_service',
    'user': 'postgres',
    'password': '101204'
}


def sync_reviews():
    """
    Sync reviews from MySQL (ecommerce_order_db) to PostgreSQL
    """
    logger.info("Starting review sync...")
    logger.info(f"Source: MySQL {MYSQL_ORDER_CONFIG['database']}")
    logger.info(f"Target: PostgreSQL {PG_CONFIG['database']}")
    
    # Connect to MySQL Order DB (có reviews)
    mysql_order_conn = mysql.connector.connect(**MYSQL_ORDER_CONFIG)
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(**PG_CONFIG)
    
    try:
        # Get reviews from MySQL ecommerce_order_db
        with mysql_order_conn.cursor(dictionary=True) as cur:
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
                WHERE pc.parent_id IS NULL  -- Only main reviews, not replies
                AND pc.rating IS NOT NULL
                GROUP BY pc.comment_id, pc.product_id, pc.sku_id, pc.user_id, 
                         pc.rating, pc.title, pc.content, pc.created_at
                ORDER BY pc.created_at DESC
            """)
            
            reviews = cur.fetchall()
        
        logger.info(f" Found {len(reviews)} reviews in MySQL ecommerce_order_db")
        
        if len(reviews) == 0:
            logger.warning("No reviews found! Make sure product_comment table has data.")
            return
        
        # Clear existing reviews in PostgreSQL
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM product_reviews")
            pg_conn.commit()
        
        logger.info(" Cleared existing reviews in PostgreSQL")
        
        # Insert reviews to PostgreSQL
        inserted = 0
        skipped = 0
        
        with pg_conn.cursor() as cur:
            for review in reviews:
                try:
                    # Simple sentiment based on rating
                    # 5,4 = positive, 3 = neutral, 2,1 = negative
                    rating = review['rating']
                    if rating >= 4:
                        sentiment_score = 0.5 + (rating - 4) * 0.5  # 0.5 to 1.0
                        sentiment_label = 'positive'
                    elif rating == 3:
                        sentiment_score = 0.0
                        sentiment_label = 'neutral'
                    else:
                        sentiment_score = -0.5 - (3 - rating) * 0.25  # -0.5 to -1.0
                        sentiment_label = 'negative'
                    
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
                        sentiment_score,
                        sentiment_label,
                        review['created_at']
                    ))
                    
                    inserted += 1
                    
                except Exception as e:
                    logger.error(f"Failed to insert review {review['review_id']}: {e}")
                    skipped += 1
        
        pg_conn.commit()
        logger.info(f" Inserted {inserted} reviews to PostgreSQL")
        if skipped > 0:
            logger.warning(f" Skipped {skipped} reviews due to errors")
        
        # Update product_features with review metrics
        update_product_review_metrics(pg_conn)
        
        # Update user_profiles with review behavior
        update_user_review_metrics(pg_conn)
        
        logger.info("=" * 60)
        logger.info(" Review sync completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Summary:")
        logger.info(f"  - Total reviews: {len(reviews)}")
        logger.info(f"  - Inserted: {inserted}")
        logger.info(f"  - Skipped: {skipped}")
        
    except Exception as e:
        logger.error(f" Error during sync: {e}")
        raise
    
    finally:
        mysql_order_conn.close()
        pg_conn.close()


def update_product_review_metrics(pg_conn):
    """
    Update product_features with review statistics
    """
    logger.info("Updating product review metrics...")
    
    with pg_conn.cursor() as cur:
        # Update review_count and avg_rating
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
        
        rows_updated = cur.rowcount
        pg_conn.commit()
    
    logger.info(f" Updated review metrics for {rows_updated} products")


def update_user_review_metrics(pg_conn):
    """
    Update user_profiles with review behavior
    """
    logger.info("Updating user review metrics...")
    
    with pg_conn.cursor() as cur:
        # Update user review stats
        cur.execute("""
            INSERT INTO user_profiles (user_id, review_count, avg_rating_given, is_verified_reviewer)
            SELECT 
                user_id,
                COUNT(*) as review_count,
                AVG(rating) as avg_rating_given,
                CASE WHEN COUNT(*) >= 5 AND AVG(helpful_count) >= 2 THEN TRUE ELSE FALSE END as is_verified
            FROM product_reviews
            GROUP BY user_id
            ON CONFLICT (user_id) DO UPDATE SET
                review_count = EXCLUDED.review_count,
                avg_rating_given = EXCLUDED.avg_rating_given,
                is_verified_reviewer = EXCLUDED.is_verified_reviewer,
                profile_updated_at = NOW()
        """)
        
        rows_updated = cur.rowcount
        pg_conn.commit()
    
    logger.info(f" Updated review behavior for {rows_updated} users")


if __name__ == '__main__':
    try:
        sync_reviews()
    except Exception as e:
        logger.error(f"Script failed: {e}")
        exit(1)