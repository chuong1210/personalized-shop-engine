"""
train.py - Train all AI models (CF + Content-Based + Hybrid)
Run this script to train or retrain models.
Includes fixes for Windows Threading crashes.
"""

import os
import sys

# =============================================================================
# CRITICAL FIX FOR WINDOWS & LIGHTFM/OPENBLAS CRASHES
# Phải đặt biến môi trường trước khi import numpy/lightfm
# =============================================================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml
import logging
import traceback
import numpy as np
from datetime import datetime

# Import custom modules
from database import Database, MySQLDatabase
from cf_engine import CollaborativeFilteringEngine
from cb_engine import ContentBasedEngine

# Try importing HybridEngine gracefully
try:
    from hybrid_engine import HybridEngine
except ImportError:
    HybridEngine = None
    print("Warning: hybrid_engine.py not found or failed to import.")

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from yaml file"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def train_collaborative_filtering(db: Database, config: dict):
    """
    Train Collaborative Filtering model (ALS)
    """
    logger.info("=" * 70)
    logger.info("TRAINING COLLABORATIVE FILTERING MODEL (ALS)")
    logger.info("=" * 70)
    
    cf_config = config.get('model', {}).get('collaborative_filtering', {})
    
    # Initialize CF engine
    cf_engine = CollaborativeFilteringEngine(
        factors=cf_config.get('factors', 64),
        regularization=cf_config.get('regularization', 0.01),
        iterations=cf_config.get('iterations', 15)
    )
    
    # Get training data with Time Decay logic
    training_days = cf_config.get('training_days', 30)
    logger.info(f"Loading interaction data from last {training_days} days...")
    
    interaction_data = db.query("""
        SELECT 
            user_id,
            product_id,
            SUM(
                score * 
                EXP(-EXTRACT(EPOCH FROM (NOW() - created_at)) / (30 * 86400))
            ) as final_score
        FROM user_interactions
      WHERE created_at >= NOW() - INTERVAL '%s days' AND
         action_type IN ('view', 'cart_add', 'purchase', 'wishlist')
        GROUP BY user_id, product_id
        HAVING SUM(score) > 0
    """, (training_days,))
    
    if interaction_data.empty:
        logger.error("No interaction data found! Skipping CF training.")
        return None
        
    interaction_data = interaction_data.rename(columns={'final_score': 'score'})
    logger.info(f"Loaded {len(interaction_data)} interactions.")
    logger.info(f"Unique users: {interaction_data['user_id'].nunique()}")
    logger.info(f"Unique products: {interaction_data['product_id'].nunique()}")
    
    # Train model
    start_time = datetime.now()
    cf_engine.train(interaction_data)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"CF model trained in {training_time:.2f} seconds")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"models/cf_model_{timestamp}.pkl"
    cf_engine.save_model(model_path)
    cf_engine.save_model("models/cf_model_latest.pkl")
    logger.info(f"Model saved to {model_path}")
    
    # Cache similar products based on Matrix Factorization
    # cache_similar_products(cf_engine, db)
    
    return cf_engine


def train_hybrid_model(db: Database, config: dict):
    """
    Train Hybrid Model (LightFM) and update vectors to DB
    """
    if HybridEngine is None:
        logger.error("HybridEngine class missing. Skipping Hybrid training.")
        return None

    logger.info("=" * 70)
    logger.info("TRAINING HYBRID MODEL (LIGHTFM)")
    logger.info("=" * 70)
    
    try:
        # 1. Load Interactions (Positive feedback only)
        logger.info("Loading interaction data...")
        interaction_df = db.query("""
            SELECT user_id, product_id, 1.0 as score 
            FROM user_interactions 
            WHERE created_at >= NOW() - INTERVAL '90 days'
            AND action_type IN ('purchase', 'cart_add', 'view', 'wishlist')
        """)
        
        if interaction_df.empty:
            logger.warning("No interactions found for Hybrid model.")
            return None
            
        # 2. Load Item Features
        logger.info("Loading product features...")
        products_df = db.query("""
            SELECT product_id, category_id, brand_id
            FROM product_features
        """)
        
        if products_df.empty:
            logger.warning("No product features found.")
            return None
        
        # 3. Train Model
        # Important: Using Warp loss for ranking
        engine = HybridEngine(no_components=64, loss='warp')
        logger.info("Starting training (Single Threaded)...")
        engine.train(interaction_df, products_df)
        
        # 4. Save Model
        engine.save_model("models/hybrid_model_latest.pkl")
        logger.info("Hybrid model saved.")
        
        # 5. Update Vectors to PostgreSQL (for pgvector search)
        logger.info("Updating product vectors to Database...")
        
        valid_products = products_df['product_id'].tolist()
        logger.info("Updating hybrid vectors to Postgres...")
        
        update_data = []
        # Lấy danh sách product_id từ dataframe đã load
        for pid in products_df['product_id'].tolist():
            vector = engine.get_item_vector(pid)
            # Kiểm tra vector có hợp lệ và đúng chiều không
            if vector and len(vector) == 64:
                update_data.append((vector, pid))

        if update_data:
            # Batch update to avoid memory issues
            batch_size = 1000
            total_updated = 0
            
            with db.connect() as conn:
                with conn.cursor() as cur:
                    # Dùng thư viện psycopg2.extras.execute_batch thì tốt hơn, 
                    # nhưng ở đây dùng loop đơn giản để tránh phụ thuộc
                    for i in range(0, len(update_data), batch_size):
                        batch = update_data[i:i + batch_size]
                        # Lưu ý: Cập nhật vào cột 'text_embedding' để tái sử dụng logic tìm kiếm
                        # Hoặc tạo cột 'hybrid_embedding' riêng nếu muốn
                        cur.executemany("""
                            UPDATE product_features 
                            SET hybrid_embedding  = %s::vector
                            WHERE product_id = %s
                        """, batch)
                        conn.commit()
                        total_updated += len(batch)
                        
            logger.info(f"Successfully updated vectors for {total_updated} products.")
        else:
            logger.warning("No vectors generated to update.")
            
        return engine

    except Exception as e:
        logger.error(f"Hybrid Training Error: {e}")
        traceback.print_exc()
        return None


def train_content_based(db: Database, mysql_db: MySQLDatabase, config: dict):
    """
    Train Content-Based model (Text Embeddings with BERT)
    Optional if Hybrid model is used, but good for cold-start products.
    """
    logger.info("=" * 70)
    logger.info("TRAINING CONTENT-BASED MODEL (BERT)")
    logger.info("=" * 70)
    
    cb_config = config.get('model', {}).get('content_based', {})
    model_name = cb_config.get('text_model', 'dangvantuan/vietnamese-embedding')
    
    cb_engine = ContentBasedEngine(model_name=model_name)
    
    logger.info("Loading products from Main MySQL DB...")
    
    try:
        products_df = mysql_db.query("""
            SELECT 
                p.id as product_id,
                p.name as name,
                p.description as description,
                p.short_description,
                c.name as category,
                b.name as brand,
                p.shop_id,
                (SELECT AVG(ps.price) FROM product_sku ps WHERE ps.product_id = p.id) as price
            FROM product p
            LEFT JOIN category c ON p.category_id = c.category_id
            LEFT JOIN brand b ON p.brand_id = b.brand_id
            WHERE p.delete_status = 'Active'
            LIMIT 10000
        """)
        
        if products_df.empty:
            logger.warning("No products found in MySQL.")
            return None
            
        logger.info(f"Loaded {len(products_df)} products from MySQL.")
        
        # Load Reviews to enrich data
        logger.info("Loading top reviews from AI DB...")
        reviews_df = db.query("""
            SELECT product_id, STRING_AGG(LEFT(content, 200), ' ') as reviews
            FROM product_reviews
            WHERE rating >= 4 AND content IS NOT NULL
            GROUP BY product_id
        """)
        
        if not reviews_df.empty:
            products_df = products_df.merge(reviews_df, on='product_id', how='left')
        
        # Prepare for encoding
        products_list = []
        for _, row in products_df.iterrows():
            desc = f"{row['name']} {row['short_description'] or ''} {row['description'] or ''}"
            if 'reviews' in row and row['reviews']:
                desc += f" {str(row['reviews'])[:500]}" # Limit review length
                
            products_list.append({
                'product_id': row['product_id'],
                'name': row['name'],
                'description': desc,
                'category': row['category'],
                'brand': row['brand']
            })
            
        # Encode
        start_time = datetime.now()
        cb_engine.add_products_batch(products_list)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Encoded {len(products_list)} products in {duration:.2f}s")
        
        # Save
        cb_engine.save_embeddings("models/cb_embeddings_latest.pkl")
        
        # Update DB (Only if Hybrid model didn't update already, or separate column)
        # For now, we skip updating DB here to avoid overwriting Hybrid vectors
        cb_engine.update_database_embeddings(db)
        
        return cb_engine

    except Exception as e:
        logger.error(f"CB Training Error: {e}")
        traceback.print_exc()
        return None
def cache_similar_products(cf_engine, cb_engine, db):
    """
    Cache similar products: Ưu tiên CF, nếu không có thì dùng Content-Based (Embedding)
    """
    logger.info("Caching similar products to database (Hybrid Strategy)...")
    
    # Lấy tất cả sản phẩm
    products = db.query("SELECT product_id FROM product_features")
    
    if products.empty:
        logger.warning("No products found in database")
        return
    
    cached_count = 0
    cf_count = 0
    cb_count = 0
    
    # Prepare batch update data
    updates = []
    
    for _, row in products.iterrows():
        product_id = row['product_id']
        similar_ids = []
        source = ""

        # CACH 1: Thử dùng Collaborative Filtering (Hành vi)
        if cf_engine:
            try:
                # Tìm tương đồng dựa trên hành vi người dùng
                cf_sim = cf_engine.similar_products(product_id, n=20)
                if cf_sim:
                    similar_ids = [p[0] for p in cf_sim]
                    source = "CF"
                    cf_count += 1
            except Exception:
                pass
        
        # CACH 2: Nếu CF thất bại (Sản phẩm mới/Cold Start), dùng Content-Based (Embedding)
        if not similar_ids and cb_engine:
            try:
                # Tìm tương đồng dựa trên nội dung (Tên/Mô tả/Hình ảnh)
                # Lưu ý: Hàm này dùng vector search (như ta đã làm ở các bước trước)
                cb_sim = cb_engine.find_similar(product_id, n=20)
                if cb_sim:
                    similar_ids = [p[0] for p in cb_sim]
                    source = "CB"
                    cb_count += 1
            except Exception:
                pass
        
        # Nếu tìm được (bằng cách này hay cách kia) thì lưu vào DB
        if similar_ids:
            updates.append((similar_ids, product_id))
            cached_count += 1
            
            # Batch update mỗi 100 items
            if len(updates) >= 100:
                _batch_update_similar(db, updates)
                updates = []
                print(f"Cached {cached_count}/{len(products)} products...", end='\r')

    # Update nốt phần còn lại
    if updates:
        _batch_update_similar(db, updates)
    
    print("") # Xuống dòng
    logger.info(f" Finished Caching.")
    logger.info(f"   - Used Collaborative Filtering (Behavior): {cf_count}")
    logger.info(f"   - Used Content-Based (Embedding): {cb_count} (Cold Start Fixed)")

def _batch_update_similar(db, data):
    """Helper để update batch vào DB"""
    try:
        db.execute_many("""
            UPDATE product_features
            SET similar_product_ids = %s, last_updated = NOW()
            WHERE product_id = %s
        """, data)
    except Exception as e:
        logger.error(f"Batch update failed: {e}")
def main():
    logger.info("=" * 70)
    logger.info("AI RECOMMENDATION ENGINE - TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Load Config
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Config Error: {e}")
        return

    # Connect DBs
    try:
        db = Database(config['database'])
        mysql_db = MySQLDatabase(config['mysql_database'])
        # Test connections
        db.fetchone("SELECT 1")
        mysql_db.query("SELECT 1")
        logger.info("Databases connected.")
    except Exception as e:
        logger.error(f"DB Connection Error: {e}")
        return

    # 1. Train CF
    cf_model = None
    try:
        cf_model = train_collaborative_filtering(db, config)
    except Exception:
        logger.error("CF Training crashed.")
        traceback.print_exc()

    # 2. Train Hybrid (LightFM)
    hybrid_model = None
    # try:
    #     hybrid_model = train_hybrid_model(db, config)
    # except Exception:
    #     logger.error("Hybrid Training crashed.")
    #     traceback.print_exc()
        
    # 3. Train Content-Based (Optional - Uncomment if needed)
    cb_model = None
    try:
        cb_model = train_content_based(db, mysql_db, config)
    except Exception:
        logger.error("CB Training crashed.")
        traceback.print_exc()
    cache_similar_products(cf_model, cb_model, db)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"CF Model: {'SUCCESS' if cf_model else 'FAILED'}")
    logger.info(f"CB Model: {'SUCCESS' if cb_model else 'FAILED'}")

    logger.info(f"Hybrid Model: {'SUCCESS' if hybrid_model else 'FAILED'}")
    
    logger.info("=" * 70)
    
    db.close()
    mysql_db.close()


if __name__ == '__main__':
    main()