"""
train.py - Train all AI models (CF + Content-Based)
Run this script to train or retrain models
"""

import yaml
import logging
import os
import sys
from datetime import datetime

from database import Database, MySQLDatabase
from cf_engine import CollaborativeFilteringEngine
from cb_engine import ContentBasedEngine

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
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def train_collaborative_filtering(db: Database, config: dict):
    """
    Train Collaborative Filtering model (ALS)
    
    Args:
        db: Database instance
        config: Configuration dict
    """
    logger.info("=" * 70)
    logger.info("TRAINING COLLABORATIVE FILTERING MODEL")
    logger.info("=" * 70)
    
    cf_config = config.get('model', {}).get('collaborative_filtering', {})
    
    # Initialize CF engine
    cf_engine = CollaborativeFilteringEngine(
        factors=cf_config.get('factors', 64),
        regularization=cf_config.get('regularization', 0.01),
        iterations=cf_config.get('iterations', 15)
    )
    
    # Get training data
    training_days = cf_config.get('training_days', 90)
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
        WHERE created_at >= NOW() - INTERVAL '%s days'
        AND action_type IN ('view', 'cart_add', 'purchase', 'wishlist')
        GROUP BY user_id, product_id
        HAVING SUM(score) > 0
    """, (training_days,))
    
    if interaction_data.empty:
        logger.error(" No interaction data found! Cannot train CF model.")
        logger.info("Please ensure user_interactions table has data.")
        return None
    interaction_data = interaction_data.rename(columns={'final_score': 'score'})
    logger.info(f" Loaded {len(interaction_data)} user-product interactions")
    logger.info(f"  - Unique users: {interaction_data['user_id'].nunique()}")
    logger.info(f"  - Unique products: {interaction_data['product_id'].nunique()}")
    
    # Train model
    start_time = datetime.now()
    cf_engine.train(interaction_data)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f" CF model trained in {training_time:.2f} seconds")
    
    # Get model stats
    stats = cf_engine.get_stats()
    logger.info(f"Model statistics:")
    logger.info(f"  - Users: {stats['num_users']}")
    logger.info(f"  - Products: {stats['num_products']}")
    logger.info(f"  - Interactions: {stats['num_interactions']}")
    logger.info(f"  - Matrix density: {stats['matrix_density']:.4f}%")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"models/cf_model_{timestamp}.pkl"
    cf_engine.save_model(model_path)
    
    # Also save as latest
    cf_engine.save_model("models/cf_model_latest.pkl")
    
    logger.info(f" Model saved to {model_path}")
    
    # Pre-compute similar products and cache to database
    logger.info("Pre-computing similar products...")
    cache_similar_products(cf_engine, db)
    
    return cf_engine


def train_content_based(db: Database, mysql_db: MySQLDatabase, config: dict):
    """
    Train Content-Based model (Text Embeddings)
    Bao gồm cả review content để cải thiện similarity
    
    Args:
        db: AI Database instance
        mysql_db: Main MySQL database instance
        config: Configuration dict
    """
    logger.info("=" * 70)
    logger.info("TRAINING CONTENT-BASED MODEL (WITH REVIEWS)")
    logger.info("=" * 70)
    
    cb_config = config.get('model', {}).get('content_based', {})
    
    # Initialize CB engine với model Vietnamese
    model_name = cb_config.get('text_model', 'dangvantuan/vietnamese-embedding')
    cb_engine = ContentBasedEngine(model_name=model_name)
    
    # Get products from MySQL database
    logger.info("Loading products from main database...")
    
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
            logger.error(" No products found in main database!")
            return None
        
        logger.info(f" Loaded {len(products_df)} products")
        
        # Get reviews for products (để enrich description)
        logger.info("Loading product reviews...")
        reviews_df = db.query("""
            SELECT 
                product_id,
                STRING_AGG(
                    CASE 
                        WHEN rating >= 4 THEN content 
                        ELSE NULL 
                    END, 
                    ' '
                ) as positive_reviews
            FROM product_reviews
            WHERE content IS NOT NULL
            AND LENGTH(content) > 20
            AND rating >= 4
            GROUP BY product_id
        """)
        
        # Merge reviews vào products
        if not reviews_df.empty:
            products_df = products_df.merge(reviews_df, on='product_id', how='left')
            logger.info(f" Loaded reviews for {len(reviews_df)} products")
        else:
            products_df['positive_reviews'] = None
            logger.info("No reviews found, using products only")
        
    except Exception as e:
        logger.error(f" Failed to load products: {e}")
        return None
    
    # Prepare products for batch encoding
    products = []
    for _, row in products_df.iterrows():
        # Combine product info + positive reviews
        combined_text = f"{row['name'] or ''} {row['short_description'] or ''} {row['description'] or ''}"
        
        # Add top positive reviews (max 200 chars)
        if row.get('positive_reviews'):
            review_snippet = str(row['positive_reviews'])[:200]
            combined_text += f" {review_snippet}"
        
        products.append({
            'product_id': row['product_id'],
            'name': row['name'] or '',
            'description': combined_text,
            'category': row['category'] or '',
            'brand': row['brand'] or '',
            'metadata': {
                'shop_id': row['shop_id'],
                'price': float(row['price']) if row['price'] else 0
            }
        })
    
    # Compute embeddings (batch processing)
    start_time = datetime.now()
    cb_engine.add_products_batch(products)
    encoding_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f" Computed embeddings in {encoding_time:.2f} seconds")
    logger.info(f"  - Average time per product: {encoding_time/len(products)*1000:.2f}ms")
    
    # Get stats
    stats = cb_engine.get_stats()
    logger.info(f"Model statistics:")
    logger.info(f"  - Products: {stats['num_products']}")
    logger.info(f"  - Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"  - Model: {stats['model_name']}")
    
    # Save embeddings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    embeddings_path = f"models/cb_embeddings_{timestamp}.pkl"
    cb_engine.save_embeddings(embeddings_path)
    
    # Also save as latest
    cb_engine.save_embeddings("models/cb_embeddings_latest.pkl")
    
    logger.info(f" Embeddings saved to {embeddings_path}")
    
    # Update database with embeddings
    logger.info("Updating embeddings in database...")
    cb_engine.update_database_embeddings(db)
    
    return cb_engine


def cache_similar_products(cf_engine: CollaborativeFilteringEngine, db: Database):
    """
    Pre-compute and cache similar products in database
    
    Args:
        cf_engine: Trained CF engine
        db: Database instance
    """
    if cf_engine is None:
        logger.warning("CF engine is None, skipping similar products caching")
        return
    
    logger.info("Caching similar products to database...")
    
    # Get all products
    products = db.query("SELECT product_id FROM product_features")
    
    if products.empty:
        logger.warning("No products found in database")
        return
    
    cached_count = 0
    failed_count = 0
    
    for _, row in products.iterrows():
        product_id = row['product_id']
        
        try:
            # Get similar products
            similar = cf_engine.similar_products(product_id, n=20)
            
            if similar and len(similar) > 0:
                similar_ids = [p[0] for p in similar]
                
                # Update database
                db.execute("""
                    UPDATE product_features
                    SET similar_product_ids = %s, last_updated = NOW()
                    WHERE product_id = %s
                """, (similar_ids, product_id))
                
                cached_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logger.debug(f"Failed to compute similar for {product_id}: {e}")
            failed_count += 1
    
    logger.info(f" Cached similar products: {cached_count} success, {failed_count} failed")


def test_recommendations(cf_engine: CollaborativeFilteringEngine, 
                         cb_engine: ContentBasedEngine, 
                         db: Database):
    """
    Test trained models with sample queries
    
    Args:
        cf_engine: Trained CF engine (can be None)
        cb_engine: Trained CB engine (can be None)
        db: Database instance
    """
    logger.info("=" * 70)
    logger.info("TESTING MODELS")
    logger.info("=" * 70)
    
    # Test CF recommendations
    if cf_engine is not None:
        logger.info("\n1. Testing Collaborative Filtering:")
        
        # Get a sample user
        sample_user = db.fetchone("""
            SELECT user_id FROM user_interactions 
            LIMIT 1
        """)
        
        if sample_user:
            user_id = sample_user[0]
            logger.info(f"   Getting recommendations for user: {user_id}")
            
            try:
                recs = cf_engine.recommend(user_id, n=5)
                if recs and len(recs) > 0:
                    logger.info(f"    Got {len(recs)} recommendations:")
                    for i, (pid, score) in enumerate(recs[:3], 1):
                        logger.info(f"      {i}. Product {pid[:8]}... (score: {score:.4f})")
                else:
                    logger.warning("    No recommendations returned")
            except Exception as e:
                logger.error(f"    Recommendation failed: {e}")
        else:
            logger.warning("    No users found in database")
    else:
        logger.warning("\n1. Collaborative Filtering: SKIPPED (engine is None)")
    
    # Test CB similar products
    if cb_engine is not None:
        logger.info("\n2. Testing Content-Based Filtering:")
        
        # Get a sample product
        sample_product = db.fetchone("""
            SELECT product_id FROM product_features 
            LIMIT 1
        """)
        
        if sample_product:
            product_id = sample_product[0]
            logger.info(f"   Finding similar products to: {product_id[:8]}...")
            
            try:
                similar = cb_engine.find_similar(product_id, n=5)
                if similar and len(similar) > 0:
                    logger.info(f"    Found {len(similar)} similar products:")
                    for i, (pid, score) in enumerate(similar[:3], 1):
                        logger.info(f"      {i}. Product {pid[:8]}... (similarity: {score:.4f})")
                else:
                    logger.warning("    No similar products found")
            except Exception as e:
                logger.error(f"    Similar search failed: {e}")
        else:
            logger.warning("    No products found")
        
        # Test search
        logger.info("\n3. Testing Content-Based Search:")
        test_query = "điện thoại"
        logger.info(f"   Searching for: '{test_query}'")
        
        try:
            results = cb_engine.search(test_query, n=5)
            if results and len(results) > 0:
                logger.info(f"    Found {len(results)} results:")
                for i, (pid, score) in enumerate(results[:3], 1):
                    logger.info(f"      {i}. Product {pid[:8]}... (score: {score:.4f})")
            else:
                logger.warning("    No search results")
        except Exception as e:
            logger.error(f"    Search failed: {e}")
    else:
        logger.warning("\n2. Content-Based Filtering: SKIPPED (engine is None)")


def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("AI RECOMMENDATION ENGINE - MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load config
    try:
        config = load_config()
        logger.info(" Configuration loaded")
    except Exception as e:
        logger.error(f" Failed to load config: {e}")
        return
    
    # Initialize databases
    try:
        db = Database(config['database'])
        db.connect()
        logger.info(" Connected to PostgreSQL (AI database)")
        
        mysql_db = MySQLDatabase(config['mysql_database'])
        mysql_db.connect()
        logger.info(" Connected to MySQL (main database)\n")
    except Exception as e:
        logger.error(f" Database connection failed: {e}")
        return
    
    # Train Collaborative Filtering
    cf_engine = None
    try:
        cf_engine = train_collaborative_filtering(db, config)
        if cf_engine is None:
            logger.error("CF training failed!")
            cf_success = False
        else:
            cf_success = True
    except Exception as e:
        logger.error(f" CF training error: {e}")
        import traceback
        traceback.print_exc()
        cf_success = False
    
    print()  # Blank line
    
    # Train Content-Based
    cb_engine = None
    try:
        cb_engine = train_content_based(db, mysql_db, config)
        if cb_engine is None:
            logger.error("CB training failed!")
            cb_success = False
        else:
            cb_success = True
    except Exception as e:
        logger.error(f" CB training error: {e}")
        import traceback
        traceback.print_exc()
        cb_success = False
    
    print()  # Blank line
    
    # Test models
    if cf_success or cb_success:
        try:
            test_recommendations(cf_engine, cb_engine, db)
        except Exception as e:
            logger.error(f" Testing error: {e}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    
    if cf_success:
        logger.info(" Collaborative Filtering: SUCCESS")
    else:
        logger.info(" Collaborative Filtering: FAILED")
    
    if cb_success:
        logger.info(" Content-Based Filtering: SUCCESS")
    else:
        logger.info(" Content-Based Filtering: FAILED")
    
    logger.info("\nModels saved in: ./models/")
    logger.info("Training log: ./logs/training.log")
    
    logger.info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Cleanup
    db.close()
    mysql_db.close()


if __name__ == '__main__':
    main()