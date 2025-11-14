"""
load_embeddings_from_db.py - Load embeddings đã có từ database
Dùng khi đã chạy script embedding của bạn rồi
"""

import yaml
import logging
import numpy as np
from database import Database
from cb_engine import ContentBasedEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_embeddings_from_database():
    """
    Load embeddings từ product_features table
    Embeddings đã được tạo bởi script của bạn
    """
    logger.info("=" * 70)
    logger.info("LOADING EMBEDDINGS FROM DATABASE")
    logger.info("=" * 70)
    
    # Load config
    config = load_config()
    
    # Connect to database
    db = Database(config['database'])
    db.connect()
    logger.info(" Connected to database")
    
    # Initialize CB engine (không cần encode lại)
    cb_engine = ContentBasedEngine(model_name='dangvantuan/vietnamese-embedding')
    logger.info(f" CB Engine initialized (dim={cb_engine.embedding_dim})")
    
    # Load products with embeddings từ DB
    logger.info("Loading products and embeddings from database...")
    
    products_df = db.query("""
        SELECT 
            product_id,
            category_id,
            brand_id,
            shop_id,
            current_price,
            text_embedding
        FROM product_features
        WHERE text_embedding IS NOT NULL
    """)
    
    if products_df.empty:
        logger.error(" No products with embeddings found!")
        logger.info("Please run your embedding script first.")
        return None
    
    logger.info(f" Loaded {len(products_df)} products with embeddings")
    
    # Load embeddings into CB engine
    count = 0
    for _, row in products_df.iterrows():
        product_id = row['product_id']
        
        # Embedding đã là list (hoặc string của list)
        embedding = row['text_embedding']
        
        # Convert nếu cần (check type)
        if isinstance(embedding, str):
            # Nếu là string, parse
            import json
            embedding = json.loads(embedding)
        
        # Convert to numpy array
        embedding = np.array(embedding)
        
        # Verify dimension
        if len(embedding) != 768:
            logger.warning(f"Product {product_id}: Wrong dimension {len(embedding)}, expected 768. Skipping.")
            continue
        
        # Store in CB engine
        cb_engine.product_embeddings[product_id] = embedding
        cb_engine.product_metadata[product_id] = {
            'category_id': row['category_id'],
            'brand_id': row['brand_id'],
            'shop_id': row['shop_id'],
            'price': float(row['current_price']) if row['current_price'] else 0
        }
        
        count += 1
    
    logger.info(f" Loaded {count} embeddings into CB engine")
    
    # Save to pickle for quick loading
    cb_engine.save_embeddings('models/cb_embeddings_latest.pkl')
    logger.info(" Saved embeddings to models/cb_embeddings_latest.pkl")
    
    # Test similarity search
    logger.info("\n" + "=" * 70)
    logger.info("TESTING SIMILARITY SEARCH")
    logger.info("=" * 70)
    
    # Get first product
    sample_product = products_df.iloc[0]['product_id']
    logger.info(f"Finding similar products to: {sample_product}")
    
    similar = cb_engine.find_similar(sample_product, n=5)
    
    if similar:
        logger.info(f" Found {len(similar)} similar products:")
        for i, (pid, score) in enumerate(similar, 1):
            logger.info(f"  {i}. {pid} (similarity: {score:.4f})")
    else:
        logger.warning(" No similar products found")
    
    # Update similar_product_ids in database
    logger.info("\n" + "=" * 70)
    logger.info("UPDATING SIMILAR PRODUCTS IN DATABASE")
    logger.info("=" * 70)
    
    update_count = 0
    for product_id in cb_engine.product_embeddings.keys():
        similar = cb_engine.find_similar(product_id, n=20)
        
        if similar:
            similar_ids = [p[0] for p in similar]
            
            db.execute("""
                UPDATE product_features
                SET similar_product_ids = %s, last_updated = NOW()
                WHERE product_id = %s
            """, (similar_ids, product_id))
            
            update_count += 1
    
    logger.info(f" Updated similar_product_ids for {update_count} products")
    
    # Get statistics
    stats = cb_engine.get_stats()
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total products: {stats['num_products']}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Model: {stats['model_name']}")
    
    logger.info("\n✅ DONE! Embeddings loaded and ready to use.")
    logger.info("You can now:")
    logger.info("  1. Run train.py (will skip CB embedding step)")
    logger.info("  2. Run api.py (will load from models/cb_embeddings_latest.pkl)")
    
    db.close()
    return cb_engine


if __name__ == '__main__':
    load_embeddings_from_database()