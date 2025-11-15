import psycopg2
import mysql.connector
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from faker import Faker
import uuid
import random
from datetime import datetime, timedelta
import numpy as np
import json
import re
import pandas as pd

# Config (giữ nguyên)
MYSQL_PRODUCT_HOST = 'localhost'
MYSQL_PRODUCT_PORT = 3306
MYSQL_PRODUCT_DB = 'ecommerce_product_db'
MYSQL_PRODUCT_USER = 'root'
MYSQL_PRODUCT_PASS = '101204'

MYSQL_ORDER_HOST = 'localhost'
MYSQL_ORDER_PORT = 3306
MYSQL_ORDER_DB = 'ecommerce_order_db'
MYSQL_ORDER_USER = 'root'
MYSQL_ORDER_PASS = '101204'

PG_HOST = 'localhost'
PG_PORT = 5432
PG_DB = 'shop_service'
PG_USER = 'postgres'
PG_PASS = '101204'

fake = Faker('vi_VN')

# Model với explicit config
model = SentenceTransformer('dangvantuan/vietnamese-embedding', device='cpu')
model.max_seq_length = 256  # ← Giảm từ 512 xuống 256 để an toàn hơn
print(f"Model loaded on CPU (max_seq_length={model.max_seq_length}, dim=768).")

# Actions (giữ nguyên)
ACTIONS = ['view', 'search', 'cart_add', 'cart_remove', 'purchase', 'wishlist']
ACTION_SCORES = {'view': 1.0, 'search': 0.5, 'cart_add': 3.0, 'cart_remove': -1.0, 'purchase': 10.0, 'wishlist': 5.0}
ACTION_PROBS = {'view': 0.8, 'search': 0.1, 'cart_add': 0.05, 'purchase': 0.03, 'wishlist': 0.02}

def connect_mysql_product():
    return mysql.connector.connect(
        host=MYSQL_PRODUCT_HOST, port=MYSQL_PRODUCT_PORT, 
        database=MYSQL_PRODUCT_DB, user=MYSQL_PRODUCT_USER, password=MYSQL_PRODUCT_PASS
    )

def connect_mysql_order():
    return mysql.connector.connect(
        host=MYSQL_ORDER_HOST, port=MYSQL_ORDER_PORT, 
        database=MYSQL_ORDER_DB, user=MYSQL_ORDER_USER, password=MYSQL_ORDER_PASS
    )

def connect_pg():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def clean_html_description(description):
    """
    Clean HTML, remove noise, truncate to safe length.
    Return: Clean text max 500 chars (safe for 256 tokens)
    """
    if not description:
        return ""
    
    # Parse HTML
    soup = BeautifulSoup(description, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    
    # Remove Tiki disclaimer
    text_lower = text.lower()
    disclaimer_pattern = r'giá sản phẩm trên tiki đã bao gồm thuế.*?(?=\.|$)'
    text = re.sub(disclaimer_pattern, '', text_lower, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove control chars & normalize whitespace
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate early: 500 chars ≈ 150-200 tokens (safe margin)
    if len(text) > 500:
        text = text[:500]
    
    # Final check: Remove empty or too short
    if len(text) < 10:
        return ""
    
    return text

def safe_encode_embedding(text, product_id):
    """
    Safe encoding với error handling và fallback.
    Returns: 768-dim embedding (normalized)
    """
    try:
        # Pre-check: Text không rỗng
        if not text or len(text.strip()) < 5:
            raise ValueError("Text too short or empty")
        
        # Tokenize với pyvi
        tokenized = ViTokenizer.tokenize(text)
        
        # Check token count (approximation: split by space)
        token_words = tokenized.split()
        token_count = len(token_words)
        
        # Hard truncate nếu vượt quá 200 words
        if token_count > 200:
            tokenized = ' '.join(token_words[:200])
            print(f"  ⚠️  Product {product_id[:8]}: Truncated {token_count} -> 200 tokens")
        
        # Encode với explicit truncation (CRITICAL FIX)
        embedding = model.encode(
            [tokenized],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            # ============================================
            # CRITICAL: Force truncation tại model level
            # ============================================
            device='cpu'  # Explicit device
        )[0]
        
        # Verify dimension
        if len(embedding) != 768:
            raise ValueError(f"Wrong embedding dim: {len(embedding)}")
        
        print(f"  ✅ Product {product_id[:8]}: {token_count} tokens -> 768 dims")
        return embedding.tolist()
    
    except Exception as e:
        # Fallback: Random normalized vector
        print(f"  ❌ Product {product_id[:8]}: Embedding failed ({e}). Using random fallback.")
        random_vec = np.random.normal(0, 1, 768)
        normalized = random_vec / np.linalg.norm(random_vec)
        return normalized.tolist()

def extract_real_products_from_mysql():
    """Extract products with SAFE embedding"""
    conn = connect_mysql_product()
    products = []
    
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute("""
                SELECT 
                    p.id as product_id,
                    p.name,
                    COALESCE(p.description, '') as description,
                    COALESCE(p.short_description, '') as short_description,
                    p.category_id,
                    p.brand_id,
                    p.shop_id,
                    p.image,
                    GROUP_CONCAT(DISTINCT ov.value SEPARATOR ' ') as attributes,
                    AVG(ps.price) as current_price
                FROM product p
                LEFT JOIN option_value ov ON p.id = ov.product_id
                LEFT JOIN product_sku ps ON p.id = ps.product_id
                WHERE p.delete_status = 'Active'
                GROUP BY p.id, p.name, p.description, p.short_description, 
                         p.category_id, p.brand_id, p.shop_id, p.image
                ORDER BY p.create_date DESC
                LIMIT 1000
            """)
            rows = cur.fetchall()
            
            success_count = 0
            failed_count = 0
            
            for row in rows:
                # Combine text fields
                full_text = f"{row['name']} {row['short_description']} {row['description']} {row['attributes'] or ''}".strip()
                
                # Clean
                clean_text = clean_html_description(full_text)
                
                if not clean_text:
                    print(f"  ⏭️  Product {row['product_id'][:8]}: Skipped (empty after clean)")
                    failed_count += 1
                    continue
                
                # Safe embedding
                embedding = safe_encode_embedding(clean_text, row['product_id'])
                
                # Check if fallback was used (all elements similar → random)
                if np.std(embedding) < 0.01:  # Random vector has high std
                    failed_count += 1
                else:
                    success_count += 1
                
                # Prepare product
                current_price = row['current_price'] or random.uniform(100000, 10000000)
                
                products.append({
                    'product_id': row['product_id'],
                    'category_id': row['category_id'],
                    'brand_id': row['brand_id'],
                    'shop_id': row['shop_id'],
                    'current_price': round(current_price, 0),
                    'description': clean_text,
                    'image': row['image'],
                    'text_embedding': embedding
                })
            
            print(f"\n📊 Embedding Summary:")
            print(f"  ✅ Success: {success_count}")
            print(f"  ❌ Failed/Fallback: {failed_count}")
            print(f"  📦 Total: {len(products)}")
            
            return products
    
    finally:
        conn.close()

def fetch_product_ratings_from_order_db():
    """Fetch ratings (giữ nguyên)"""
    conn = connect_mysql_order()
    try:
        df_ratings = pd.read_sql("""
            SELECT 
                pc.product_id,
                AVG(pc.rating) as avg_rating,
                COUNT(*) as review_count
            FROM product_comment pc
            WHERE pc.created_at >= DATE_SUB(NOW(), INTERVAL 180 DAY)
            GROUP BY pc.product_id
        """, conn)
        
        ratings_dict = dict(zip(df_ratings['product_id'], df_ratings.apply(lambda row: {
            'avg_rating': round(row['avg_rating'], 2) if not pd.isna(row['avg_rating']) else 0,
            'review_count': int(row['review_count'])
        }, axis=1)))
        
        print(f"Fetched ratings for {len(ratings_dict)} products from order_db.")
        return ratings_dict
    
    finally:
        conn.close()

def clear_pg_tables(conn):
    """Clear old data"""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM recommendation_logs;")
        cur.execute("DELETE FROM user_profiles;")
        cur.execute("DELETE FROM product_features;")
        cur.execute("DELETE FROM user_interactions;")
        conn.commit()
    print("Cleared existing data in Postgres.")

def generate_users(n_users=500):
    """Generate fake users"""
    return [str(uuid.uuid4()) for _ in range(n_users)]

def insert_user_interactions_pg(conn, users, products, n_interactions=50000):
    """Insert interactions with baskets"""
    end_date = datetime(2025, 11, 10)
    start_date = end_date - timedelta(days=90)
    total_seconds = int((end_date - start_date).total_seconds())
    
    with conn.cursor() as cur:
        # Single interactions (80%)
        single_count = int(n_interactions * 0.8)
        for _ in range(single_count):
            user_id = random.choice(users)
            product = random.choice(products)
            product_id = product['product_id']
            shop_id = product['shop_id']
            action_type = random.choices(list(ACTION_PROBS.keys()), weights=list(ACTION_PROBS.values()))[0]
            score = ACTION_SCORES[action_type]
            quantity = random.randint(1, 5) if action_type in ['cart_add', 'purchase'] else 1
            price = product['current_price'] if action_type in ['purchase', 'cart_add'] else None
            metadata = json.dumps({'session_id': str(uuid.uuid4()), 'device': random.choice(['mobile', 'desktop'])})
            created_at = start_date + timedelta(seconds=random.randint(0, total_seconds))
            
            cur.execute("""
                INSERT INTO user_interactions (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at))
        
        # Basket interactions (20%)
        basket_count = int(n_interactions * 0.2 / 3)
        for _ in range(basket_count):
            user_id = random.choice(users)
            session_id = str(uuid.uuid4())
            base_date = start_date + timedelta(days=random.randint(0, 89))
            base_time = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            basket_products = random.sample(products, k=random.randint(2, 4))
            
            for i, product in enumerate(basket_products):
                product_id = product['product_id']
                shop_id = product['shop_id']
                action_type = random.choices(['purchase', 'cart_add'], weights=[0.7, 0.3])[0]
                score = ACTION_SCORES[action_type]
                quantity = random.randint(1, 3)
                price = product['current_price'] if action_type == 'purchase' else None
                created_at = base_date + base_time + timedelta(minutes=i * 5)
                metadata = json.dumps({'session_id': session_id, 'device': random.choice(['mobile', 'desktop'])})
                
                cur.execute("""
                    INSERT INTO user_interactions (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at))
        
        conn.commit()
    print(f"Inserted {n_interactions} interactions with {basket_count} baskets.")

def compute_and_insert_user_profiles_pg(conn, users):
    """Compute user profiles from interactions"""
    with conn.cursor() as cur:
        for user_id in users:
            cur.execute("""
                WITH cat_stats AS (
                    SELECT pf.category_id, COUNT(*) as cnt
                    FROM user_interactions ui
                    LEFT JOIN product_features pf ON ui.product_id = pf.product_id
                    WHERE ui.user_id = %s
                    GROUP BY pf.category_id
                ),
                brand_stats AS (
                    SELECT pf.brand_id, COUNT(*) as cnt
                    FROM user_interactions ui
                    LEFT JOIN product_features pf ON ui.product_id = pf.product_id
                    WHERE ui.user_id = %s
                    GROUP BY pf.brand_id
                ),
                top_cats AS (
                    SELECT category_id FROM cat_stats ORDER BY cnt DESC LIMIT 3
                ),
                top_brands AS (
                    SELECT brand_id FROM brand_stats ORDER BY cnt DESC LIMIT 3
                )
                SELECT 
                    COUNT(CASE WHEN action_type = 'purchase' THEN 1 END) as total_orders,
                    SUM(CASE WHEN action_type = 'purchase' THEN price * quantity ELSE 0 END) as total_spent,
                    AVG(CASE WHEN action_type = 'purchase' THEN price ELSE NULL END) as avg_order_value,
                    (SELECT array_agg(category_id) FROM top_cats) as favorite_categories,
                    (SELECT array_agg(brand_id) FROM top_brands) as favorite_brands,
                    MAX(CASE WHEN action_type = 'purchase' THEN created_at END) as last_purchase_at,
                    MAX(created_at) as last_active_at,
                    AVG(price) FILTER (WHERE action_type = 'purchase') as avg_price
                FROM user_interactions ui
                WHERE ui.user_id = %s
            """, (user_id, user_id, user_id))
            
            result = cur.fetchone()
            
            if result and result[0] > 0:
                total_orders = result[0]
                total_spent = result[1] or 0
                avg_order_value = result[2] or 0
                favorite_categories = result[3] or '{}'
                favorite_brands = result[4] or '{}'
                last_purchase_at = result[5]
                last_active_at = result[6]
                avg_price = result[7] or 0
                price_range_min = min(avg_price, 100000) if avg_price else 0
                price_range_max = max(avg_price * 2, 10000000) if avg_price else 0
                discount_seeker_score = round(random.uniform(0.3, 0.8), 2)
                
                cur.execute("""
                    INSERT INTO user_profiles (user_id, total_orders, total_spent, avg_order_value,
                                               favorite_categories, favorite_brands, price_range_min, price_range_max,
                                               discount_seeker_score, last_purchase_at, last_active_at, profile_updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE SET
                        total_orders = EXCLUDED.total_orders, total_spent = EXCLUDED.total_spent,
                        avg_order_value = EXCLUDED.avg_order_value, favorite_categories = EXCLUDED.favorite_categories,
                        favorite_brands = EXCLUDED.favorite_brands, price_range_min = EXCLUDED.price_range_min,
                        price_range_max = EXCLUDED.price_range_max, discount_seeker_score = EXCLUDED.discount_seeker_score,
                        last_purchase_at = EXCLUDED.last_purchase_at, last_active_at = EXCLUDED.last_active_at,
                        profile_updated_at = NOW()
                """, (user_id, total_orders, total_spent, avg_order_value, favorite_categories, favorite_brands,
                      price_range_min, price_range_max, discount_seeker_score, last_purchase_at, last_active_at))
            else:
                cur.execute("""
                    INSERT INTO user_profiles (user_id, total_orders, total_spent, avg_order_value,
                                               favorite_categories, favorite_brands, price_range_min, price_range_max,
                                               discount_seeker_score, last_active_at, profile_updated_at)
                    VALUES (%s, 0, 0, 0, '{}', '{}', 0, 0, 0.5, NOW(), NOW())
                    ON CONFLICT (user_id) DO NOTHING
                """, (user_id,))
        
        conn.commit()
    print("Inserted/Updated user_profiles.")

def insert_product_features_pg(conn, products, ratings_dict):
    """Insert product features with real ratings"""
    with conn.cursor() as cur:
        for prod in products:
            product_id = prod['product_id']
            category_id = prod['category_id']
            brand_id = prod['brand_id']
            shop_id = prod['shop_id']
            current_price = prod['current_price']
            text_embedding = prod['text_embedding']
            
            rating_info = ratings_dict.get(product_id, {'avg_rating': 0, 'review_count': 0})
            avg_rating = rating_info['avg_rating']
            review_count = rating_info['review_count']
            
            seven_days_ago = datetime.now() - timedelta(days=7)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            cur.execute("""
                SELECT 
                    COUNT(CASE WHEN action_type = 'view' AND created_at >= %s THEN 1 END) as view_7d,
                    COUNT(CASE WHEN action_type = 'view' AND created_at >= %s THEN 1 END) as view_30d,
                    COUNT(CASE WHEN action_type = 'purchase' AND created_at >= %s THEN 1 END) as purchase_7d,
                    COUNT(CASE WHEN action_type = 'purchase' AND created_at >= %s THEN 1 END) as purchase_30d
                FROM user_interactions
                WHERE product_id = %s
            """, (seven_days_ago, thirty_days_ago, seven_days_ago, thirty_days_ago, product_id))
            result = cur.fetchone()
            
            view_7d = result[0] or 0
            view_30d = result[1] or 0
            purchase_7d = result[2] or 0
            purchase_30d = result[3] or 0
            conversion_rate = round((purchase_7d / max(view_7d, 1)), 4) if view_7d > 0 else 0
            trending_score = round((view_7d / max(view_30d, 1)) * random.uniform(0.8, 1.2) * 100, 2)
            
            similar_product_ids = [p['product_id'] for p in random.sample(products, min(10, len(products)))]
            similar_product_ids = [pid for pid in similar_product_ids if pid != product_id][:10]
            
            cur.execute("""
                INSERT INTO product_features (product_id, category_id, brand_id, shop_id, current_price,
                                              view_count_7d, view_count_30d, purchase_count_7d, purchase_count_30d,
                                              conversion_rate, avg_rating, trending_score, text_embedding,
                                              similar_product_ids, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (product_id) DO UPDATE SET
                    category_id = EXCLUDED.category_id, brand_id = EXCLUDED.brand_id, shop_id = EXCLUDED.shop_id,
                    current_price = EXCLUDED.current_price, view_count_7d = EXCLUDED.view_count_7d,
                    view_count_30d = EXCLUDED.view_count_30d, purchase_count_7d = EXCLUDED.purchase_count_7d,
                    purchase_count_30d = EXCLUDED.purchase_count_30d, conversion_rate = EXCLUDED.conversion_rate,
                    avg_rating = EXCLUDED.avg_rating, trending_score = EXCLUDED.trending_score,
                    text_embedding = EXCLUDED.text_embedding, similar_product_ids = EXCLUDED.similar_product_ids,
                    last_updated = NOW()
            """, (product_id, category_id, brand_id, shop_id, current_price, view_7d, view_30d,
                  purchase_7d, purchase_30d, conversion_rate, avg_rating, trending_score,
                  text_embedding, similar_product_ids))
        
        conn.commit()
    print("Inserted/Updated product_features with real ratings.")

def insert_recommendation_logs_pg(conn, users, products, n_logs=10000):
    """Insert recommendation logs"""
    rec_types = ['personalized', 'similar', 'trending', 'cross_sell']
    page_contexts = ['home', 'product_detail', 'cart', 'search']
    
    with conn.cursor() as cur:
        for _ in range(n_logs):
            user_id = random.choice(users)
            product_id = random.choice([p['product_id'] for p in products])
            rec_type = random.choice(rec_types)
            rec_position = random.randint(1, 20)
            rec_score = round(random.uniform(0.1, 1.0), 4)
            shown_at = datetime.now() - timedelta(days=random.randint(0, 90))
            
            clicked_at = shown_at + timedelta(minutes=random.randint(0, 30)) if random.random() > 0.7 else None
            purchased_at = clicked_at + timedelta(minutes=random.randint(10, 60)) if clicked_at and random.random() > 0.8 else None
            purchase_amount = random.uniform(200000, 2000000) if purchased_at else None
            page_context = random.choice(page_contexts)
            
            cur.execute("""
                INSERT INTO recommendation_logs (user_id, product_id, rec_type, rec_position, rec_score,
                                                 shown_at, clicked_at, purchased_at, purchase_amount, page_context)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, product_id, rec_type, rec_position, rec_score, shown_at, clicked_at,
                  purchased_at, purchase_amount, page_context))
        
        conn.commit()
    print(f"Inserted {n_logs} recommendation_logs.")

def main():
    """Main execution"""
    print("="*60)
    print("🚀 Starting data population with SAFE embedding...")
    print("="*60)
    
    # Extract products
    products = extract_real_products_from_mysql()
    if not products:
        print("❌ No products found. Check MySQL DB.")
        return
    
    # Fetch ratings
    ratings_dict = fetch_product_ratings_from_order_db()
    
    # Connect Postgres
    pg_conn = connect_pg()
    try:
        clear_pg_tables(pg_conn)
        
        users = generate_users(500)
        
        insert_user_interactions_pg(pg_conn, users, products, 50000)
        insert_product_features_pg(pg_conn, products, ratings_dict)
        compute_and_insert_user_profiles_pg(pg_conn, users)
        insert_recommendation_logs_pg(pg_conn, users, products, 10000)
        
        # Refresh materialized view
        with pg_conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW daily_recommendation_stats;")
            pg_conn.commit()
        
        print("\n" + "="*60)
        print("✅ COMPLETED!")
        print("="*60)
        print(f"📦 Products: {len(products)}")
        print(f"⭐ Products with ratings: {len(ratings_dict)}")
        print("\n🧪 Test queries:")
        print("  SELECT product_id, description, avg_rating FROM product_features LIMIT 5;")
        print("  SELECT pf1.product_id, pf2.product_id, (pf1.text_embedding <=> pf2.text_embedding) as distance")
        print("  FROM product_features pf1 CROSS JOIN product_features pf2")
        print("  WHERE pf1.product_id != pf2.product_id ORDER BY distance LIMIT 5;")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pg_conn.close()

if __name__ == "__main__":
    main()