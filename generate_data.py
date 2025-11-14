import psycopg2
import mysql.connector
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
from bs4 import BeautifulSoup  # pip install beautifulsoup4
from pyvi import ViTokenizer  # pip install pyvi
from faker import Faker
import uuid
import random
from datetime import datetime, timedelta
import numpy as np
import json
import re  # Để clean thêm nếu cần
import torch  # Để check CUDA nếu cần
import pandas as pd  # Để aggregate reviews

# Config 2 MySQL DB (thêm order_db cho reviews)
MYSQL_PRODUCT_HOST = 'localhost'
MYSQL_PRODUCT_PORT = 3306
MYSQL_PRODUCT_DB = 'ecommerce_product_db'
MYSQL_PRODUCT_USER = 'root'  # Giả định user là root, thay nếu khác
MYSQL_PRODUCT_PASS = '101204'

MYSQL_ORDER_HOST = 'localhost'
MYSQL_ORDER_PORT = 3306
MYSQL_ORDER_DB = 'ecommerce_order_db'
MYSQL_ORDER_USER = 'root'
MYSQL_ORDER_PASS = '101204'

# Config Postgres (dữ liệu AI)
PG_HOST = 'localhost'
PG_PORT = 5432
PG_DB = 'shop_service'
PG_USER = 'postgres'  # Thay nếu khác
PG_PASS = '101204'  # Thay bằng pass Postgres

# Khởi tạo Faker cho users/interactions
fake = Faker('vi_VN')

# Model embedding: dangvantuan/vietnamese-embedding (dim=768), force CPU để tránh CUDA issues
model = SentenceTransformer('dangvantuan/vietnamese-embedding', device='cpu')
print(f"Model loaded on CPU (dim=768).")

# Action types và scores (giữ nguyên)
ACTIONS = ['view', 'search', 'cart_add', 'cart_remove', 'purchase', 'wishlist']
ACTION_SCORES = {'view': 1.0, 'search': 0.5, 'cart_add': 3.0, 'cart_remove': -1.0, 'purchase': 10.0, 'wishlist': 5.0}
ACTION_PROBS = {'view': 0.8, 'search': 0.1, 'cart_add': 0.05, 'purchase': 0.03, 'wishlist': 0.02}

def connect_mysql_product():
    return mysql.connector.connect(
        host=MYSQL_PRODUCT_HOST, port=MYSQL_PRODUCT_PORT, database=MYSQL_PRODUCT_DB, user=MYSQL_PRODUCT_USER, password=MYSQL_PRODUCT_PASS
    )

def connect_mysql_order():
    return mysql.connector.connect(
        host=MYSQL_ORDER_HOST, port=MYSQL_ORDER_PORT, database=MYSQL_ORDER_DB, user=MYSQL_ORDER_USER, password=MYSQL_ORDER_PASS
    )

def connect_pg():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def clean_html_description(description):
    """
    Xử lý description: Gỡ HTML tags, extract text sạch, remove repeated phrases.
    Cắt ngắn hơn để tránh token overflow (max ~1000 chars).
    """
    if not description:
        return ""
    
    # Parse HTML với BeautifulSoup
    soup = BeautifulSoup(description, 'html.parser')
    
    # Extract text từ tất cả elements, separator=' ' để giữ cấu trúc
    text = soup.get_text(separator=' ', strip=True)
    
    # Lowercase để tìm patterns
    text_lower = text.lower()
    
    # Remove repeated Tiki disclaimer (ví dụ)
    disclaimer_pattern = r'giá sản phẩm trên tiki đã bao gồm thuế theo luật hiện hành\. bên cạnh đó, tuỳ vào loại sản phẩm, hình thức và địa chỉ giao hàng mà có thể phát sinh thêm chi phí khác như phí vận chuyển, phụ phí hàng cồng kềnh, thuế nhập khẩu \(đối với đơn hàng giao từ nước ngoài có giá trị trên 1 triệu đồng\)\.*'
    text = re.sub(disclaimer_pattern, '', text_lower, flags=re.IGNORECASE | re.DOTALL)
    
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # Remove control chars
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove extra spaces, newlines
    
    # Cắt ngắn: 1000 chars để an toàn với RoBERTa max 512 tokens
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    return text

def extract_real_products_from_mysql():
    """Trích xuất dữ liệu sản phẩm thực tế từ MySQL, clean HTML, tokenize, embedding thật (dim=768)"""
    conn = connect_mysql_product()
    products = []
    try:
        with conn.cursor(dictionary=True) as cur:
            # Query tổng hợp: product + attributes + avg_price từ sku
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
                GROUP BY p.id, p.name, p.description, p.short_description, p.category_id, p.brand_id, p.shop_id, p.image
                ORDER BY p.create_date DESC
                LIMIT 1000  -- Giới hạn để tránh overload, tăng nếu cần
            """)
            rows = cur.fetchall()
            
            for row in rows:
                # Combine name + short_desc + desc + attributes
                full_text = f"{row['name']} {row['short_description']} {row['description']} {row['attributes'] or ''}".strip()
                
                # Clean HTML và noise
                clean_text = clean_html_description(full_text)
                
                if not clean_text:
                    continue  # Skip nếu không có text
                print(clean_text)
                # Tokenize với pyvi
                tokenized_text = ViTokenizer.tokenize(clean_text)
                
                # Debug: Check token length (split để đếm words/tokens)
                token_words = tokenized_text.split()
                token_count = len(token_words)
                print(f"Product {row['product_id'][:8]}...: Token count = {token_count} (safe <512)")
                if token_count > 450:  # Cắt thủ công nếu gần max
                    tokenized_text = ' '.join(token_words[:450])
                    print(f"  -> Truncated to 450 tokens.")
                
                # Sinh embedding với truncate và max_length để tránh IndexError
                try:
                    embedding = model.encode(
                        [tokenized_text],
                        batch_size=1,
                        show_progress_bar=False,
                        normalize_embeddings=True  # Optional, nhưng tốt cho cosine sim
                    )[0].tolist()
                except Exception as e:  # Catch IndexError hoặc RuntimeError
                    print(f"Embedding failed for {row['product_id'][:8]}...: {e}. Using random fallback.")
                    # Fallback: Random vector 768 dims (normalize)
                    embedding = np.random.normal(0, 1, 768).tolist()
                    embedding = [x / np.linalg.norm(embedding) for x in embedding] if np.linalg.norm(embedding) != 0 else embedding
                
                # Chuẩn hóa giá (nếu null)
                current_price = row['current_price'] or random.uniform(100000, 10000000)
                
                products.append({
                    'product_id': row['product_id'],
                    'category_id': row['category_id'],
                    'brand_id': row['brand_id'],
                    'shop_id': row['shop_id'],
                    'current_price': round(current_price, 0),
                    'description': clean_text,  # Lưu text sạch
                    'image': row['image'],
                    'text_embedding': embedding
                })
                print(f"Processed product {row['product_id'][:8]}...: {len(embedding)} dims")
        
        print(f"Extracted and embedded {len(products)} real products from MySQL (dim=768).")
        return products
    
    finally:
        conn.close()

def fetch_product_ratings_from_order_db():
    """Fetch real avg_rating và review_count từ product_comment (order_db), group by product_id"""
    conn = connect_mysql_order()
    try:
        df_ratings = pd.read_sql("""
            SELECT 
                pc.product_id,
                AVG(pc.rating) as avg_rating,
                COUNT(*) as review_count
            FROM product_comment pc
            WHERE pc.created_at >= DATE_SUB(NOW(), INTERVAL 180 DAY)  -- 6 tháng gần nhất
            GROUP BY pc.product_id
        """, conn)
        
        # Convert to dict cho easy lookup
        ratings_dict = dict(zip(df_ratings['product_id'], df_ratings.apply(lambda row: {
            'avg_rating': round(row['avg_rating'], 2) if not pd.isna(row['avg_rating']) else 0,
            'review_count': int(row['review_count'])
        }, axis=1)))
        
        print(f"Fetched ratings for {len(ratings_dict)} products from order_db.")
        return ratings_dict
    
    finally:
        conn.close()

def clear_pg_tables(conn):
    """Clear dữ liệu cũ trong Postgres"""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM recommendation_logs;")
        cur.execute("DELETE FROM user_profiles;")
        cur.execute("DELETE FROM product_features;")
        cur.execute("DELETE FROM user_interactions;")
        conn.commit()
    print("Cleared existing data in Postgres.")

def generate_users(n_users=500):
    """Tạo users giả (UUID)"""
    users = [str(uuid.uuid4()) for _ in range(n_users)]
    return users

def insert_user_interactions_pg(conn, users, products, n_interactions=50000):
    """Insert interactions dựa trên real products"""
    end_date = datetime(2025, 11, 10)  # Ngày hiện tại
    start_date = end_date - timedelta(days=90)
    
    with conn.cursor() as cur:
        for _ in range(n_interactions):
            user_id = random.choice(users)
            product = random.choice(products)
            product_id = product['product_id']
            shop_id = product['shop_id']
            
            action_type = random.choices(list(ACTION_PROBS.keys()), weights=list(ACTION_PROBS.values()))[0]
            score = ACTION_SCORES[action_type]
            quantity = random.randint(1, 5) if action_type in ['cart_add', 'purchase'] else 1
            price = product['current_price'] if action_type in ['purchase', 'cart_add'] else None
            metadata = json.dumps({'session_id': str(uuid.uuid4()), 'device': random.choice(['mobile', 'desktop'])})
            
            created_at = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))
            
            cur.execute("""
                INSERT INTO user_interactions (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, product_id, shop_id, action_type, score, quantity, price, metadata, created_at))
        
        conn.commit()
    print(f"Inserted {n_interactions} interactions using real products.")

def compute_and_insert_user_profiles_pg(conn, users):
    """Tính và insert user_profiles từ interactions (sửa ORDER BY cho aggregate)"""
    with conn.cursor() as cur:
        for user_id in users:
            # Sử dụng subquery để top categories/brands
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
    """Insert product_features với real embedding (768 dims) và real avg_rating/review_count từ ratings_dict"""
    with conn.cursor() as cur:
        for prod in products:
            product_id = prod['product_id']
            category_id = prod['category_id']
            brand_id = prod['brand_id']
            shop_id = prod['shop_id']
            current_price = prod['current_price']
            text_embedding = prod['text_embedding']  # Đã là list 768
            
            # Lấy real avg_rating và review_count từ ratings_dict (từ order_db)
            rating_info = ratings_dict.get(product_id, {'avg_rating': 0, 'review_count': 0})
            avg_rating = rating_info['avg_rating']
            review_count = rating_info['review_count']
            
            # Tính metrics từ interactions (sau khi insert interactions)
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
            
            # Similar products: Giả top 10 (có thể tính real từ embedding similarity sau, dùng cosine)
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
    print("Inserted/Updated product_features with real avg_rating and review_count from order_db.")

def insert_recommendation_logs_pg(conn, users, products, n_logs=10000):
    """Insert logs (giữ nguyên)"""
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
    # Extract real products từ MySQL với cleaning và embedding mới
    products = extract_real_products_from_mysql()
    if not products:
        print("❌ Không tìm thấy sản phẩm trong MySQL. Kiểm tra DB và chạy INSERT data trước.")
        return
    
    # Fetch real ratings từ order_db
    ratings_dict = fetch_product_ratings_from_order_db()
    
    # Kết nối Postgres
    pg_conn = connect_pg()
    try:
        # ALTER column nếu cần (chạy 1 lần, comment sau khi vector(768))
        # with pg_conn.cursor() as cur:
        #     cur.execute("ALTER TABLE product_features ALTER COLUMN text_embedding TYPE vector(768);")
        #     pg_conn.commit()
        # print("Updated text_embedding to vector(768).")
        
        clear_pg_tables(pg_conn)
        
        users = generate_users(500)  # Giữ 500 users
        
        insert_user_interactions_pg(pg_conn, users, products, 50000)
        insert_product_features_pg(pg_conn, products, ratings_dict)  # Pass ratings_dict để dùng real avg_rating
        compute_and_insert_user_profiles_pg(pg_conn, users)
        insert_recommendation_logs_pg(pg_conn, users, products, 10000)
        
        # Refresh view
        with pg_conn.cursor() as cur:
            cur.execute("REFRESH MATERIALIZED VIEW  daily_recommendation_stats;")
            pg_conn.commit()
        
        print("✅ Hoàn tất! Dữ liệu thực tế từ MySQL (cleaned HTML + embedding vietnamese-embedding 768 dims) đã populate vào Postgres.")
        print(f"- Số sản phẩm thực: {len(products)}")
        print("- Số sản phẩm có rating thực: {len(ratings_dict)}")
        print("- Chạy test: SELECT product_id, description, avg_rating FROM product_features LIMIT 5; (xem avg_rating thực)")
        print("- Test similarity: SELECT pf1.product_id, pf2.product_id, (pf1.text_embedding <=> pf2.text_embedding) as distance FROM product_features pf1 CROSS JOIN product_features pf2 WHERE pf1.product_id != pf2.product_id ORDER BY distance LIMIT 5;")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pg_conn.close()

if __name__ == "__main__":
    main()