"""
test_connections.py - Test all database connections
Chạy file này để verify config trước khi sync/train
"""

import mysql.connector
import psycopg2
import redis
import yaml

def test_mysql_product():
    """Test MySQL Product Database connection"""
    print("\n1. Testing MySQL Product Database (ecommerce_product_db)...")
    try:
        conn = mysql.connector.connect(
            host='localhost',
            port=3306,
            database='ecommerce_product_db',
            user='root',
            password='101204'
        )
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM product")
        count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM category")
        cat_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM brand")
        brand_count = cur.fetchone()[0]
        
        print(f"    Connected to ecommerce_product_db")
        print(f"   - Products: {count}")
        print(f"   - Categories: {cat_count}")
        print(f"   - Brands: {brand_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def test_mysql_order():
    """Test MySQL Order Database connection"""
    print("\n2. Testing MySQL Order Database (ecommerce_order_db)...")
    try:
        conn = mysql.connector.connect(
            host='localhost',
            port=3306,
            database='ecommerce_order_db',
            user='root',
            password='101204'
        )
        
        cur = conn.cursor()
        
        # Check if tables exist
        cur.execute("SHOW TABLES")
        tables = [table[0] for table in cur.fetchall()]
        
        has_comment = 'product_comment' in tables
        has_likes = 'review_likes' in tables
        
        if has_comment:
            cur.execute("SELECT COUNT(*) FROM product_comment")
            comment_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM product_comment WHERE rating IS NOT NULL")
            rated_count = cur.fetchone()[0]
            
            print(f"    Connected to ecommerce_order_db")
            print(f"   - Total comments: {comment_count}")
            print(f"   - Comments with rating: {rated_count}")
        else:
            print(f"    Connected but product_comment table not found")
            print(f"   - Available tables: {', '.join(tables)}")
        
        if has_likes:
            cur.execute("SELECT COUNT(*) FROM review_likes")
            likes_count = cur.fetchone()[0]
            print(f"   - Review likes: {likes_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def test_postgresql():
    """Test PostgreSQL AI Database connection"""
    print("\n3. Testing PostgreSQL AI Database (shop_service)...")
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='shop_service',
            user='postgres',
            password='101204'
        )
        
        cur = conn.cursor()
        
        # Check if AI tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name IN ('user_interactions', 'product_features', 
                               'user_profiles', 'recommendation_logs', 'product_reviews')
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        print(f"    Connected to shop_service")
        print(f"   - AI tables found: {len(tables)}/5")
        
        for table in ['user_interactions', 'product_features', 'user_profiles', 
                      'recommendation_logs', 'product_reviews']:
            if table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"      {table}: {count} rows")
            else:
                print(f"      {table}: NOT FOUND")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"    Failed: {e}")
        return False


def test_redis():
    """Test Redis connection"""
    print("\n4. Testing Redis...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        
        # Get info
        info = r.info()
        
        print(f"    Redis connected")
        print(f"   - Version: {info['redis_version']}")
        print(f"   - Used memory: {info['used_memory_human']}")
        
        return True
        
    except Exception as e:
        print(f"    Failed: {e}")
        print(f"   Tip: Start Redis with 'redis-server'")
        return False


def main():
    print("="*70)
    print("DATABASE CONNECTION TEST")
    print("="*70)
    
    results = []
    
    results.append(("MySQL Product DB", test_mysql_product()))
    results.append(("MySQL Order DB", test_mysql_order()))
    results.append(("PostgreSQL AI DB", test_postgresql()))
    results.append(("Redis Cache", test_redis()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, status in results:
        icon = "" if status else ""
        status_text = "OK" if status else "FAILED"
        print(f"{icon} {name}: {status_text}")
    
    all_ok = all(status for _, status in results)
    
    if all_ok:
        print("\nAll connections OK! Ready to sync and train.")
        print("\nNext steps:")
        print("  1. python migrate_reviews.py  # Sync reviews")
        print("  2. python train.py            # Train models")
        print("  3. python api.py              # Start API")
    else:
        print("\n Some connections failed. Fix errors above before continuing.")
    
    print("="*70)


if __name__ == '__main__':
    main()