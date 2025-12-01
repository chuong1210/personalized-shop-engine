"""
service.py - Main Recommendation Service
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import redis

from database import Database
from cf_engine import CollaborativeFilteringEngine
from cb_engine import ContentBasedEngine

import os
logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Main recommendation service that combines multiple recommendation strategies
    """
    
    def __init__(self, db: Database, redis_client: redis.Redis, config: dict):
        """
        Initialize recommendation service
        
        Args:
            db: Database instance
            redis_client: Redis client for caching
            config: Configuration dictionary
        """
        self.db = db
        self.redis = redis_client
        self.config = config
        
        # Initialize CF engine
        cf_config = config.get('model', {}).get('collaborative_filtering', {})
        self.cf_engine = CollaborativeFilteringEngine(
            factors=cf_config.get('factors', 64),
            regularization=cf_config.get('regularization', 0.01),
            iterations=cf_config.get('iterations', 15)
        )

        # --- BẮT BUỘC THÊM ĐOẠN NÀY ---
        # Load model đã train từ file (nếu có)
        model_path = "models/cf_model_latest.pkl"
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading trained CF model from {model_path}...")
                self.cf_engine.load_model(model_path)
                logger.info("✅ CF Model loaded successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to load CF model: {e}")
        else:
            logger.warning("⚠️ No trained model found at models/cf_model_latest.pkl. CF will be empty.")
        
        # Recommendation weights
        self.weights = config.get('recommendation', {}).get('weights', {
            'collaborative': 0.70,
            'trending': 0.20,
            'popular': 0.10
        })
        
        logger.info("RecommendationService initialized")
    
        try:
            logger.info("⏳ Loading Content-Based Model (Text Embedding)...")
            # Load model tiếng Việt (Nặng khoảng 500MB RAM)
            self.cb_engine = ContentBasedEngine(model_name='dangvantuan/vietnamese-embedding')
            
            # Load embeddings đã train trước đó từ file .pkl (Nếu có)
            # Giúp tìm kiếm nhanh hơn mà không cần query DB
            cb_path = "models/cb_embeddings_latest.pkl"
            if os.path.exists(cb_path):
                self.cb_engine.load_embeddings(cb_path)
                logger.info(f"✅ Loaded {len(self.cb_engine.product_embeddings)} product embeddings from file.")
            else:
                logger.warning("⚠️ No pre-computed embeddings found. Will use DB vector search (Slower but works).")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Content-Based Model: {e}")
            self.cb_engine = None
    def train_cf_model(self, days: int = 90):
        """
        Train collaborative filtering model
        
        Args:
            days: Number of days of data to use for training
        """
        logger.info(f"Training CF model with {days} days of data...")
        
        # Get interaction data with time decay
        interaction_data = self.db.query("""
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
        """, (days,))
        
        if interaction_data.empty:
            logger.warning("No interaction data found for training")
            return
        
        # Train model
        self.cf_engine.train(interaction_data)
        
        # Cache similar products for all products
        self._cache_similar_products()
        
        logger.info("CF model training completed")
    
    def _normalize_scores(self, data_dict):
            """Helper: Chuẩn hóa điểm số về khoảng 0-1"""
            if not data_dict:
                return {}
            
            max_score = max(data_dict.values())
            if max_score == 0:
                return data_dict
                
            return {pid: score / max_score for pid, score in data_dict.items()}

    def get_personalized_recommendations(self, user_id, n=20, context=None):
        # 0. Check Cache (Bật lại khi chạy thật)
        # cache_key = f"rec:personalized:{user_id}"
        # cached = self.redis.get(cache_key)
        # if cached: return json.loads(cached)

        # 1. NGUỒN: Collaborative Filtering (70%)
        # Lấy nhiều hơn N để còn lọc
        cf_recs = self.cf_engine.recommend(user_id, n=n*3) 
        cf_raw = {pid: score for pid, score in cf_recs}
        cf_norm = self._normalize_scores(cf_raw) # Chuẩn hóa về 0-1
        
        # 2. NGUỒN: Search History (Redis) - Boost điểm
        search_bonus = {}
        last_query = self.redis.get(f"user:{user_id}:last_search")
        if last_query:
            # Tìm 10 sp liên quan từ khóa, gán điểm max (1.0)
            search_items = self.cb_engine.search(last_query, n=10)
            for pid, _ in search_items:
                search_bonus[pid] = 1.0 # Bonus 1.0 điểm
        
        # 3. NGUỒN: Trending (20%)
        trending = self._get_trending_products(n=50)
        trending_raw = {pid: score for pid, score in trending}
        trending_norm = self._normalize_scores(trending_raw) # Chuẩn hóa về 0-1
        
        # 4. NGUỒN: Popular in Category (10%)
        popular = self._get_popular_in_user_categories(user_id, n=50)
        popular_raw = {pid: score for pid, score in popular}
        popular_norm = self._normalize_scores(popular_raw) # Chuẩn hóa về 0-1

        # 5. TỔNG HỢP ĐIỂM (Hybrid)
        all_products = set(cf_norm.keys()) | set(trending_norm.keys()) | set(popular_norm.keys()) | set(search_bonus.keys())
        combined = []
        
        # Trọng số
        W_CF = 0.6      # Giảm chút để nhường cho Search
        W_TREND = 0.2
        W_POP = 0.1
        W_SEARCH = 0.1  # Trọng số cho search
        
        for pid in all_products:
            # Công thức Hybrid chuẩn hóa
            total_score = (
                cf_norm.get(pid, 0) * W_CF +
                trending_norm.get(pid, 0) * W_TREND +
                popular_norm.get(pid, 0) * W_POP +
                search_bonus.get(pid, 0) * W_SEARCH
            )
            
            combined.append({
                'product_id': pid,
                'score': float(total_score),
                'reason': self._generate_reason(pid, user_id, cf_norm, trending_norm, search_bonus)
            })
            
        # Sort tạm để lấy Top đầu ứng viên
        combined.sort(key=lambda x: x['score'], reverse=True)
        candidates = combined[:n*3] # Lấy top 60 để lọc giá
        
        # 6. LỌC NGÂN SÁCH & LẤY INFO CHI TIẾT
        # Lấy profile user
        user_profile = self.db.fetchone("SELECT avg_order_value FROM user_profiles WHERE user_id = %s", (user_id,))
        avg_spend = float(user_profile[0] or 0)
        max_budget = avg_spend * 3.0 if avg_spend > 0 else float('inf')

        candidate_ids = [c['product_id'] for c in candidates]
        
        if not candidate_ids: 
            return []

        # Query DB để lấy giá, tên, ảnh và filter giá luôn
        placeholders = ','.join(['%s'] * len(candidate_ids))
        query = f"""
            SELECT product_id, current_price, avg_rating_updated
            FROM product_features
            WHERE product_id IN ({placeholders})
            AND current_price <= %s
        """
        
        rows = self.db.query(query, tuple(candidate_ids) + (max_budget,))
        
        # 7. FORMAT KẾT QUẢ CUỐI CÙNG
        final_results = []
        for item in candidates:
            # Tìm thông tin trong DB result
            prod_info = rows[rows['product_id'] == item['product_id']]
            if not prod_info.empty:
                price = float(prod_info.iloc[0]['current_price'])
                rating = float(prod_info.iloc[0]['avg_rating_updated'] or 0)
                
                item['price'] = price
                item['rating'] = rating
                final_results.append(item)
                
        # Cắt lấy đúng N sản phẩm
        result = final_results[:n]
        
        # Log & Cache
        self._log_impressions(user_id, result, 'personalized', context)
        # self.redis.setex(cache_key, 3600, json.dumps(result))
        
        return result

    def _generate_reason(self, pid, uid, cf_dict, trend_dict, search_dict):
        """Helper sinh lý do gợi ý"""
        if search_dict.get(pid, 0) > 0:
            return "Liên quan đến tìm kiếm của bạn"
        if cf_dict.get(pid, 0) > 0.5: # Ngưỡng cao
            return "Phù hợp sở thích của bạn"
        if trend_dict.get(pid, 0) > 0.8:
            return "Đang dẫn đầu xu hướng"
        return "Gợi ý cho bạn"
    def get_similar_products(self, product_id: str, n: int = 10) -> List[Dict]:
        """
        Get similar products (Super Hybrid Strategy)
        Priority: DB Cache -> CF Engine (RAM) -> Content-Based (Vector DB) -> Category
        """
        # ---------------------------------------------------------
        # 1. Ưu tiên 1: Lấy từ Cache Database (Kết quả của lần train trước)
        # ---------------------------------------------------------
        cached_similar = self.db.fetchone("""
            SELECT similar_product_ids 
            FROM product_features 
            WHERE product_id = %s
        """, (product_id,))
        
        if cached_similar and cached_similar[0]:
            # Nếu có cache, trả về ngay
            similar_ids = cached_similar[0][:n]
            results = self._fetch_product_details(similar_ids, reason="Similar (Cache/Behavior)")
            self._log_impressions('anonymous', results, 'similar', {'source': product_id})
            return results

        # ---------------------------------------------------------
        # 2. Ưu tiên 2: Thử hỏi CF Engine (Trong RAM)
        # ---------------------------------------------------------
        # Dành cho trường hợp mới train xong nhưng chưa kịp sync vào DB
        # try:
        #     cf_sim = self.cf_engine.similar_products(product_id, n)
        #     if cf_sim:
        #         logger.info(f"CF Hit for {product_id}")
        #         results = [{'product_id': pid, 'score': score, 'reason': 'Similar (Behavior)'} 
        #                    for pid, score in cf_sim]
        #         self._log_impressions('anonymous', results, 'similar', {'source': product_id})
        #         return results
        # except Exception:
        #     pass # CF fail thì đi tiếp, không báo lỗi

        # ---------------------------------------------------------
        # 3. Ưu tiên 3: Dùng Content-Based (Vector Search trong DB)
        # ---------------------------------------------------------
        # Cứu cánh cho sản phẩm mới (Cold Start)
        logger.info(f"Cache & CF miss for {product_id}. Switching to Vector Search...")
        
        try:
            # Lấy vector của sản phẩm hiện tại
            query_vec = self.db.fetchone("""
                SELECT text_embedding FROM product_features WHERE product_id = %s
            """, (product_id,))
            
            if query_vec and query_vec[0]:
                vector_str = query_vec[0]
                
                # Tìm kiếm bằng pgvector
                similar_rows = self.db.query("""
                    SELECT product_id, 1 - (text_embedding <=> %s) as score
                    FROM product_features
                    WHERE product_id != %s
                    AND text_embedding IS NOT NULL
                    ORDER BY text_embedding <=> %s ASC
                    LIMIT %s
                """, (vector_str, product_id, vector_str, n))
                
                results = []
                for _, row in similar_rows.iterrows():
                    results.append({
                        'product_id': row['product_id'],
                        'score': float(row['score']),
                        'reason': 'Similar (Content)'
                    })
                
                if results:
                    self._log_impressions('anonymous', results, 'similar', {'source': product_id})
                    return results
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # ---------------------------------------------------------
        # 4. Ưu tiên 4: Fallback về cùng Danh Mục (Category)
        # ---------------------------------------------------------
        # Lưới an toàn cuối cùng nếu sản phẩm chưa có cả vector
        logger.info("Vector search failed. Fallback to Category.")
        cat_products = self.db.query("""
            SELECT t1.product_id 
            FROM product_features t1
            JOIN product_features t2 ON t1.category_id = t2.category_id
            WHERE t2.product_id = %s AND t1.product_id != %s
            ORDER BY t1.view_count_30d DESC
            LIMIT %s
        """, (product_id, product_id, n))
        
        results = [{'product_id': r['product_id'], 'score': 0.1, 'reason': 'Same Category'} 
                   for _, r in cat_products.iterrows()]
        
        return results

    def _fetch_product_details(self, product_ids, reason=""):
        """Helper để format kết quả trả về đồng nhất"""
        # Ở đây bạn có thể query thêm giá/ảnh nếu cần, hiện tại trả về ID là đủ
        return [{'product_id': pid, 'score': 0.9 - (i*0.01), 'reason': reason} 
                for i, pid in enumerate(product_ids)]
    def get_cross_sell(self, product_ids: List[str], n: int = 5) -> List[Dict]:
        """
        Get cross-sell recommendations (frequently bought together)
        
        Args:
            product_ids: List of product IDs in cart
            n: Number of recommendations
            
        Returns:
            List of cross-sell recommendations
        """
        all_recs = []
        
        for pid in product_ids:
            # Get frequently bought together products
            fbt = self.db.query("""
                SELECT * FROM get_frequently_bought_together(%s, %s)
            """, (pid, n))
            
            for _, row in fbt.iterrows():
                all_recs.append({
                    'product_id': row['product_id'],
                    'score': float(row['frequency']),
                    'reason': 'Frequently bought together'
                })
        
        # Deduplicate and sort
        seen = set(product_ids)
        unique = []
        for rec in all_recs:
            if rec['product_id'] not in seen:
                unique.append(rec)
                seen.add(rec['product_id'])
        
        unique.sort(key=lambda x: x['score'], reverse=True)
        return unique[:n]
    
    def get_upsell(self, product_id: str, n: int = 5) -> List[Dict]:
        """
        Get upsell recommendations (higher-priced alternatives)
        
        Args:
            product_id: Current product ID
            n: Number of recommendations
            
        Returns:
            List of upsell recommendations
        """
        # Get current product info
        current = self.db.fetchone("""
            SELECT current_price, category_id 
            FROM product_features 
            WHERE product_id = %s
        """, (product_id,))
        
        if not current:
            return []
        
        current_price, category_id = current
        
        # Get higher-priced products in same category
        upsell = self.db.query("""
            SELECT product_id, current_price, conversion_rate
            FROM product_features
            WHERE category_id = %s
            AND current_price > %s * 1.2
            AND current_price < %s * 2.0
            ORDER BY conversion_rate DESC
            LIMIT %s
        """, (category_id, current_price, current_price, n))
        
        result = []
        for _, row in upsell.iterrows():
            result.append({
                'product_id': row['product_id'],
                'score': float(row['conversion_rate']),
                'price_difference': float(row['current_price'] - current_price),
                'reason': 'Premium alternative'
            })
        
        return result
    
    def track_click(self, user_id: str, product_id: str, rec_type: str):
        """
        Track when user clicks on a recommendation
        
        Args:
            user_id: User ID
            product_id: Product ID clicked
            rec_type: Recommendation type
        """
        try:
            self.db.execute("""
                UPDATE recommendation_logs
                SET clicked_at = NOW()
                WHERE user_id = %s 
                AND product_id = %s 
                AND rec_type = %s
                AND shown_at >= NOW() - INTERVAL '1 hour'
                AND clicked_at IS NULL
            """, (user_id, product_id, rec_type))
            
            logger.debug(f"Tracked click: user={user_id}, product={product_id}")
        except Exception as e:
            logger.error(f"Failed to track click: {e}")
    
    def track_purchase(self, user_id: str, product_id: str, amount: float):
        """
        Track when user purchases a recommended product
        
        Args:
            user_id: User ID
            product_id: Product ID purchased
            amount: Purchase amount
        """
        try:
            self.db.execute("""
                UPDATE recommendation_logs
                SET purchased_at = NOW(), purchase_amount = %s
                WHERE user_id = %s 
                AND product_id = %s
                AND shown_at >= NOW() - INTERVAL '7 days'
                AND purchased_at IS NULL
            """, (amount, user_id, product_id))
            
            logger.debug(f"Tracked purchase: user={user_id}, product={product_id}, amount={amount}")
        except Exception as e:
            logger.error(f"Failed to track purchase: {e}")
    
    def get_metrics(self, days: int = 7) -> List[Dict]:
        """
        Get recommendation performance metrics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of metrics by recommendation type
        """
        df = self.db.query("""
            SELECT * FROM daily_recommendation_stats 
            WHERE date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC, rec_type
        """, (days,))
        
        return df.to_dict('records')
    
    # Private helper methods
    
    def _get_trending_products(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get trending products"""
        df = self.db.query("""
            SELECT product_id, trending_score
            FROM product_features
            WHERE trending_score > 0
            ORDER BY trending_score DESC
            LIMIT %s
        """, (n,))
        
        return list(df.itertuples(index=False, name=None))
    
    def _get_popular_in_user_categories(self, user_id: str, n: int = 20) -> List[Tuple[str, float]]:
        """Get popular products from user's favorite categories"""
        user_profile = self.db.fetchone("""
            SELECT favorite_categories FROM user_profiles WHERE user_id = %s
        """, (user_id,))
        
        if not user_profile or not user_profile[0]:
            return []
        
        categories = user_profile[0]
        
        df = self.db.query("""
            SELECT product_id, conversion_rate
            FROM product_features
            WHERE category_id = ANY(%s)
            ORDER BY conversion_rate DESC
            LIMIT %s
        """, (categories, n))
        
        return list(df.itertuples(index=False, name=None))
    
    def _generate_reason(self, product_id: str, user_id: str, 
                        cf_dict: Dict, trending_dict: Dict) -> str:
        """Generate explanation for recommendation"""
        
        # Check if highly rated
        product_info = self.db.fetchone("""
            SELECT avg_rating_updated, review_count FROM product_features 
            WHERE product_id = %s
        """, (product_id,))
        
        if product_info:
            avg_rating = product_info[0] or 0
            review_count = product_info[1] or 0
            
            if avg_rating >= 4.5 and review_count >= 50:
                return f"⭐ Highly rated ({avg_rating:.1f}/5 from {review_count} reviews)"
        
        if product_id in trending_dict and trending_dict[product_id] > 0.1:
            return "🔥 Trending now"
        elif product_id in cf_dict:
            return "✨ Based on your preferences"
        else:
            return "⭐ Popular choice"
    
    def _cache_similar_products(self):
        """Pre-compute and cache similar products"""
        logger.info("Caching similar products...")
        
        # Get all products
        products = self.db.query("SELECT product_id FROM product_features")
        
        count = 0
        for _, row in products.iterrows():
            pid = row['product_id']
            similar = self.cf_engine.similar_products(pid, n=20)
            
            if similar:
                similar_ids = [p[0] for p in similar]
                self.db.execute("""
                    UPDATE product_features
                    SET similar_product_ids = %s
                    WHERE product_id = %s
                """, (similar_ids, pid))
                count += 1
        
        logger.info(f"Cached similar products for {count} products")
    
    def _log_impressions(self, user_id: str, recommendations: List[Dict], 
                        rec_type: str, context: Optional[Dict] = None):
        """Log recommendation impressions"""
        try:
            data = [
                (user_id, rec['product_id'], rec_type, i+1, rec.get('score', 0),
                 json.dumps(context) if context else None)
                for i, rec in enumerate(recommendations)
            ]
            
            self.db.execute_many("""
                INSERT INTO recommendation_logs 
                (user_id, product_id, rec_type, rec_position, rec_score, page_context)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, data)
            
        except Exception as e:
            logger.error(f"Failed to log impressions: {e}")