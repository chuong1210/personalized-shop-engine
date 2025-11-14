"""
service.py - Main Recommendation Service
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import redis

from database import Database
from cf_engine import CollaborativeFilteringEngine

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
        
        # Recommendation weights
        self.weights = config.get('recommendation', {}).get('weights', {
            'collaborative': 0.70,
            'trending': 0.20,
            'popular': 0.10
        })
        
        logger.info("RecommendationService initialized")
    
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
    
    def get_personalized_recommendations(
        self, 
        user_id: str, 
        n: int = 20,
        context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get personalized recommendations for a user
        
        Combines:
        - Collaborative filtering (70%)
        - Trending products (20%)
        - Popular in user's favorite categories (10%)
        
        Args:
            user_id: User ID
            n: Number of recommendations
            context: Additional context (page, device, etc.)
            
        Returns:
            List of recommendation dictionaries with product_id, score, reason
        """
        # Check cache
        cache_key = f"rec:personalized:{user_id}"
        cached = self.redis.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for user {user_id}")
            return json.loads(cached)
        
        logger.info(f"Generating personalized recommendations for user {user_id}")
        
        # 1. Collaborative Filtering (70%)
        cf_recs = self.cf_engine.recommend(user_id, n=n*2)
        cf_dict = {pid: score * self.weights['collaborative'] for pid, score in cf_recs}
        
        # 2. Trending products (20%)
        trending = self._get_trending_products(n=20)
        trending_dict = {pid: score * self.weights['trending'] for pid, score in trending}
        
        # 3. Popular in favorite categories (10%)
        popular = self._get_popular_in_user_categories(user_id, n=20)
        popular_dict = {pid: score * self.weights['popular'] for pid, score in popular}
        
        # Combine scores
        all_products = set(cf_dict.keys()) | set(trending_dict.keys()) | set(popular_dict.keys())
        combined = []
        
        for pid in all_products:
            total_score = (
                cf_dict.get(pid, 0) + 
                trending_dict.get(pid, 0) + 
                popular_dict.get(pid, 0)
            )
            
            combined.append({
                'product_id': pid,
                'score': float(total_score),
                'reason': self._generate_reason(pid, user_id, cf_dict, trending_dict)
            })
        
        # Sort by score and get top N
        combined.sort(key=lambda x: x['score'], reverse=True)
        result = combined[:n]
        
        # Cache for 1 hour
        cache_ttl = self.config.get('recommendation', {}).get('cache_ttl', 3600)
        self.redis.setex(cache_key, cache_ttl, json.dumps(result))
        
        # Log impressions
        self._log_impressions(user_id, result, 'personalized', context)
        
        logger.info(f"Generated {len(result)} recommendations for user {user_id}")
        return result
    
    def get_similar_products(self, product_id: str, n: int = 10) -> List[Dict]:
        """
        Get similar products
        
        Args:
            product_id: Product ID
            n: Number of similar products
            
        Returns:
            List of similar product dictionaries
        """
        # Try from cache first
        cached_similar = self.db.fetchone("""
            SELECT similar_product_ids 
            FROM product_features 
            WHERE product_id = %s
        """, (product_id,))
        
        if cached_similar and cached_similar[0]:
            similar_ids = cached_similar[0][:n]
            result = [{'product_id': pid, 'score': 1.0 - i*0.05} for i, pid in enumerate(similar_ids)]
        else:
            # Compute using CF model
            similar = self.cf_engine.similar_products(product_id, n)
            result = [{'product_id': pid, 'score': score} for pid, score in similar]
        
        self._log_impressions('anonymous', result, 'similar', {'source_product': product_id})
        
        return result
    
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