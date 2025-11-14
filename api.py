"""
api.py - Flask REST API for recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import yaml
import redis

from database import Database
from recommend_service import RecommendationService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global service instance
service = None


def init_service():
    """Initialize recommendation service"""
    global service
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = Database(config['database'])
    
    # Initialize Redis
    redis_client = redis.Redis(**config['redis'], decode_responses=True)
    
    # Initialize service
    service = RecommendationService(db, redis_client, config)
    
    logger.info("Service initialized successfully")


# ============================================================================
# RECOMMENDATION ENDPOINTS
# ============================================================================

@app.route('/api/recommendations/personalized', methods=['POST'])
def get_personalized():
    """
    Get personalized recommendations for a user
    
    Request body:
    {
        "user_id": "user123",
        "n": 20,
        "context": {
            "page": "home",
            "device": "mobile"
        }
    }
    
    Response:
    {
        "success": true,
        "recommendations": [
            {
                "product_id": "prod001",
                "score": 0.85,
                "reason": "Based on your preferences"
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        n = data.get('n', 20)
        context = data.get('context', {})
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
        
        recommendations = service.get_personalized_recommendations(
            user_id, n, context
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Personalized recommendations failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recommendations/similar', methods=['POST'])
def get_similar():
    """
    Get similar products
    
    Request body:
    {
        "product_id": "prod001",
        "n": 10
    }
    
    Response:
    {
        "success": true,
        "similar_products": [
            {
                "product_id": "prod002",
                "score": 0.92
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        product_id = data.get('product_id')
        n = data.get('n', 10)
        
        if not product_id:
            return jsonify({
                'success': False,
                'error': 'product_id is required'
            }), 400
        
        similar = service.get_similar_products(product_id, n)
        
        return jsonify({
            'success': True,
            'similar_products': similar,
            'count': len(similar)
        })
        
    except Exception as e:
        logger.error(f"Similar products failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recommendations/cross-sell', methods=['POST'])
def get_cross_sell():
    """
    Get cross-sell recommendations (frequently bought together)
    
    Request body:
    {
        "product_ids": ["prod001", "prod002"],
        "n": 5
    }
    
    Response:
    {
        "success": true,
        "recommendations": [
            {
                "product_id": "prod003",
                "score": 15.0,
                "reason": "Frequently bought together"
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        product_ids = data.get('product_ids', [])
        n = data.get('n', 5)
        
        if not product_ids:
            return jsonify({
                'success': False,
                'error': 'product_ids is required'
            }), 400
        
        recommendations = service.get_cross_sell(product_ids, n)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Cross-sell recommendations failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recommendations/top-rated', methods=['POST'])
def get_top_rated():
    """
    Get top-rated products (based on reviews)
    
    Request body:
    {
        "category_id": "cat001",  // optional
        "min_reviews": 10,  // optional
        "n": 20
    }
    
    Response:
    {
        "success": true,
        "products": [
            {
                "product_id": "prod001",
                "avg_rating": 4.8,
                "review_count": 150,
                "positive_ratio": 0.92
            }
        ]
    }
    """
    try:
        data = request.json
        category_id = data.get('category_id')
        min_reviews = data.get('min_reviews', 10)
        n = data.get('n', 20)
        
        # Query top-rated products
        query = """
            SELECT 
                pf.product_id,
                pf.avg_rating_updated as avg_rating,
                pf.review_count,
                pf.current_price,
                pf.category_id,
                s.avg_sentiment,
                s.positive_ratio
            FROM product_features pf
            CROSS JOIN LATERAL get_product_sentiment_stats(pf.product_id) s
            WHERE pf.review_count >= %s
        """
        
        params = [min_reviews]
        
        if category_id:
            query += " AND pf.category_id = %s"
            params.append(category_id)
        
        query += """
            ORDER BY 
                pf.avg_rating_updated DESC,
                pf.review_count DESC,
                s.positive_ratio DESC
            LIMIT %s
        """
        params.append(n)
        
        df = service.db.query(query, tuple(params))
        
        products = []
        for _, row in df.iterrows():
            products.append({
                'product_id': row['product_id'],
                'avg_rating': float(row['avg_rating']) if row['avg_rating'] else 0,
                'review_count': int(row['review_count']) if row['review_count'] else 0,
                'sentiment_score': float(row['avg_sentiment']) if row['avg_sentiment'] else 0,
                'positive_ratio': float(row['positive_ratio']) if row['positive_ratio'] else 0,
                'price': float(row['current_price']) if row['current_price'] else 0
            })
        
        return jsonify({
            'success': True,
            'products': products,
            'count': len(products)
        })
        
    except Exception as e:
        logger.error(f"Top-rated products failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recommendations/by-reviews', methods=['POST'])
def get_by_reviews():
    """
    Get recommendations for users who like highly-rated products
    
    Request body:
    {
        "user_id": "user123",
        "n": 20
    }
    """
    try:
        data = request.json
        user_id = data['user_id']
        n = data.get('n', 20)
        
        # Get user's favorite highly-rated products
        user_liked = service.db.query("""
            SELECT DISTINCT ui.product_id
            FROM user_interactions ui
            JOIN product_features pf ON ui.product_id = pf.product_id
            WHERE ui.user_id = %s
            AND ui.action_type IN ('purchase', 'wishlist', 'cart_add')
            AND pf.avg_rating_updated >= 4.0
            AND pf.review_count >= 10
            ORDER BY ui.created_at DESC
            LIMIT 5
        """, (user_id,))
        
        if user_liked.empty:
            return jsonify({
                'success': True,
                'recommendations': [],
                'message': 'No purchase history found'
            })
        
        liked_products = user_liked['product_id'].tolist()
        
        # Find similar highly-rated products
        recommendations = []
        for product_id in liked_products:
            similar = service.get_similar_products(product_id, n=10)
            
            # Filter by high rating
            for rec in similar:
                product_info = service.db.fetchone("""
                    SELECT avg_rating_updated, review_count
                    FROM product_features
                    WHERE product_id = %s
                    AND avg_rating_updated >= 4.0
                    AND review_count >= 10
                """, (rec['product_id'],))
                
                if product_info:
                    recommendations.append({
                        'product_id': rec['product_id'],
                        'score': rec['score'],
                        'avg_rating': float(product_info[0]),
                        'review_count': int(product_info[1]),
                        'reason': 'Similar to products you loved'
                    })
        
        # Deduplicate and sort
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['product_id'] not in seen:
                unique_recs.append(rec)
                seen.add(rec['product_id'])
        
        unique_recs.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'success': True,
            'recommendations': unique_recs[:n]
        })
        
    except Exception as e:
        logger.error(f"Review-based recommendations failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
def get_upsell():
    """
    Get upsell recommendations (premium alternatives)
    
    Request body:
    {
        "product_id": "prod001",
        "n": 5
    }
    
    Response:
    {
        "success": true,
        "recommendations": [
            {
                "product_id": "prod005",
                "score": 0.35,
                "price_difference": 200000,
                "reason": "Premium alternative"
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        product_id = data.get('product_id')
        n = data.get('n', 5)
        
        if not product_id:
            return jsonify({
                'success': False,
                'error': 'product_id is required'
            }), 400
        
        recommendations = service.get_upsell(product_id, n)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Upsell recommendations failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# TRACKING ENDPOINTS
# ============================================================================

@app.route('/api/track/click', methods=['POST'])
def track_click():
    """
    Track click on recommendation
    
    Request body:
    {
        "user_id": "user123",
        "product_id": "prod001",
        "rec_type": "personalized"
    }
    """
    try:
        data = request.json
        service.track_click(
            data['user_id'],
            data['product_id'],
            data['rec_type']
        )
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Track click failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/track/purchase', methods=['POST'])
def track_purchase():
    """
    Track purchase from recommendation
    
    Request body:
    {
        "user_id": "user123",
        "product_id": "prod001",
        "amount": 500000
    }
    """
    try:
        data = request.json
        service.track_purchase(
            data['user_id'],
            data['product_id'],
            data['amount']
        )
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Track purchase failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get recommendation performance metrics
    
    Query params:
    - days: Number of days (default: 7)
    
    Response:
    {
        "success": true,
        "metrics": [
            {
                "date": "2025-11-07",
                "rec_type": "personalized",
                "impressions": 1500,
                "clicks": 225,
                "conversions": 45,
                "ctr": 0.15,
                "conversion_rate": 0.03,
                "revenue": 22500000
            },
            ...
        ]
    }
    """
    try:
        days = request.args.get('days', 7, type=int)
        metrics = service.get_metrics(days)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get model statistics
    
    Response:
    {
        "success": true,
        "stats": {
            "num_users": 5000,
            "num_products": 10000,
            "num_interactions": 150000,
            "matrix_density": 0.3
        }
    }
    """
    try:
        stats = service.cf_engine.get_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    
    Response:
    {
        "status": "healthy",
        "components": {
            "database": "healthy",
            "redis": "healthy",
            "cf_model": "healthy"
        }
    }
    """
    status = {
        'status': 'healthy',
        'components': {}
    }
    
    # Check database
    try:
        service.db.fetchone("SELECT 1")
        status['components']['database'] = 'healthy'
    except:
        status['components']['database'] = 'unhealthy'
        status['status'] = 'degraded'
    
    # Check Redis
    try:
        service.redis.ping()
        status['components']['redis'] = 'healthy'
    except:
        status['components']['redis'] = 'unhealthy'
        status['status'] = 'degraded'
    
    # Check CF model
    stats = service.cf_engine.get_stats()
    if stats['num_users'] > 0:
        status['components']['cf_model'] = 'healthy'
    else:
        status['components']['cf_model'] = 'not_trained'
        status['status'] = 'degraded'
    
    return jsonify(status)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize service
    init_service()
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    api_config = config.get('api', {})
    
    # Run app
    app.run(
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 5000),
        debug=api_config.get('debug', False)
    )