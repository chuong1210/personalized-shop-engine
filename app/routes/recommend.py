# app/routes/recommend.py
from flask import Blueprint, request, jsonify
from app.extensions import service, logger

recommend_bp = Blueprint('recommend', __name__)

@recommend_bp.route('/api/recommendations/personalized', methods=['POST'])
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
        # print(recommendations)
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

@recommend_bp.route('/api/recommendations/similar', methods=['POST'])
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


@recommend_bp.route('/api/recommendations/cross-sell', methods=['POST'])
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


@recommend_bp.route('/api/recommendations/top-rated', methods=['POST'])
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
