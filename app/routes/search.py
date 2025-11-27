# app/routes/search.py
from flask import Blueprint, request, jsonify
from PIL import Image
from app.extensions import clip_model, service, logger

search_bp = Blueprint('search', __name__)

@search_bp.route('/api/search/image', methods=['POST'])
def search_by_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    try:
        image = Image.open(file)
        
        if not clip_model:
            return jsonify({'error': 'Model not loaded'}), 503
            
        query_vector = clip_model.encode(image, convert_to_numpy=True, normalize_embeddings=True).tolist()
        
        # SQL Query mới:
        # 1. Tìm trong bảng image_embeddings
        # 2. Join với product_features để lấy giá
        # 3. DISTINCT ON hoặc GROUP BY để mỗi sản phẩm chỉ hiện 1 lần (lần giống nhất)
        results = service.db.query("""
            WITH best_matches AS (
                SELECT 
                    pie.product_id,
                    pie.image_url as matched_image,
                    1 - (pie.embedding <=> %s::vector) as similarity
                FROM product_image_embeddings pie
                ORDER BY pie.embedding <=> %s::vector ASC
                LIMIT 20
            )
            -- DISTINCT ON để lỡ 1 sản phẩm có 2 ảnh giống, chỉ lấy ảnh giống nhất
            SELECT DISTINCT ON (bm.product_id)
                bm.product_id,
                bm.similarity,
                bm.matched_image,
                pf.current_price
            FROM best_matches bm
            JOIN product_features pf ON bm.product_id = pf.product_id
            ORDER BY bm.product_id, bm.similarity DESC
            LIMIT 10
        """, (query_vector, query_vector))

        # Sort lại lần cuối theo similarity giảm dần (vì DISTINCT ON làm mất thứ tự)
        results_sorted = results.sort_values(by='similarity', ascending=False)
        
        
        products = []
        for _, row in results_sorted.iterrows():
            products.append({
                'product_id': row['product_id'],
                'similarity': float(row['similarity']),
                'price': float(row['current_price']) if row['current_price'] else 0,
                'matched_image': row['matched_image'] # Frontend có thể hiển thị: "Giống với ảnh này của sp"
            })
            
        return jsonify({'success': True, 'results': products})

    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return jsonify({'error': str(e)}), 500