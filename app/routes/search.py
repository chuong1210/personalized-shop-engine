# app/routes/search.py
from flask import Blueprint, request, jsonify
from PIL import Image
from app.extensions import clip_model, service, logger
from ultralytics import YOLO
from PIL import Image
search_bp = Blueprint('search', __name__)


# Load model YOLO (nó sẽ tự tải file yolov8n.pt - bản nano rất nhẹ)
# Chỉ load 1 lần ở đầu file hoặc trong extensions.py
object_detector = YOLO("yolo12n.pt") 

def crop_object(image):
    """
    Dùng YOLO để tìm vật thể chính trong ảnh và cắt ra
    """
    results = object_detector(image)
    
    # Lấy box có độ tin cậy cao nhất
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            # Lấy tọa độ box đầu tiên (thường là vật to nhất)
            box = boxes[0].xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            
            # Cắt ảnh
            cropped_image = image.crop((box[0], box[1], box[2], box[3]))
            return cropped_image
            
    return image # Nếu không tìm thấy vật gì, trả về ảnh gốc
@search_bp.route('/api/search/image', methods=['POST'])
def search_by_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    try:
        original_image = Image.open(file)
        processed_image = crop_object(original_image)

        
        if not clip_model:
            return jsonify({'error': 'Model not loaded'}), 503
            
        query_vector = clip_model.encode(processed_image, convert_to_numpy=True, normalize_embeddings=True).tolist()
        
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
# Trong api.py
@search_bp.route('/api/search/smart', methods=['GET'])
def smart_search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Missing query'}), 400
        
    try:
        if not service.cb_engine:
             return jsonify({'error': 'AI Engine not ready'}), 503
             
        query_vector = service.cb_engine.compute_embedding(query).tolist()
        
        # --- CẬP NHẬT LOGIC SQL ---
        # 1. Similarity: 1 - distance (Khoảng 0.0 đến 1.0)
        # 2. Rating Normalized: avg_rating_updated / 5.0 (Khoảng 0.0 đến 1.0)
        # 3. Hybrid Score: Kết hợp 2 chỉ số trên
        
        results = service.db.query("""
            SELECT 
                product_id, 
                current_price,
                avg_rating_updated,
                1 - (text_embedding <=> %s::vector) as similarity,
                
                -- Công thức tính điểm ưu tiên Rating
                (
                    (1 - (text_embedding <=> %s::vector)) * 0.7 +  -- 70%% trọng số cho độ liên quan
                    (COALESCE(avg_rating_updated, 0) / 5.0) * 0.3  -- 30%% trọng số cho đánh giá cao
                ) as hybrid_score
                
            FROM product_features
            WHERE text_embedding IS NOT NULL
            
            -- Lọc sơ bộ: Chỉ lấy những sp có độ giống > 0.25 để tránh noise trước khi sort
            AND (1 - (text_embedding <=> %s::vector)) > 0.25
            
            -- Sắp xếp theo điểm tổng hợp (cao xuống thấp)
            ORDER BY hybrid_score DESC
            LIMIT 20
        """, (query_vector, query_vector, query_vector))
        
        products = []
        for _, row in results.iterrows():
            products.append({
                'product_id': row['product_id'],
                'similarity': round(float(row['similarity']), 4),
                'price': float(row['current_price']),
                'rating': float(row['avg_rating_updated']) if row['avg_rating_updated'] else 0,
                'score': round(float(row['hybrid_score']), 4) # Debug xem điểm
            })
                
        return jsonify({'success': True, 'results': products})
        
    except Exception as e:
        logger.error(f"Smart search error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500