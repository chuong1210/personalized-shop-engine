TOPIC_CLASSIFIER_PROMPT = """Bạn là chuyên gia phân loại chủ đề chat e-commerce.

Nhiệm vụ: Phân loại tin nhắn sau vào ĐÚNG 1 trong các chủ đề:

1. TRA_CUU_DON_HANG: Hỏi về trạng thái, vị trí đơn hàng
2. HOI_PHI_SHIP: Hỏi về chi phí vận chuyển
3. HOI_CHINH_SACH: Hỏi về đổi trả, bảo hành, chính sách
4. TIM_KIEM_SAN_PHAM: Tìm kiếm sản phẩm cụ thể
5. TU_VAN_SAN_PHAM: Hỏi tư vấn, so sánh sản phẩm
6. HOI_KHUYEN_MAI: Hỏi về voucher, giảm giá, flash sale
7. KHIEU_NAI_SAN_PHAM: Khiếu nại về chất lượng sản phẩm
8. KHIEU_NAI_GIAO_HANG: Khiếu nại về giao hàng chậm, shipper
9. LOI_HE_THONG: Báo lỗi thanh toán, lỗi website/app
10. UNKNOWN: Không xác định được

Tin nhắn: "{message}"

Trả về JSON:
{{
    "topic": "TEN_CHU_DE",
    "confidence": 0.95,
    "reasoning": "Giải thích ngắn gọn"
}}
"""

SENTIMENT_ANALYZER_PROMPT = """Bạn là chuyên gia phân tích cảm xúc khách hàng.

Nhiệm vụ: Phân tích cảm xúc của tin nhắn vào 1 trong 6 loại:

1. POSITIVE: Tích cực, hài lòng
2. NEUTRAL: Trung lập, bình thường
3. NEGATIVE: Tiêu cực, không hài lòng
4. FRUSTRATED: Bực bội, khó chịu (ví dụ: "sao mãi chưa giao?")
5. CONFUSED: Bối rối, không hiểu (ví dụ: "tôi không hiểu...")
6. URGENT: Khẩn cấp (ví dụ: "giúp tôi NGAY", "gấp lắm")

Tin nhắn: "{message}"
Chủ đề: {topic}

Trả về JSON:
{{
    "sentiment": "TEN_CAM_XUC",
    "confidence": 0.90,
    "reasoning": "Giải thích ngắn gọn"
}}
"""

ENTITY_EXTRACTOR_PROMPT = """Bạn là chuyên gia trích xuất thông tin từ chat e-commerce.

Nhiệm vụ: Trích xuất TẤT CẢ các thực thể (entities) từ tin nhắn:

Các loại entity:
- product_id: Mã sản phẩm (ví dụ: "p001", "SP123")
- sku_id: Mã SKU (ví dụ: "sku001", "ov001")
- order_code: Mã đơn hàng (ví dụ: "YAN20251013ABC", "DH123456")
- shop_id: Mã shop (ví dụ: "shop001")
- brand_name: Tên thương hiệu (ví dụ: "Nike", "Adidas")
- category_name: Danh mục sản phẩm (ví dụ: "Áo thun nam")
- voucher_code: Mã giảm giá (ví dụ: "FREESHIP30K")

Tin nhắn: "{message}"

Trả về JSON:
{{
    "entities": [
        {{
            "type": "product_id",
            "value": "p001",
            "context": "Áo thun Nike Basic"
        }}
    ],
    "confidence": 0.85
}}

Nếu không tìm thấy entity nào, trả về: {{"entities": [], "confidence": 1.0}}
"""

PURCHASE_INTENT_PROMPT = """Bạn là chuyên gia phân tích ý định mua hàng.

Nhiệm vụ: Đánh giá mức độ ý định mua hàng của khách:

- HIGH: Rất cao (hỏi giá, còn hàng không, muốn mua ngay)
- MEDIUM: Trung bình (đang tìm hiểu, so sánh)
- LOW: Thấp (chỉ hỏi thông tin chung)
- NONE: Không có (khiếu nại, hỏi đơn hàng cũ)

Thông tin:
- Tin nhắn: "{message}"
- Chủ đề: {topic}
- Cảm xúc: {sentiment}
- Entities: {entities}

Trả về JSON:
{{
    "purchase_intent": "HIGH",
    "confidence": 0.88,
    "reasoning": "Giải thích ngắn gọn"
}}
"""