import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

class SimplifiedEntity(BaseModel):
    """Entity đơn giản hóa"""
    type: str
    context: str = ""

class SimplifiedAnalysis(BaseModel):
    """Kết quả phân tích đơn giản hóa"""
    topic: str
    sentiment: str
    entities: List[SimplifiedEntity] = Field(default_factory=list)
    purchase_intent: str

class BatchAnalyzer:
    """Phân tích batch nhiều messages cùng lúc"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=api_key
        )
    
    def create_batch_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Tạo prompt cho batch analysis
        Format: Phân tích nhiều câu hỏi cùng lúc
        """
        
        # Tạo danh sách câu hỏi với ID
        questions_list = []
        for idx, msg in enumerate(messages, 1):
            questions_list.append(f"{idx}. (ID: {msg['id']}) {msg['message']}")
        
        questions_text = "\n".join(questions_list)
        
        prompt = f"""Bạn là chuyên gia phân tích chat e-commerce. Phân tích TẤT CẢ các câu hỏi sau:

{questions_text}

---

📋 HƯỚNG DẪN PHÂN TÍCH:

1️⃣ **Topic** (Chủ đề) - Chọn 1 trong các loại:
- TRA_CUU_DON_HANG: Hỏi về đơn hàng
- HOI_PHI_SHIP: Hỏi về phí ship
- HOI_CHINH_SACH: Hỏi về chính sách
- TIM_KIEM_SAN_PHAM: Tìm sản phẩm
- TU_VAN_SAN_PHAM: Tư vấn sản phẩm
- HOI_KHUYEN_MAI: Hỏi về khuyến mãi
- KHIEU_NAI_SAN_PHAM: Khiếu nại sản phẩm
- KHIEU_NAI_GIAO_HANG: Khiếu nại giao hàng
- LOI_HE_THONG: Lỗi hệ thống
- UNKNOWN: Không xác định

2️⃣ **Sentiment** (Cảm xúc):
- POSITIVE: Tích cực
- NEUTRAL: Trung lập
- NEGATIVE: Tiêu cực
- FRUSTRATED: Bực bội
- CONFUSED: Bối rối
- URGENT: Khẩn cấp

3️⃣ **Entities** (Thực thể) - Trích xuất nếu có:
- product_id: Mã sản phẩm
- sku_id: Mã SKU
- order_code: Mã đơn hàng
- shop_id: Mã shop
- brand_name: Thương hiệu
- category_name: Danh mục
- voucher_code: Mã giảm giá

4️⃣ **Purchase Intent** (Ý định mua):
- HIGH: Rất cao
- MEDIUM: Trung bình
- LOW: Thấp
- NONE: Không có

---

️ YÊU CẦU QUAN TRỌNG:
- Trả về JSON array với đúng thứ tự câu hỏi
- Mỗi entity chỉ có "type" và "context" (không có value)
- Không cần "reasoning", "confidence", "success", "data"

📤 FORMAT ĐÚNG:
```json
[
  {{
    "id": "event-id-1",
    "topic": "TIM_KIEM_SAN_PHAM",
    "sentiment": "NEUTRAL",
    "entities": [
      {{
        "type": "category_name",
        "context": "giày thể thao chạy bộ"
      }},
      {{
        "type": "brand_name", 
        "context": "Nike hoặc Adidas"
      }}
    ],
    "purchase_intent": "MEDIUM"
  }},
  {{
    "id": "event-id-2",
    "topic": "TRA_CUU_DON_HANG",
    "sentiment": "FRUSTRATED",
    "entities": [
      {{
        "type": "order_code",
        "context": "đơn hàng YAN20251013ABC"
      }}
    ],
    "purchase_intent": "NONE"
  }}
]
```

Bắt đầu phân tích:
"""
        return prompt
    
    def analyze_batch(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phân tích batch messages
        
        Args:
            messages: List of {'id': event_id, 'message': text}
        
        Returns:
            List of analysis results
        """
        if not messages:
            return []
        
        print(f"🔍 Đang phân tích batch {len(messages)} messages...")
        
        # Tạo prompt
        prompt = self.create_batch_prompt(messages)
        
        # Gọi LLM
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # Parse JSON
            # Loại bỏ markdown code blocks
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            results = json.loads(result_text.strip())
            
            print(f" Phân tích thành công {len(results)} messages")
            return results
            
        except json.JSONDecodeError as e:
            print(f" Lỗi parse JSON: {e}")
            print(f"Raw response: {response.content[:500]}...")
            return []
        except Exception as e:
            print(f" Lỗi khi gọi LLM: {e}")
            return []