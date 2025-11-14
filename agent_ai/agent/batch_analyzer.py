import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class BatchChatAnalyzer:
    """Phân tích batch nhiều câu hỏi cùng lúc để tiết kiệm token"""
    
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=api_key
        )
    
    def analyze_batch(self, messages: List[str]) -> List[Dict[str, Any]]:
        """
        Phân tích batch nhiều tin nhắn cùng lúc
        
        Args:
            messages: List các tin nhắn cần phân tích
        
        Returns:
            List[Dict]: Danh sách kết quả phân tích
        """
        if not messages:
            return []
        
        # Tạo prompt batch
        prompt = self._create_batch_prompt(messages)
        
        # Gọi LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse response
        results = self._parse_batch_response(response.content, len(messages))
        
        return results
    
    def _create_batch_prompt(self, messages: List[str]) -> str:
        """Tạo prompt phân tích batch"""
        
        # Format messages thành numbered list
        messages_text = "\n".join([
            f"{i+1}. {msg}"
            for i, msg in enumerate(messages)
        ])
        
        prompt = f"""Bạn là chuyên gia phân tích chat e-commerce.

Nhiệm vụ: Phân tích TỪNG tin nhắn dưới đây và trả về JSON array.

DANH SÁCH TIN NHẮN:
{messages_text}

CHỈ TIÊU PHÂN TÍCH:

1. TOPIC (Chủ đề):
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

2. SENTIMENT (Cảm xúc):
   - POSITIVE: Tích cực
   - NEUTRAL: Trung lập
   - NEGATIVE: Tiêu cực
   - FRUSTRATED: Bực bội
   - CONFUSED: Bối rối
   - URGENT: Khẩn cấp

3. ENTITIES (Thực thể) - CHỈ trích xuất type và context:
   - product_id, sku_id, order_code, shop_id, brand_name, category_name, voucher_code

4. PURCHASE_INTENT (Ý định mua):
   - HIGH: Cao
   - MEDIUM: Trung bình
   - LOW: Thấp
   - NONE: Không có

YÊU CẦU OUTPUT:
- Trả về JSON array với {len(messages)} phần tử
- Mỗi phần tử tương ứng với 1 tin nhắn theo đúng thứ tự
- Format từng phần tử:
{{
    "topic": "TEN_CHU_DE",
    "sentiment": "CAM_XUC",
    "entities": [
        {{"type": "product_id", "context": "ngữ cảnh"}}
    ],
    "purchase_intent": "MUC_DO"
}}

CHỈ TRẢ VỀ JSON ARRAY, KHÔNG KÈM TEXT KHÁC:
"""
        return prompt
    
    def _parse_batch_response(
        self,
        response: str,
        expected_count: int
    ) -> List[Dict[str, Any]]:
        """Parse JSON array response từ LLM"""
        try:
            # Loại bỏ markdown code blocks
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Parse JSON
            results = json.loads(response.strip())
            
            # Validate
            if not isinstance(results, list):
                print("️ Response không phải là array, wrap lại")
                results = [results]
            
            # Đảm bảo đủ số lượng kết quả
            while len(results) < expected_count:
                results.append({
                    "topic": "UNKNOWN",
                    "sentiment": "NEUTRAL",
                    "entities": [],
                    "purchase_intent": "NONE"
                })
            
            # Truncate nếu thừa
            results = results[:expected_count]
            
            # Validate từng kết quả
            validated_results = []
            for result in results:
                validated_results.append({
                    "topic": result.get("topic", "UNKNOWN"),
                    "sentiment": result.get("sentiment", "NEUTRAL"),
                    "entities": self._validate_entities(result.get("entities", [])),
                    "purchase_intent": result.get("purchase_intent", "NONE")
                })
            
            return validated_results
            
        except json.JSONDecodeError as e:
            print(f" Lỗi parse JSON: {e}")
            print(f"Raw response: {response}")
            
            # Fallback: trả về default values
            return [{
                "topic": "UNKNOWN",
                "sentiment": "NEUTRAL",
                "entities": [],
                "purchase_intent": "NONE"
            } for _ in range(expected_count)]
    
    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate và chỉ giữ lại type + context"""
        validated = []
        for entity in entities:
            if isinstance(entity, dict) and "type" in entity:
                validated.append({
                    "type": entity["type"],
                    "context": entity.get("context", "")
                })
        return validated