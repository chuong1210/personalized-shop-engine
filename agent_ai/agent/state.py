from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State của Agent - được truyền qua các node"""
    messages: List[BaseMessage]  # Lịch sử chat
    user_message: str  # Tin nhắn hiện tại
    topic: Optional[str]  # Kết quả phân loại topic
    sentiment: Optional[str]  # Kết quả phân tích cảm xúc
    entities: Optional[List[dict]]  # Entities được trích xuất
    purchase_intent: Optional[str]  # Ý định mua hàng
    confidence: Optional[float]  # Độ tin cậy
    reasoning: Optional[str]  # Lý do