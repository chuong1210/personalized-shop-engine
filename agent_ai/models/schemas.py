from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Entity(BaseModel):
    """Thực thể được trích xuất từ chat"""
    type: Literal[
        "product_id", "sku_id", "order_code", 
        "shop_id", "brand_name", "category_name", "voucher_code"
    ]
    value: str
    context: Optional[str] = None  # Ngữ cảnh xung quanh entity

class AnalysisResult(BaseModel):
    """Kết quả phân tích chat"""
    topic: Literal[
        "TRA_CUU_DON_HANG",
        "HOI_PHI_SHIP",
        "HOI_CHINH_SACH",
        "TIM_KIEM_SAN_PHAM",
        "TU_VAN_SAN_PHAM",
        "HOI_KHUYEN_MAI",
        "KHIEU_NAI_SAN_PHAM",
        "KHIEU_NAI_GIAO_HANG",
        "LOI_HE_THONG",
        "UNKNOWN"
    ] = Field(description="Chủ đề của tin nhắn")
    
    sentiment: Literal[
        "POSITIVE", "NEUTRAL", "NEGATIVE", 
        "FRUSTRATED", "CONFUSED", "URGENT"
    ] = Field(description="Cảm xúc của người dùng")
    
    entities: List[Entity] = Field(
        default_factory=list,
        description="Các thực thể được trích xuất"
    )
    
    purchase_intent: Literal["HIGH", "MEDIUM", "LOW", "NONE"] = Field(
        description="Mức độ ý định mua hàng"
    )
    
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Độ tin cậy của phân tích"
    )
    
    reasoning: str = Field(
        description="Lý do đưa ra kết luận"
    )

class ChatRequest(BaseModel):
    """Request từ client"""
    message: str = Field(min_length=1, max_length=2000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Response trả về client"""
    success: bool
    data: Optional[AnalysisResult] = None
    error: Optional[str] = None

from datetime import datetime

class EventCreate(BaseModel):
    """Model để tạo event trong database"""
    app_name: str = "chat_analyzer"
    user_id: str
    session_id: str
    author: Literal["user", "agent", "system"] = "agent"
    content: Optional[str] = None

class EventResponse(BaseModel):
    """Model response khi tạo event thành công"""
    event_id: str
    session_id: str
    timestamp: datetime
    
class AnalysisWithEvent(BaseModel):
    """Response bao gồm cả analysis và event_id"""
    success: bool
    analysis: Optional[AnalysisResult] = None
    event_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None