import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from .state import AgentState
from .prompts import (
    TOPIC_CLASSIFIER_PROMPT,
    SENTIMENT_ANALYZER_PROMPT,
    ENTITY_EXTRACTOR_PROMPT,
    PURCHASE_INTENT_PROMPT
)
from dotenv import load_dotenv
import os
load_dotenv()

# Kiểm tra key
print(os.getenv("GOOGLE_API_KEY"))  # nên in ra key nếu load đúng

# Khởi tạo LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,  # Thấp để có kết quả ổn định
)

def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON response từ LLM, xử lý lỗi"""
    try:
        # Loại bỏ markdown code blocks nếu có
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": response}

# NODE 1: Phân loại Topic
def classify_topic_node(state: AgentState) -> Dict[str, Any]:
    """Node phân loại chủ đề của tin nhắn"""
    message = state["user_message"]
    
    prompt = TOPIC_CLASSIFIER_PROMPT.format(message=message)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    result = parse_json_response(response.content)
    
    return {
        "topic": result.get("topic", "UNKNOWN"),
        "confidence": result.get("confidence", 0.5),
        "reasoning": result.get("reasoning", "")
    }

# NODE 2: Phân tích Sentiment
def analyze_sentiment_node(state: AgentState) -> Dict[str, Any]:
    """Node phân tích cảm xúc"""
    message = state["user_message"]
    topic = state.get("topic", "UNKNOWN")
    
    prompt = SENTIMENT_ANALYZER_PROMPT.format(
        message=message,
        topic=topic
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    
    result = parse_json_response(response.content)
    
    return {
        "sentiment": result.get("sentiment", "NEUTRAL")
    }

# NODE 3: Trích xuất Entities
def extract_entities_node(state: AgentState) -> Dict[str, Any]:
    """Node trích xuất thực thể"""
    message = state["user_message"]
    
    prompt = ENTITY_EXTRACTOR_PROMPT.format(message=message)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    result = parse_json_response(response.content)
    
    return {
        "entities": result.get("entities", [])
    }

# NODE 4: Đánh giá Purchase Intent
def evaluate_purchase_intent_node(state: AgentState) -> Dict[str, Any]:
    """Node đánh giá ý định mua hàng"""
    message = state["user_message"]
    topic = state.get("topic", "UNKNOWN")
    sentiment = state.get("sentiment", "NEUTRAL")
    entities = state.get("entities", [])
    
    prompt = PURCHASE_INTENT_PROMPT.format(
        message=message,
        topic=topic,
        sentiment=sentiment,
        entities=json.dumps(entities, ensure_ascii=False)
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    
    result = parse_json_response(response.content)
    
    return {
        "purchase_intent": result.get("purchase_intent", "NONE"),
        "reasoning": state.get("reasoning", "") + " " + result.get("reasoning", "")
    }