from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    classify_topic_node,
    analyze_sentiment_node,
    extract_entities_node,
    evaluate_purchase_intent_node
)

def build_analysis_graph():
    """Xây dựng graph phân tích chat"""
    
    # Khởi tạo graph với AgentState
    workflow = StateGraph(AgentState)
    
    # Thêm các node
    workflow.add_node("classify_topic", classify_topic_node)
    workflow.add_node("analyze_sentiment", analyze_sentiment_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("evaluate_intent", evaluate_purchase_intent_node)
    
    # Định nghĩa luồng:
    # START → classify_topic → analyze_sentiment → extract_entities → evaluate_intent → END
    workflow.set_entry_point("classify_topic")
    workflow.add_edge("classify_topic", "analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "extract_entities")
    workflow.add_edge("extract_entities", "evaluate_intent")
    workflow.add_edge("evaluate_intent", END)
    
    # Compile graph
    graph = workflow.compile()
    
    return graph

# Khởi tạo graph global để dùng trong API
analysis_graph = build_analysis_graph()