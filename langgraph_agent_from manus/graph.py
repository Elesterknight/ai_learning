from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import State
from nodes import (
    report_node,
    execute_node,
    create_planner_node,
    update_planner_node
)


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "create_planner")
    builder.add_node("create_planner", create_planner_node)
    builder.add_node("update_planner", update_planner_node)
    builder.add_node("execute", execute_node)
    builder.add_node("report", report_node)
    builder.add_edge("report", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    memory = MemorySaver()
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()


inputs = {"user_message": "對所給文檔進行分析，生成分析報告，文檔路徑為student_habits_performance.csv", #kaggle的學生習慣表現資料集
          "plan": None,
          "observations": [], 
          "final_report": ""}

graph.invoke(inputs, {"recursion_limit":100})