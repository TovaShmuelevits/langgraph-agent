import os
import httpx
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- 1. מפתח ה-API ---
os.environ["GROQ_API_KEY"] = "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"

# --- 2. עקיפת בדיקת SSL עבור סינון אינטרנט ---
custom_http_client = httpx.Client(verify=False)

# --- 3. הגדרת כלי חישוב ---
@tool
def multiply(a: int, b: int) -> int:
    """מכפיל שני מספרים שלמים ומחזיר את התוצאה."""
    return a * b

tools = [multiply]
tool_node = ToolNode(tools)

# --- 4. הגדרת ה-State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 5. הגדרת המודל עם הגבלת max_tokens לבריחת ה-OTPM ---
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    max_tokens=250,  # שומר על פלט נמוך מ-1,000 טוקנים בדקה
    http_client=custom_http_client
)
llm_with_tools = llm.bind_tools(tools)

# --- 6. צומת הסוכן ---
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- 7. תנאי מעבר ---
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 8. בניית הגרף ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- 9. הרצה ---
if __name__ == "__main__":
    prompt = "כמה זה 15 כפול 24? תשתמש בכלי החישוב."
    print(f"שאלה: {prompt}\n")
    
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"=== שלב: {key} ===")
            latest_msg = value["messages"][-1]
            if latest_msg.content:
                print(f"תשובה: {latest_msg.content}")
            if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                print(f"הפעלת כלי: {latest_msg.tool_calls}")
            print("-" * 40)