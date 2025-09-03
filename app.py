import os
from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ---- Load environment variables ----
load_dotenv()

# ---- Define State ----
class State(TypedDict, total=False):  # total=False allows optional fields
    question: str
    answer: str
    a: int
    b: int
    sum: int

# ---- Initialize Google GenAI Client ----
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ---- Define Nodes ----
def ask_ai(state: State) -> State:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=state["question"]
    )
    return {"answer": response.text}

def add_numbers(state: State) -> State:
    total = state["a"] + state["b"]
    return {"sum": total}

# ---- Build Graphs ----
graph_ai = StateGraph(State)
graph_ai.add_node("ask_ai", ask_ai)
graph_ai.add_edge(START, "ask_ai")
graph_ai.add_edge("ask_ai", END)
app_ai = graph_ai.compile()

graph_add = StateGraph(State)
graph_add.add_node("add_numbers", add_numbers)
graph_add.add_edge(START, "add_numbers")
graph_add.add_edge("add_numbers", END)
app_add = graph_add.compile()

# ---- Run ----
if __name__ == "__main__":
    choice = input("What do you want to do? (1: Ask AI, 2: Add numbers): ")

    if choice == "1":
        q = input("Enter your question for AI: ")
        result = app_ai.invoke({"question": q})
        print("\nAI Answer:", result["answer"])

    elif choice == "2":
        a = int(input("Enter first number: "))
        b = int(input("Enter second number: "))
        result = app_add.invoke({"a": a, "b": b})
        print("\nSum:", result["sum"])

    else:
        print("Invalid choice. Please select 1 or 2.")
