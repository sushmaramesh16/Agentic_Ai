from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.retriever import retrieve_documents
from agents.analyst import analyse_application
from agents.decision import make_decision

# Define the state that flows through the graph
class AgentState(TypedDict):
    user_query: str
    retrieved_docs: list[str]
    analysis: str
    decision: str
    full_response: str

# Node 1: Retriever
def retriever_node(state: AgentState) -> AgentState:
    print("\n🔍 Node 1: Retrieving relevant documents...")
    docs = retrieve_documents(state["user_query"])
    return {**state, "retrieved_docs": docs}

# Node 2: Analyst
def analyst_node(state: AgentState) -> AgentState:
    print("\n🧠 Node 2: Analysing application...")
    analysis = analyse_application(state["user_query"], state["retrieved_docs"])
    return {**state, "analysis": analysis}

# Node 3: Decision
def decision_node(state: AgentState) -> AgentState:
    print("\n⚖️  Node 3: Making final decision...")
    result = make_decision(state["user_query"], state["analysis"])
    return {**state, "decision": result["decision"], "full_response": result["full_response"]}

# Build the LangGraph
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "decision")
    graph.add_edge("decision", END)

    return graph.compile()

def run_agent(query: str) -> dict:
    app = build_graph()
    initial_state = AgentState(
        user_query=query,
        retrieved_docs=[],
        analysis="",
        decision="",
        full_response=""
    )
    result = app.invoke(initial_state)
    return result

if __name__ == "__main__":
    query = "Applicant has credit score of 650, annual income of 50k USD, and is requesting a loan of 100k USD. Should they be approved?"
    print(f"\n📋 Query: {query}\n")
    result = run_agent(query)
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    print(result["full_response"])