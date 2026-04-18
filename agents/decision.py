import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def make_decision(user_query: str, analysis: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )
    messages = [
        SystemMessage(content="""You are a loan approval officer. Based on an analyst report, give a final decision.
Always respond in this exact format:

DECISION: [GREEN / YELLOW / RED]
REASON: [One clear sentence]
RECOMMENDATION: [One actionable next step]

GREEN = Approved, YELLOW = Manual review, RED = Rejected"""),
        HumanMessage(content=f"""
## Original Application:
{user_query}

## Analyst Report:
{analysis}

## Give your final decision:
""")
    ]
    response = llm.invoke(messages)
    content = response.content
    decision = "YELLOW"
    if "DECISION: GREEN" in content:
        decision = "GREEN"
    elif "DECISION: RED" in content:
        decision = "RED"
    print(f"✅ Decision agent completed: {decision}")
    return {"decision": decision, "full_response": content}
