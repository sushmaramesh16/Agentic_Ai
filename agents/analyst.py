import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

def analyse_application(user_query: str, retrieved_docs: list[str]) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])
    messages = [
        SystemMessage(content="You are a senior financial analyst at a bank. Analyze loan applications against lending guidelines. Be concise and professional."),
        HumanMessage(content=f"""
## Applicant Query:
{user_query}

## Relevant Lending Guidelines:
{context}

## Your Task:
Analyze this application step by step:
1. Identify key applicant metrics
2. Check each metric against the guidelines
3. Identify red flags or positive signals
4. Provide clear reasoning

Do NOT give the final decision yet.
""")
    ]
    response = llm.invoke(messages)
    print("🧠 Analyst agent completed reasoning")
    return response.content
