"""
src/chat_agent.py
Task 4 — Natural-language "chat with your data" via a LangChain SQL agent.

Requires GOOGLE_API_KEY set as an environment variable.
"""
import os
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent

DB_PATH = "olist.db"
MODEL_NAME = "gemini-3.1-flash-lite"  # confirmed working — gemini-1.5-flash is deprecated (404)


def build_agent(db_path: str = DB_PATH, model_name: str = MODEL_NAME, verbose: bool = True):
    """Set up the SQL agent, connected to the existing olist.db database."""
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Set it before calling build_agent():\n"
            "  import os; os.environ['GOOGLE_API_KEY'] = 'your-key-here'"
        )

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=verbose,
    )
    return agent


def ask(agent, question: str) -> str:
    """Ask a plain-English question, get a plain-English answer."""
    result = agent.invoke({"input": question})
    return result["output"]


def chat_loop(agent) -> None:
    """Interactive loop — keep asking questions until you type 'exit'."""
    print("Ask anything about your order data. Type 'exit' to stop.\n")
    while True:
        question = input("You: ")
        if question.strip().lower() in ["exit", "quit"]:
            print("Session ended.")
            break
        try:
            answer = ask(agent, question)
            print(f"Answer: {answer}\n")
        except Exception as e:
            print(f"Something went wrong: {e}\n")


if __name__ == "__main__":
    agent = build_agent()
    chat_loop(agent)
