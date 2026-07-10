import os
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

load_dotenv()


class State(TypedDict):
    query: str
    messages: list


def llm_node(state: State) -> State:
    """LangGraph node that calls Groq LLM."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": state["query"]}],
        model="llama-3.3-70b-versatile",
    )

    answer = chat_completion.choices[0].message.content
    state["messages"].append({"role": "assistant", "content": answer})

    return state


def build_graph():
    """Build and compile the LangGraph."""
    graph = StateGraph(State)
    graph.add_node("llm_call", llm_node)
    graph.add_edge(START, "llm_call")
    graph.add_edge("llm_call", END)
    return graph.compile()


def main():
    # Example usage
    question = "What was the score in yesterday's match?"
    app = build_graph()

    result = app.invoke({
        "query": question,
        "messages": [{"role": "user", "content": question}]
    })

    print(f"Q: {question}")
    print(f"A: {result['messages'][-1]['content']}")


if __name__ == "__main__":
    main()
    