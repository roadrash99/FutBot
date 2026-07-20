import os
import requests
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

load_dotenv()

FOOTBALL_API_BASE_URL = "https://api.football-data.org/v4"


class State(TypedDict):
    query: str
    messages: list
    match_data: list


def fetch_match_data(competition: str = "PL", season: int = 2025, team: str = "Manchester United FC") -> list:
    """Fetch a team's finished match results for a given season from the football-data.org API.

    `season` is the year the season started, e.g. 2025 for the 2025/26 season.
    """
    response = requests.get(
        f"{FOOTBALL_API_BASE_URL}/competitions/{competition}/matches",
        headers={"X-Auth-Token": os.environ.get("FOOTBALL_API_KEY")},
        params={
            "status": "FINISHED",
            "season": season,
        },
    )
    response.raise_for_status()
    matches = response.json().get("matches", [])

    return [
        {
            "date": match["utcDate"][:10],
            "matchday": match["matchday"],
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "home_score": match["score"]["fullTime"]["home"],
            "away_score": match["score"]["fullTime"]["away"],
        }
        for match in matches
        if team in (match["homeTeam"]["name"], match["awayTeam"]["name"])
    ]


def fetch_match_data_node(state: State) -> State:
    """LangGraph node that populates state['match_data'] via the football API."""
    state["match_data"] = fetch_match_data()
    return state


def llm_node(state: State) -> State:
    """LangGraph node that calls Groq LLM."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    context = f"Match data:\n{state['match_data']}\n\nQuestion: {state['query']}"

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": context}],
        model="llama-3.3-70b-versatile",
    )

    answer = chat_completion.choices[0].message.content
    state["messages"].append({"role": "assistant", "content": answer})

    return state


def build_graph():
    """Build and compile the LangGraph."""
    graph = StateGraph(State)
    graph.add_node("fetch_match_data", fetch_match_data_node)
    graph.add_node("llm_call", llm_node)
    graph.add_edge(START, "fetch_match_data")
    graph.add_edge("fetch_match_data", "llm_call")
    graph.add_edge("llm_call", END)
    return graph.compile()


def main():
    # Example usage
    question = "What were Manchester United's results in the last 5 games?"
    app = build_graph()

    result = app.invoke({
        "query": question,
        "messages": [{"role": "user", "content": question}],
        "match_data": []
    })

    print(f"Q: {question}")
    print(f"A: {result['messages'][-1]['content']}")


if __name__ == "__main__":
    main()
    