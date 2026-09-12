from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst
from models.schema import DataEngineerSchema, RouterSchema
from utils.llm_pick import pick_llm


# -------------------------------------------------------------------
# LLM configuration
# -------------------------------------------------------------------

router_llm = pick_llm("medium").with_structured_output(RouterSchema)


# -------------------------------------------------------------------
# Nodes
# -------------------------------------------------------------------

def router_node(state: DataEngineerSchema):
    """Route the user's request to the appropriate specialist agent."""

    user_message = state.messages[-1].content

    route_response = router_llm.invoke(user_message)

    return {
        "route_response": route_response.answer
    }


def etl_node(state: DataEngineerSchema):
    """Delegate ETL requests to the ETL specialist."""

    user_message = state.messages[-1].content

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(content=user_message)
            ]
        }
    )

    final_message = response["messages"][-1]

    return {
        "messages": [
            AIMessage(content=final_message.content)
        ]
    }


def sql_node(state: DataEngineerSchema):
    """Delegate SQL requests to the SQL specialist."""

    user_message = state.messages[-1].content

    response = sql_analyst.invoke(
        {
            "user_question": user_message
        }
    )

    return {
        "messages": [
            AIMessage(content=response["final_answer"])
        ]
    }


# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------

def route_request(state: DataEngineerSchema) -> str:

    route = state.route_response

    if route == "sql":
        return "sql"

    if route == "etl":
        return "etl"

    raise ValueError(f"Unsupported route: {route}")


# -------------------------------------------------------------------
# Graph
# -------------------------------------------------------------------

graph = StateGraph(DataEngineerSchema)

graph.add_node("router", router_node)
graph.add_node("etl", etl_node)
graph.add_node("sql", sql_node)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_request,
    {
        "sql": "sql",
        "etl": "etl",
    },
)

graph.add_edge("sql", END)
graph.add_edge("etl", END)

data_engineer = graph.compile()


# -------------------------------------------------------------------
# Local debugging
# -------------------------------------------------------------------

if __name__ == "__main__":

    response = data_engineer.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I want to extract data from "
                        "https://pokeapi.co/api/v2/pokemon "
                        "and save it as CSV."
                    )
                )
            ]
        }
    )

    print(response)