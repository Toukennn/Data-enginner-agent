import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.ETL_tools import ETLTools
from models.schema import ETLAgentSchema, RouterSchema, DataEngineerSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool 
from langchain_anthropic import ChatAnthropic
from agents.ETL_analyst import etl_analyst # note that you import them from the root so that your main.py can see them!
from agents.SQL_analyst import sql_analyst

llm = pick_llm("medium")
llm_router = llm.with_structured_output(RouterSchema)

# print(llm_router.invoke("I want to extract data from an API and save it as a csv file"))  

# creating the Data Engineer graph 

def router_node(state: DataEngineerSchema): 

    message = state.messages[-1].content
    route_response_dict = llm_router.invoke(message).model_dump()
    route_response = route_response_dict["answer"]
    state.route_response = route_response

    return state

def etl_node(state:DataEngineerSchema):

    message = state.messages[-1].content

    etl_prompt = f"""
        You are the ETL analyst agent, invoked by the Data Engineer orchestrator
        because the user's request was classified as an ETL task.

        Original user request:
        \"\"\"{message}\"\"\"

        You have two tools available:
        - extract_load_tool(url, output_folder, format): pulls data from an API endpoint
          and saves it locally.
        - transform_load_tool(input_file_path, output_folder, output_format, user_question):
          reads an existing file, writes and runs Pandas code to transform it per the
          user's question, and saves the result.

        Instructions:
        - Decide which tool(s) the request needs. If it asks to pull data from a URL/API,
          use extract_load_tool. If it asks to clean, filter, aggregate, join, or reshape
          data already sitting in a file, use transform_load_tool. If it needs both,
          extract first and then transform the extracted output.
        - If output_folder or format aren't specified, default to "data/extract" for
          extraction and "data/transform" for transformation, and "csv" for format.
        - Do not ask the user clarifying questions — infer sensible defaults and proceed.
        - When finished, reply with a concise summary of what was done and where the
          output file(s) were saved.
    """

    response = etl_analyst.invoke(
        {"messages": [HumanMessage(content=etl_prompt)]}
    )

    state.messages = state.messages + response["messages"]
    return state


def sql_node(state: DataEngineerSchema): 

    message = state.messages[-1].content

    input_schema = {
        "messages": [], 
        "user_question": f"{message}", 
        "curated_ques": "", 
        "prompt_query": "", 
        "generated_sql_query": "", 
        "is_safe": "NO", 
        "comments": "", 
        "sql_query_execution_result": "", 
        "final_answer": ""
    }

    response = sql_analyst.invoke(input_schema)

    state.messages = state.messages + [response]

    return state


data_engineer_graph = StateGraph(DataEngineerSchema)

data_engineer_graph.add_node("router_node", router_node)
data_engineer_graph.add_node("etl_node", etl_node)
data_engineer_graph.add_node("sql_node", sql_node)

data_engineer_graph.add_edge(START, "router_node")

def route_edge(state: DataEngineerSchema) -> str: 
    if state.route_response == "sql": 
        return "sql_node" 
    elif state.route_response == 'etl':
        return "etl_node"
    else: 
        raise ValueError(f"Invalid route response: {state.route_response}")

data_engineer_graph.add_conditional_edges("router_node", route_edge,
                                          {
                                              "sql_node": "sql_node",
                                              "etl_node": "etl_node"
                                          })

data_engineer_graph.add_edge("sql_node", END)
data_engineer_graph.add_edge("etl_node", END)

data_engineer = data_engineer_graph.compile()

from IPython.display import display, Image
img = Image(data_engineer.get_graph().draw_mermaid_png())
with open("data_engineer_graph.png", "wb") as f: 
    f.write(img.data)


if __name__ == "__main__":
    # response = data_engineer.invoke(
    #    {"messages": [HumanMessage(content=f"What are the different types of payment methods we have in our database?")],
    #     "route_response": ""}
    # )

    # print(response)

    response = data_engineer.invoke(
        {"messages": [HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
         "route_response": ""}
    )

    print(response)