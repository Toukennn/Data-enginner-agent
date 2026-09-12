from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from models.schema import ETLAgentSchema
from utils.etl_tools import ETLTools
from utils.llm_pick import pick_llm


# ============================================================
# TOOLS
# ============================================================

@tool
def extract_load_tool(
    url: str,
    output_folder: str = "data/extract",
    format: str = "csv",
) -> str:
    """
    Extract data from an API endpoint and save it locally.

    Args:
        url:
            API endpoint from which data should be extracted.

        output_folder:
            Folder where the extracted data should be saved.
            Defaults to "data/extract".

        format:
            Output format.
            Supported values: csv, json, parquet.

    Returns:
        A message describing whether the extraction succeeded or failed.
    """

    etl_tools = ETLTools()

    return etl_tools.extract_load(
        url=url,
        output_folder=output_folder,
        format=format,
    )


@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str = "data/transform",
    output_format: str = "csv",
    user_question: str = "",
) -> str:
    """
    Transform an existing dataset according to the user's request
    and save the result.

    Args:
        input_file_path:
            Path to the file that should be transformed.

        output_folder:
            Folder where the transformed dataset should be saved.
            Defaults to "data/transform".

        output_format:
            Output format.
            Supported values: csv, json, parquet.

        user_question:
            Natural-language description of the requested transformation.

    Returns:
        A message describing the transformation and its result.
    """

    etl_tools = ETLTools()

    # Give the LLM some context about the dataset
    dataset_context = etl_tools.transform_load_context(
        input_file_path
    )

    transformation_llm = pick_llm("claude")

    prompt = f"""
You are a Python data transformation specialist.

Your task is to generate Pandas code that transforms an existing dataset
according to the user's request.

Input file:
{input_file_path}

Output folder:
{output_folder}

Requested output format:
{output_format}

User request:
{user_question}

Dataset preview:
{dataset_context}

Requirements:

1. Load the dataset from the input file.
2. Perform only the transformations required by the user.
3. Save the transformed result inside:
   {output_folder}
4. Save it using this format:
   {output_format}
5. Return ONLY valid Python/Pandas code.
6. Do not include Markdown code fences.
7. Do not include explanations.
"""

    response = transformation_llm.invoke(prompt)

    pandas_code = response.content.strip()

    # Remove accidental Markdown fences if the model still produces them
    if pandas_code.startswith("```python"):
        pandas_code = pandas_code[len("```python"):]

    elif pandas_code.startswith("```"):
        pandas_code = pandas_code[len("```"):]

    if pandas_code.endswith("```"):
        pandas_code = pandas_code[:-3]

    pandas_code = pandas_code.strip()

    # NOTE:
    # This still executes LLM-generated code.
    # We will replace this with a safer transformation architecture
    # in Step 1B.
    execution_result = etl_tools.execute_code(
        pandas_code
    )

    return (
        f"Transformation request processed.\n\n"
        f"Output folder: {output_folder}\n"
        f"Output format: {output_format}\n\n"
        f"Execution result:\n{execution_result}\n\n"
        f"Generated Pandas code:\n{pandas_code}"
    )


# ============================================================
# TOOLKIT
# ============================================================

tools = [
    extract_load_tool,
    transform_load_tool,
]

tools_by_name = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# LLM
# ============================================================

etl_llm = pick_llm("claude")

etl_llm_with_tools = etl_llm.bind_tools(
    tools
)


# ============================================================
# GRAPH NODES
# ============================================================

def llm_node(state: ETLAgentSchema):
    """
    Ask the ETL agent what action should be taken next.

    The LLM may either:

    - call one of the available ETL tools
    - return a final answer to the user
    """

    system_prompt = """
You are an ETL specialist agent operating inside a larger
Data Engineer agent.

Your responsibility is to perform ETL-related tasks.

You have access to two tools:

1. extract_load_tool

   Use this when the user wants to extract data from an API
   and save it locally.

2. transform_load_tool

   Use this when the user wants to clean, filter, aggregate,
   reshape, transform, or otherwise modify an existing dataset.

Rules:

- Use the appropriate tool whenever the task requires an ETL operation.
- Do not claim an operation succeeded unless a tool actually executed it.
- If the user does not specify an extraction folder, use:
  data/extract
- If the user does not specify a transformation folder, use:
  data/transform
- If the user does not specify an output format, use:
  csv
- After the required tool operations are completed, provide a short,
  clear summary of what was done.
- Do not expose unnecessary implementation details.
"""

    conversation = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *state.messages,
    ]

    response = etl_llm_with_tools.invoke(
        conversation
    )

    # Important:
    # Because ETLAgentSchema uses LangGraph's add_messages reducer,
    # we return ONLY the new message.
    return {
        "messages": [response]
    }


def tool_node(state: ETLAgentSchema):
    """
    Execute tool calls requested by the ETL LLM.
    """

    last_message = state.messages[-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        [],
    )

    tool_messages = []

    for tool_call in tool_calls:

        tool_name = tool_call["name"]

        if tool_name not in tools_by_name:
            tool_messages.append(
                ToolMessage(
                    content=f"Unknown tool requested: {tool_name}",
                    tool_call_id=tool_call["id"],
                )
            )

            continue

        selected_tool = tools_by_name[
            tool_name
        ]

        try:
            result = selected_tool.invoke(
                tool_call["args"]
            )

        except Exception as exc:
            result = (
                f"Tool execution failed: "
                f"{type(exc).__name__}: {exc}"
            )

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    # Again, return ONLY the new messages.
    return {
        "messages": tool_messages
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_llm(
    state: ETLAgentSchema,
) -> str:
    """
    Decide whether the agent should execute a tool
    or finish the workflow.
    """

    last_message = state.messages[-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        [],
    )

    if tool_calls:
        return "tools"

    return "end"


# ============================================================
# GRAPH
# ============================================================

etl_graph = StateGraph(
    ETLAgentSchema
)

etl_graph.add_node(
    "llm",
    llm_node,
)

etl_graph.add_node(
    "tools",
    tool_node,
)


etl_graph.add_edge(
    START,
    "llm",
)


etl_graph.add_conditional_edges(
    "llm",
    route_after_llm,
    {
        "tools": "tools",
        "end": END,
    },
)


etl_graph.add_edge(
    "tools",
    "llm",
)


etl_analyst = etl_graph.compile()


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == "__main__":

    from langchain_core.messages import HumanMessage

    test_input = {
        "messages": [
            HumanMessage(
                content=(
                    "Extract the data from "
                    "https://pokeapi.co/api/v2/pokemon "
                    "and save it as CSV in data/extract."
                )
            )
        ]
    }

    result = etl_analyst.invoke(
        test_input
    )

    print("\n--- Final ETL Agent Response ---\n")

    print(
        result["messages"][-1].content
    )