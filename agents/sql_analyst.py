import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.database import DatabaseUtil
from models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# -------------- AI AGENT Code -------------------
def curated_ques(state: AgentSchema) -> AgentSchema: 
    """
    Curates the question based on the messages and context provided in the state.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state with curated question.
    """

    user_question = state.user_question # the dot notation is used since it is a Pydantic model object
    llm = pick_llm("low")
    response = llm.invoke(f"Curate the following question: {user_question}").content
    state.curated_ques = response
    state.messages = state.messages + [HumanMessage(content=f"{response}")] # Append the LLM's response to the messages
    # note that we have already used "add" when defining messages in our schema, so 
    # it would be added to the list automatically even if you don't append it to the list 
    # explicitly, but to make the code more readable, we append it here anyway!

    return state


def prompt_query(state: AgentSchema) -> AgentSchema: 
    """
    Generates a detailed prompt with SQL DB context based on the curated question.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state with prompt query context.
    """

    curated_ques = state.curated_ques

    conn_details = {
        "host": os.getenv("host"),
        "user": os.getenv("user"), 
        "password": os.getenv("password"),
        "dbname": os.getenv("database"),
        "port": int(os.getenv("port", "5432"))
    }

    obj = DatabaseUtil(conn_details)

    schema_info = obj.schema_details("public") # Assuming the schema is public, you can modify this as needed

    # now you should give the llm a very detailed prompt based on the concepts of 
    # Prompt Engineering! 
    prompt = f"""
    
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. You are provided 
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    User's Original Query: {curated_ques}

    Database Schema Details:
    {schema_info}

    """

    state.prompt_query = prompt

    return state


# Generate SQL query Node
def generate_sql(state: AgentSchema) -> AgentSchema:

    prompt = state.prompt_query
    llm = pick_llm("medium") # pick a more powerful LLM for generating the SQL query
    generated_sql_query = llm.invoke(prompt).content # the LLM creates a good prompt to use

    state.generated_sql_query = generated_sql_query

    return state


def is_safe_sql(state: AgentSchema) -> AgentSchema: 
    """
    Determines whether the generated SQL query is safe to execute.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state with safety status of the SQL query.
    """

    sql_query = state.generated_sql_query
    llm = pick_llm("medium") 
    llm_judge = llm.with_structured_output(JudgeSchema)

    # sql_query = "DELETE FROM users WHERE age > 30;"  Example query to check

    prompt = f"""
    You are a SQL Judge for data security. Your task is to determine whether the SQL
    query is safe or not. The SQL query should only be used for data retreival and should not 
    modify the database in any way. Neither the SQL query nor the prompt should contain any
    SQL commands that can modify the database, such as INSERT, UPDATE, DELETE, DROP, ALTER, 
    TRUNCATE, CREATE, or any other commands that can change the structure or data of the database.
    If the query is safe, respond with 'YES' and provide a brief explanation of why it is safe.
    Otherwise respond with 'NO' and provide a brief explanation of why it is not safe. 
    Here is the SQL query to analyze:
    {sql_query}
    """

    response = llm_judge.invoke(prompt).model_dump() # This way we get the output as a dictionary
    state.is_safe = response['answer']
    state.comments = response['comments']

    return state


# Cancel SQL query node
def canceled_sql(state: AgentSchema) -> AgentSchema: 
    """
    Cancels the SQL query execution if it is deemed unsafe and specifies a reason for it.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state indicating that the SQL query execution has been canceled.
    """

    comments = state.comments
    state.final_answer = f"SQL query execution has been canceled due to safety concerns. Reason provided by the judge: {comments}"
    state.messages = state.messages + [AIMessage(content=f"{state.final_answer}")]

    return state 


# Execute SQL query node
def execute_sql(state: AgentSchema) -> AgentSchema: 
    """
    Executes the generated SQL query on the database and stores the result.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state with the result of executing the SQL query.
    """

    sql_query = state.generated_sql_query

    conn_details = {
        "host": os.getenv("host"),
        "user": os.getenv("user"),
        "password": os.getenv("password"),
        "dbname": os.getenv("database"),
        "port": int(os.getenv("port", "5432"))
    }

    obj = DatabaseUtil(conn_details) 
    execution_result = obj.execute_sql(sql_query) # function in our database.py script using cursor

    state.sql_query_execution_result = execution_result

    return state



# Represent the final answer Node
def represent_final_answer(state: AgentSchema) -> AgentSchema: 
    """
    Represents the final answer based on the execution result of the SQL query.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state with the final answer.
    """

    execution_result = state.sql_query_execution_result
    curated_ques = state.curated_ques

    llm = pick_llm("low") # pick a less powerful LLM for representing the final answer

    prompt = f"""
    You are an SQL analyst agent. Your task is to provide a final answer based on the
    execution result of the SQL query and the user's original question. The final answer should
    be concise, clear and directly address the user's query. Avoid including any SQL code or
    technical details in the final answer. The final answer should be in a user-friendly format
    that is easy to understand. If the execution result is empty or does not provide a clear
    answer to the user's question, explain this in the final answer\n.

    The SQL query was: {curated_ques} \n 
    The execution result is: {execution_result}
    """

    llm_response = llm.invoke(prompt).content # this way we get only the final asnwer from the LLM

    state.final_answer = llm_response
    state.messages = state.messages + [AIMessage(content=f"{llm_response}")] 
    # we append the final answer to the messages list in our schema

    return state


# now, all the nodes of our graph are created, we should now add the edges of our graph!

# GRAPH BUILDING:
sql_agent_graph = StateGraph(AgentSchema, )

# NODES: (you might wanna make sure the names are the same for debugging purposes!)
sql_agent_graph.add_node(curated_ques, name="curated_ques")
sql_agent_graph.add_node(prompt_query, name="prompt_query")
sql_agent_graph.add_node(generate_sql, name="generate_sql")
sql_agent_graph.add_node(is_safe_sql, name="is_safe_sql")
sql_agent_graph.add_node(canceled_sql, name="canceled_sql")
sql_agent_graph.add_node(execute_sql, name="execute_sql")
sql_agent_graph.add_node(represent_final_answer, name="represent_final_answer")

# EDGES: 
sql_agent_graph.add_edge(START, "curated_ques")
sql_agent_graph.add_edge("curated_ques", "prompt_query")
sql_agent_graph.add_edge("prompt_query", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

# Now we have a conditional edge so we should define a function for it!
def is_safe_sql_edge(state: AgentSchema) -> str:
    is_safe = state.is_safe

    if is_safe.lower() == "yes": 
        return "execute_sql" # note that here you should put the exact same name of your node!
    else: 
        return "canceled_sql" 

sql_agent_graph.add_conditional_edges("is_safe_sql", is_safe_sql_edge,
                                      {
                                          "execute_sql": "execute_sql", 
                                          "canceled_sql": "canceled_sql"
                                      }) # the fucntion itself chooses the next node

# it is important to know that you cannot change state from 2 different nodes simultaneously!
# sql_agent_graph.add_edge("is_safe_sql", "execute_sql")
# sql_agent_graph.add_edge("is_safe_sql", "canceled_sql")

sql_agent_graph.add_edge("canceled_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)

# and now this agent is ready!!
if __name__ == "__main__":

    # COMPILE THE GRAPH: 
    sql_analyst = sql_agent_graph.compile() # basically created the graph to be used next

    # visualising the graph (optional):
    from IPython.display import display, Image
    img = Image(sql_analyst.get_graph().draw_mermaid_png())
    with open("sql_analyst_graph.png", "wb") as f: 
        f.write(img.data)

    input_schema = { # should be the same as AgentSchema be careful!
        "messages": [], 
        "user_question": "What are the different types of Payment Methods we have in out database?", 
        "curated_ques": "", 
        "prompt_query": "", 
        "generated_sql_query": "", 
        "is_safe": "NO", 
        "comments": "", 
        "sql_query_execution_result": "", 
        "final_answer": ""
    }

    # Execute the graph 
    sql_analyst_response = sql_analyst.invoke(input_schema) 
    # be careful that the name you choose for .execute() should be the same as the one you
    # choose to get compiled (.compile())

    print(sql_analyst_response['messages'])
    print("***************************")
    print(sql_analyst_response['generated_sql_query'])
    print("***************************")
    print(sql_analyst_response['sql_query_execution_result'])
    print("***************************")
    print(sql_analyst_response['prompt_query'])