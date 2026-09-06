import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from models.schema import AgentSchema


# -------------- AI AGENT Code -------------------
def curate_question(state: AgentSchema) -> AgentSchema: 
    """
    Curates the question based on the messages and context provided in the state.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state with curated question.
    """

    user_question = state.user_question # the dot notation is used since it is a Pydantic model object
    llm = pick_llm("low")
    response = llm.invoke(f"Curate the following question: {user_question}")
    state.curated_ques = response
    return state


def prompy_query_context(state: AgentSchema) -> AgentSchema: 
    """
    Generates a detailed prompt with SQL DB context based on the curated question.
    Args:
        state (AgentSchema): The current state of the agent containing messages and context.

    Returns:
        AgentSchema: Updated state with prompt query context.
    """

    curated_question = state.curated_ques
    