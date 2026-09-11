from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def pick_llm(level: str): 
    """
    Picks the appropriate LLM based on the level of the question.
    Args: 
        level (str): The level of the question, can be "low", "medium", or "high".

    Returns:
        str: The name of the LLM to be used. 
    """

    if level.lower() == "low": 
        llm = ChatOpenAI(model_name="gpt-5.6-luna", temperature=0, model_kwargs={
            "reasoning_effort": "none"
        })
    elif level.lower() == "medium":
        llm = ChatOpenAI(model_name="gpt-5.6-terra", temperature=0, model_kwargs={
            "reasoning_effort": "none"
        })
    elif level.lower() == "high":
        llm = ChatOpenAI(model_name="gpt-5.6-sol", temperature=0, model_kwargs={
            "reasoning_effort": "none"
        })
    elif level.lower() == "claude": 
        llm = ChatAnthropic(model_name="claude-sonnet-5")
    
    else:
        raise ValueError("Invalid level. Please choose from 'low', 'medium', or 'high'.")

    return llm

if __name__ == "__main__":
    llm_obj = pick_llm("low")  # Example usage, you can change the level as needed
    print(llm_obj.invoke("What is the capital of Switzerland?"))