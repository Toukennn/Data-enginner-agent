from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentSchema(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    user_question: str = ""
    curated_ques: str = ""
    prompt_query: str = ""

    is_safe: Literal["YES", "NO"] = "NO"

    generated_sql_query: str = ""
    comments: str = ""
    sql_query_execution_result: str = ""
    final_answer: str = ""


class JudgeSchema(BaseModel):
    answer: Literal["YES", "NO"]
    comments: str


class ETLAgentSchema(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


class RouterSchema(BaseModel):
    answer: Literal["sql", "etl"]
    comments: str = ""


class DataEngineerSchema(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    route_response: Literal["sql", "etl"] | None = None