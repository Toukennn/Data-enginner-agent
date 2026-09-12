from agents.data_engineer import data_engineer
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    response = data_engineer.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
         "route_response": ""}
    )

    print(response)