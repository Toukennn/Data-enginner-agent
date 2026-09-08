import os
import requests
import pandas as pd

class ETLTools:

    def __init__(self):
        pass

    def extract_load(self,url:str, output_folder:str, format:str): # Layer L1
        """
        This tool extracts the data from the API (url) and loads it into the
        the desired location (output_folder).

        Args:
            url (str): The API endpoint from which to extract data.
            output_folder (str): The folder where the extracted data will be saved.
        
        Returns:
            str: A message indicating the success or failure of the operation.

        """

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_folder = os.path.join(project_root, output_folder)      

        try:
            response = requests.get(url)
            response.raise_for_status()
            data  = response.json()

            filename = os.path.join(output_folder, f"extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)

            df = pd.json_normalize(data['results'])
            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format: {format}"

            return f"Data successfully extracted and saved to {filename}"

        except requests.exceptions.RequestException as e:
            return f"Failed to extract data: {e}"


    def transform_load_context(self, file_path: str, output_folder: str, format: str): 
        """
        This tool transforms the data from the specified file and loads it inot the 
        desired location (output_folder) 

        Args: 
            file_path (str): The path to the file containing the data to be transformed. 
            output_folder (str): The folder where the transformed data will be saved.

        Returns: 
            str: A message indicating the success or failure of the operation. 
        """


if __name__ == "__main__":
    obj = ETLTools()
    # Example usage of the tool: 
    print(obj.extract_load("https://pokeapi.co/api/v2/pokemon/", "data/extract", "csv"))