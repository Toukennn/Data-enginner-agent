import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config

        try: 
            self.connection = psycopg2.connect(**db_config) 

        except psycopg2.Error as exc:
            raise ConnectionError(
                "Could not connect to PostgreSQL. Check the database credentials."
            ) from exc

    def schema_details(self,schema_name):

        schema_info_context = ""
        
        connection = self.connection
        cursor = connection.cursor()

        schema_info_context = f"Database Schema: {schema_name}\n"

        try: 

            cursor.execute("SELECT table_name from information_schema.tables where table_schema = %s;", (schema_name,))
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context = f"{schema_info_context}\nTable: {table_name}\n"

                # Adding Columns & Data Types
                cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s;", (table_name,))
                columns_list = cursor.fetchall()

                for column in columns_list:
                    column_name = column[0]
                    data_type = column[1]
                    schema_info_context = f"{schema_info_context}  Column: {column_name}, Data Type: {data_type}\n"

                # Adding Sample Data
                cursor.execute(f"SELECT * FROM {schema_name}.{table_name} LIMIT 5;")
                sample_data = cursor.fetchall()
                schema_info_context = f"{schema_info_context}  Sample Data:\n"
                for row in sample_data:
                    schema_info_context = f"{schema_info_context}    {row}\n"

        except Exception as e:
            print(f"Error fetching schema details: {e}")
            schema_info_context = f"Error fetching schema details: {e}"

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
        
        return schema_info_context

    def execute_sql(self, query):
        try:
            connection = self.connection
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            return str(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()


def load_database_config():
    env_names = {
        "host": "host",
        "user": "user",
        "password": "password",
        "dbname": "database",
    }
    config = {key: os.getenv(env_name) for key, env_name in env_names.items()}
    missing = [env_name for key, env_name in env_names.items() if not config[key]]

    if missing:
        raise RuntimeError(
            f"Missing required database variables in .env: {', '.join(missing)}"
        )

    config["port"] = int(os.getenv("port", "5432"))
    return config


if __name__ == "__main__":
    obj = DatabaseUtil(load_database_config())
    result = obj.schema_details("public")

    with open("test_schema_details_example.txt", "w") as f:
        f.write(result)
