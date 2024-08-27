import re
import dotenv
import os
import psycopg2

from langchain.agents import initialize_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import time
from langchain_anthropic import ChatAnthropic

dotenv.load_dotenv()



DATABASE_URL = os.getenv('DATABASE_URL')


prompt_template = """

There was an error executing this sql query:

{query}


This was the error message: {error_message}

Give me the fixed query to run. Do not include any other information, I want to see the fixed query only. Also fix the logic of the query if needed.

"""

prompt_template = PromptTemplate.from_template(prompt_template)

def new_sql_query(old_query, error_message, model):

    llm = None
    if model == 'openai':
        llm = ChatOpenAI(model='gpt-4o')

    elif model == 'anthropic':
        llm = ChatAnthropic(model_name='claude-3-5-sonnet-20240620')

    llm_chain = prompt_template | llm
    answer = llm_chain.invoke({'query': old_query, 'error_message': error_message})

    print(answer.content)

    return extract_sql_query(answer.content)



def execute_query(query, r =0):

    if r == 5:
        return 'Error : Cannot not execute query'

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(DATABASE_URL)

    # Create a cursor
    cur = conn.cursor()

    # Execute the query
    try:
        cur.execute(query)
    except Exception as e:
        print(f'Error: {e}. Retrying...')
        return execute_query(new_sql_query(query, str(e), 'anthropic'), r+1)
        
        

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows


def extract_sql_query(input_string):
    if 'sql' not in input_string:
        return input_string
    # Use regular expression to find the SQL query within the triple quotes
    match = re.search(r'```sql\n(.*?)\n```', input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None
