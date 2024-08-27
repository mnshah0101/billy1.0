
import os
import sqlite3
import pandas as pd
import requests
from langchain.agents import initialize_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import time
from langchain_anthropic import ChatAnthropic
import re

# Define the prompt template
prompt_template = """


<prompt>

You are a conversational sports data assistant called Billy. You will be given a user question, a sql query to answer that question, and the result of the query. Then you will answer the question as best as you can. Use the sql query to understand what the response to the sql query might entail

This is the user question:

{user_question}


This is the sql query:
 {sql_query}


This is the result of the sql query:

{result}


Please answer the question: {user_question}


Format the response to look good on a chat interface. Make sure to be concise and clear. Do not include any special characters.

</prompt>

"""


billy_prompt = PromptTemplate.from_template(prompt_template)


def get_answer(model, question, query, sql_response):
    start = time.time()

    llm = None
    if model == 'openai':
        llm = ChatOpenAI(model='gpt-4o')

    elif model == 'anthropic':
        llm = ChatAnthropic(model_name='claude-3-opus-20240229',
                           )

    llm_chain = billy_prompt | llm

    answer = ''

    for s in llm_chain.stream(
        {'user_question': question, "sql_query": query, "result": sql_response}):
        yield s.content
        answer += str(s.content)

    return answer
