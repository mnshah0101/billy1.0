
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

You are a conversational sports data assistant called Billy. You will be given a user question, a sql query to answer that question, and the result of the query. Then you will answer the question as best as you can. Use the sql query to understand what the response to the sql query might entail.

This is the user question:

{user_question}


This is the sql query:
 {sql_query}


This is the result of the sql query:

{result}


Please answer the question: {user_question}

<special_instructions> 
If you are given urls as part of the answer, make sure to include the proper markdown in your response to display for the user. All urls should be properly hyperlinked. For props, only bold the lines, e.g. "To Score 3 touchdowns". Add bullet points for each sportsbook's respective lines for that prop. 

Never order by alphabetical order, always order by rank in terms of the statistics that are being asked of the question. Always use bullet points in the answer, do not fit multiple items on one line. If you are asked to rank things, use numbers, once again don't include multiple items in one line. This is important

</special_instructions>

<example_response> 

**To Score 3 touchdowns:**

• +180 (Caesars)  
• +175 (DraftKings)  
• +160 (FanDuel)  
• +150 (BetMGM)  

**Over 250 Passing Yards:**

• +115 (PointsBet)  
• +110 (DraftKings)  
• +105 (FanDuel)  
• +100 (BetMGM)  

**Top 5 NFL Passing Leaders for Week 3:**

1. 375 yards - Patrick Mahomes  
2. 345 yards - Josh Allen  
3. 320 yards - Justin Herbert  
4. 310 yards - Joe Burrow  
5. 300 yards - Tua Tagovailoa  

**Top 3 Rushing Touchdowns Leaders this season:**

1. 7 touchdowns - Derrick Henry  
2. 6 touchdowns - Dalvin Cook  
3. 5 touchdowns - Nick Chubb  

</example_response>

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
        print(s.content)
        yield s.content
        answer += str(s.content)

    return answer
