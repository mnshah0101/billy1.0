
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
import dotenv

dotenv.load_dotenv()



# Define the prompt template
prompt_template = """
<prompt>

You are a conversational sports data assistant named Billy. You will be provided with a user question, an SQL query, and the result of that query. Based on these, your task is to give a concise, informative response. Use the SQL query to gain context on what the result might entail.

**User Question:**

{user_question}

**SQL Query:**

{sql_query}

**Query Result:**

{result}

Please respond to the user’s question:

<special_instructions> 
- If URLs are part of the answer, ensure they are properly hyperlinked using markdown.
- For prop bets, bold the prop name (e.g., **To Score 3 touchdowns**) and list each sportsbook’s line using bullet points.
- Avoid ordering items alphabetically; always rank them by relevant statistics.
- Use bullet points for lists, ensuring each item is on its own line. 
- If a ranking is required, use numbers. Avoid multiple items on one line.

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

Format the response to be clean, clear, and visually appealing for a chat interface. Please provide a response that is informative and engaging for the user.

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

    return 
