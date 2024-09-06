import os
import sqlite3
import pandas as pd
from langchain.agents import initialize_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import time
from langchain_anthropic import ChatAnthropic
import re

futures_metadata = """
PlayerID (double precision) - If this is a player future, this is the unique identifier for the player
BettingOutcomeID (bigint)
BettingMarketID (bigint)
PayoutAmerican (bigint) - This is the payout in American odds format
GlobalTeamID (double precision) - This is the unique identifier for a team if it is a team future
BettingEventID (bigint) 
BettingOutcomeType (text) - Can be [nan, 'No', 'Yes', 'Under', 'Over']
SportsbookUrl (text) - This is the url for the sportsbook
SportsBook (text) - This can be ['FanDuel', 'Caesars', 'Consensus', 'BetMGM', 'Fanatics', 'DraftKings']
BettingMarketType (text) - Can be ['Team Future', 'Player Future', 'Coach Future',
       'Miscellaneous Future']
BettingBetType (text)- ['NFL Championship Winner', 'AFC Champion', 'NFC Champion', 'MVP',
       'To Make the Playoffs', 'Win Total', 'Coach of the Year',
       'Offensive Player of the Year', 'Defensive Player of the Year',
       'AFC South Division Winner', 'AFC West Division Winner',
       'NFC East Division Winner', 'NFC North Division Winner',
       'AFC East Division Winner', 'NFC South Division Winner',
       'NFC West Division Winner', 'AFC North Division Winner',
       'Total Receiving Yards', 'Total Receiving Touchdowns',
       'Total Rushing Yards', 'Total Rushing Touchdowns',
       'AFC East Second Place', 'AFC East Third Place',
       'AFC East Fourth Place', 'AFC North Fourth Place',
       'AFC North Second Place', 'AFC North Third Place',
       'AFC South Second Place', 'AFC South Third Place',
       'AFC South Fourth Place', 'AFC West Top 2 Finish',
       'AFC West Second Place', 'AFC West Third Place',
       'AFC West Fourth Place', 'NFC East Second Place',
       'NFC East Third Place', 'NFC East Fourth Place',
       'NFC East Top 2 Finish', 'AFC East Top 2 Finish',
       'NFC North Second Place', 'NFC North Third Place',
       'NFC North Fourth Place', 'NFC South Second Place',
       'NFC South Third Place', 'NFC South Fourth Place',
       'NFC West Third Place', 'NFC West Fourth Place',
       'NFC West Top 2 Finish', 'NFC West Second Place',
       'Total Passing Yards', 'Total Passing Touchdowns', 'Total Sacks',
       'Most Passing Yards', 'Offensive Rookie of the Year',
       'Defensive Rookie of the Year', 'Any Team To Go 0-17',
       'Any Team To Go 17-0', 'Most Rookie Passing Yards',
       'Most Rookie Receiving Yards', 'Most Passing Touchdowns',
       'Most Receiving Yards', 'Most Rushing Yards',
       'Comeback Player of the Year', 'Best Record',
       'Lowest Scoring Team', 'Highest Scoring Team', 'AFC #1 Seed',
       'NFC #1 Seed', 'Last Winless Team', 'Last Team to Lose',
       'Team To Start 5-0', 'Team To Start 0-5',
       'Any Game to Finish in a Tie', 'To Win All 6 Division Games',
       'To Win All Home Games', 'To Win All Away Games',
       'To Lose All 6 Division Games', 'To Concede Most Points',
       'To Concede Least Points', 'To Lose All Home Games',
       'To Lose All Road Games', 'Most Rushing Touchdowns',
       'Most Receiving Touchdowns', 'Total Interceptions (DEF/ST)',
       'Least Wins', 'Most Wins', 'Team to Go 20-0 and Win Super Bowl',
       'Team to Go 17-0', 'Team to Go 0-17',
       'Most Tackles Leader (Solo & Assists)',
       'Most Interceptions Thrown', 'Sack Leader', 'Total Points',
       'Total Division Wins', 'Worst Record', 'Receptions Leader',
       'Most Quarterback Rushing Yards', 'NFC Wildcard Team',
       'AFC Wildcard Team', 'To Have 750+ Receiving Yards',
       'To Have 1250+ Receiving Yards', 'To Have 1000+ Receiving Yards',
       'Highest Rushing Yards Total', 'Longest Field Goal Made',
       'Highest Passing Yards Total', 'Highest Passing TD Total',
       'Highest Interceptions Thrown Total',
       'Highest Individual Receptions Total',
       'Highest Individual Sack Total',
       'Highest Individual FG Made Total', 'Highest Rushing TD Total',
       'Longest Reception', 'Longest Rush', 'Highest Receiving TD Total',
       'Highest Individual Passing Yards Game',
       'Highest Individual Defensive Interception Total',
       'Total Games To Go To Overtime',
       'Most Receiving Yards in Any Game',
       'Most Rushing Yards in Any Game', 'Total Receptions',
       'To Have 750+ Rushing Yards', 'To Have 1000+ Rushing Yards',
       'To Have 1250+ Rushing Yards',
       'Team To Score 1+ Touchdown in Every Game',
       'Most Kickoff Return Touchdowns', 'Interceptions Thrown',
       'Total Yards of Longest Touchdown', 'To Throw 35+ Touchdowns',
       'To Throw 30+ Touchdowns', 'To Have 10+ Receiving Touchdowns',
       'To Have 12+ Receiving Touchdowns', 'To Throw 40+ Touchdowns',
       'To Score 10+ Rushing Touchdowns',
       'To Have 6+ Receiving Touchdowns',
       'To Have 8+ Receiving Touchdowns', 'Most Rookie Rushing Yards',
       'Total Individual 200+ Receiving Yard Games',
       'Highest Receiving Yards Total', 'Assistant Coach of the Year',
       'To Be Named AP First Team All-Pro DE',
       'To Be Named AP First Team All-Pro DL',
       'To Be Named AP First Team All-Pro LB',
       'To Be Named AP First Team All-Pro TE',
       'To Be Named AP First Team All-Pro LG',
       'To Be Named AP First Team All-Pro LT',
       'To Be Named AP First Team All-Pro RB',
       'To Be Named AP First Team All-Pro RG',
       'To Be Named AP First Team All-Pro C',
       'To Be Named AP First Team All-Pro RT',
       'To Be Named AP First Team All-Pro CB',
       'To Be Named AP First Team All-Pro WR',
       'To Be Named AP First Team All-Pro QB',
       'To Be Named AP First Team All-Pro S',
       'To Be Named AP First Team All-Pro K',
       'To Be Named AP First Team All-Pro P']
BettingPeriodType (text) -Can be['NFL Championship Game', 'Regular Season - Including Playoffs',
       'Regular Season']
TeamKey (text) -  If it a team future, this will have the short form of the team, for example the 49ers are SF
PlayerName (text) - If it is a player future, this will be the name of the player
Created (text) - Timestamp of when the record was created, looks like  looks like 2024-09-07T00:15:00
Updated (text) - Timestamp of when the record was last updated, looks like  looks like 2024-09-07T00:15:00

"""


prompt_template = """

<instructions>
You are a data analyst for an NFL team and you have been asked to generate a SQL query to answer the following question. You do not have to completely answer the question, just generate the SQL query to answer the question, and the result will be processed. Do your best to answer the question and do not use placeholder information. The question is:
`{user_question}`

</instructions>


<database_schema>
The query will run on a table of Betting Prop outcomes with the following schema:
{table_metadata_string}
</database_schema>


<special_instructions> 
The name of the table is futurestable. 

There will only be a player name if the question is about a player, and a team name will only be non-null if the question is about the team. All futures data is for 2024 only.
You must list all the sportsbooks (Draftkings, FanDuel, etc) and corresponding sportsbook urls for all the stats you are providing.
Do not make things more specific than they should be. For example, no need to give the restriction of BettingMarketType if it is not needed.
</special_instructions>


<question>


Given the database schema, here is the SQL query that answers `{user_question}`:

</question>


Here is an example response for the question: "What is the average point spread for home teams in games where the over/under is greater than 45 and the game has already started but is not yet over?"

If the question cannot be answered with the data provided, return the string "cannot be answered".

This is a postgres database. Do not create any new columns or tables. Only use the columns that are in the table.

<example_response>


```sql
SELECT AVG(PointSpread) AS avg_home_point_spread
FROM your_table_name
WHERE OverUnder > 45
  AND HasStarted = TRUE
  AND IsOver = FALSE
  AND PointSpread IS NOT NULL;
```

</example_response>


Your response will be executed on a database of NFL Betting Prompts and the answer will be returned to the User, so make sure the query is correct and will return the correct information.
The default SeasonType is Regular Season or 1. If the question is about a different SeasonType, please specify in the query. The default season is 2024.
Use the Wins and Losses columns to determine the number of wins and losses for a team. They reset each season and each season type. Remember, they are cumulative up to the current game.


If the question cannot be answered with the data provided, please return the string "Error: Cannot answer question with data provided."


Do not use functions that are not available in SQLite. Do not use functions that are not available in SQLite. Do not create new columns, only use what is provided.
Make sure you surround columns with double quotes since it is case sensitive. An example is p."PlayerName". 


Assistant: 


"""
sql_prompt = PromptTemplate.from_template(prompt_template)


def futures_log_get_answer(model, question):
    llm = None
    if model == 'openai':
        try:
            llm = ChatOpenAI(model='gpt-4o', temperature=0.96)
        except Exception as e:
            print("key not given", e)

    elif model == 'anthropic':
        llm = ChatAnthropic(model_name='claude-3-5-sonnet-20240620')

    print(llm)
    llm_chain = sql_prompt | llm
    answer = llm_chain.invoke(
        {'user_question': question, "table_metadata_string": futures_metadata})

    return answer.content
