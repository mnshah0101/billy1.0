import os
import sqlite3
import pandas as pd
from langchain.agents import initialize_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import time
from langchain_anthropic import ChatAnthropic
import re
from datetime import datetime
from utils.cache import get_closest_embedding
from utils.CountUtil import count_tokens
import dotenv
dotenv.load_dotenv()

add_line = os.getenv('ADD_LINE')


props_metadata = """
PointSpreadAwayTeamMoneyLine (bigint)
PointSpreadHomeTeamMoneyLine (bigint) 
ScoreID (bigint) - Unique identifier for the game score
Week (bigint) - The week number of the game in the season
OverPayout (bigint) - Payout for betting over the total points
UnderPayout (bigint) - Payout for betting under the total points
PlayerID (double precision) - Unique identifier for a player
BettingOutcomeID (double precision) - Unique identifier for a specific betting outcome
BettingEventID (bigint) - Unique identifier for a betting event
PayoutAmerican (double precision) - Payout in American odds format
Value (double precision) - The betting line or total for props
TeamID (double precision) - Unique identifier for a team
BettingPeriodTypeID (bigint) - Identifier for the betting period (e.g., full game, first half)
BettingMarketID (bigint) - Unique identifier for a betting market
PointSpread (double precision) - The point spread for the game
OverUnder (double precision) - The over/under total for the game
GameKey (bigint) - Unique identifier for the game
AwayTeamMoneyLine (bigint) - Money line for betting on the away team to win outright
HomeTeamMoneyLine (bigint) - Money line for betting on the home team to win outright
SeasonType (bigint) - Type of season (e.g., 1 for regular season, 2 for playoffs)
Season (bigint) - The year of the season
AwayTeamID (bigint) - Unique identifier for the away team
HomeTeamID (bigint) - Unique identifier for the home team
SportsBook (text) - Name of the sportsbook offering the odds Could be ['BetMGM', 'Caesars', 'FanDuel', 'Consensus', 'DraftKings', nan]
BettingMarketType (text) - Could be ['Game Line', 'Player Prop', 'Team Prop', 'Game Prop']
BettingBetType (text) - Don't query on this unless the user specifically asks for one of these. Could be ['Total Points', 'Spread', 'Moneyline', 'Total Touchdowns',
       'Race to 10 Points', 'Race To 25 Points', 'Race To 20 Points',
       'Race To 35 Points', 'Race To 15 Points', 'Race To 30 Points',
       'To Go To Overtime', 'To Score In Each Quarter',
       'Team To Lead After Every Quarter', 'Team To Win Every Quarter',
       'Team with Highest Scoring Quarter', 'To Win Both Halves',
       'To Score a Touchdown', 'Moneyline (3-Way)',
       'Total Points Odd/Even', 'Team To Score First Touchdown',
       'To Record A Safety', 'First Team To Call Timeout',
       'First Team To Score', 'Either Team To Score 3 Unanswered Times',
       'Last Team To Score', 'To Attempt an Onside Kick',
       'Punt Returned For Touchdown', 'Punt To Be Blocked',
       'Field Goal To Be Blocked',
       'Team To Score Most Touchdowns (3-Way)',
       'Team To Score Most Touchdowns', 'Total Passing Yards',
       'Both teams to score 1+ TD in each half',
       'Both Teams to Score 10 Points', 'Both Teams to Score 15 Points',
       'To Score First Field Goal', 'Race To 5 Points',
       'Both Teams to Score', 'Both teams to score 2+ TD in each half',
       'Both Teams to Score 20 Points', 'To Score First and Lose',
       'Both Teams to Score 25 Points', 'To Score First and Win',
       'Both Teams to Score 30 Points', 'Total Receiving Yards',
       'Both Teams To Score A Touchdown',
       'Both Teams To Score 2+ Touchdowns',
       'Both Teams To Score 3+ Touchdowns',
       'Both Teams To Score 3+ Points', 'Both Teams To Score 7+ Points',
       '20+ Yard Reception on 1st Drive', 'To Allow a Sack on 1st Drive',
       '20+ Yard Offensive Play on 1st Drive',
       '1st Drive To Cross Midfield',
       'Successful Fourth Down Conversion on 1st Drive',
       '10+ Yard Rush on 1st Drive',
       'To Get a First Down On the 1st Drive',
       'Offensive Score on 1st Drive of the Game',
       'Total Passing Touchdowns', 'To Score 2+ Touchdowns',
       'Player To Score Last Touchdown', 'To Score First Touchdown',
       'To Score 3+ Touchdowns', 'Total Rushing Touchdowns',
       'Total Receiving Touchdowns', 'Longest Reception', 'Longest Pass',
       'Interceptions Thrown', 'Total Field Goals Scored',
       'Extra Points Made', 'Total Kicking Points',
       'Both Teams to Score 35 Points', 'To Score a D/ST Touchdown',
       'First Team to Use Coach Challenge',
       'Either Team TD on their 1st Offensive Play',
       'First Play From Scrimmage To Result in TD',
       'To Score On 1st Offensive Play', 'To Score 2+ D/ST Touchdowns',
       'Total Rushing Yards', 'Longest Rush',
       'To Score A Defensive Touchdown',
       'To Score 2+ Defensive Touchdowns',
       'Both Teams To Complete First Pass Attempt', 'Most Passing Yards',
       'Total Receptions', 'Total Passing Attempts',
       'Total Pass Completions', 'Total Passing + Rushing Yards',
       'Total Rushing & Receiving Yards', 'Total Rushing Attempts',
       'To Throw An Interception', 'Total Net Offensive Yards']
BettingPeriodType (text) - Could be ['Full Game', '1st Quarter', '3rd Quarter', '4th Quarter','2nd Quarter', 'First Half', 'Second Half', 'Regular Season']
PlayerName (text) - If it is a player prop this will be the name of the player for player props, format is first name last name, ex: 'Jordan Love'
AwayTeam (text) - Name of the away team in short form, like the San Francisco 49ers are SF
HomeTeam (text) - Name of the home team in short form, like the San Francisco 49ers are SF
Channel (text) - Name of network provider, could be ['PEA', 'NBC', 'FOX', 'CBS', 'ABC', 'ESPN', 'AMZN', 'NFLN', 'NFLX', nan]
QuarterDescription (text) - Description of the current quarter or game state
Day (text) - Day of the week for the game, formatted like 2024-09-06T00:00:00. You can use this when you don't know game time. 
DateTime (text) - Datetime of the game, formatted like 2024-09-06T20:15:00. You can use this when you know the exact starting game time.
DateTimeUTC (text) - Datetime of the game in UTC, formatted like 2024-09-07T00:15:00. You can use this when you know the exact starting game time.
BettingOutcomeType (text) - Could be ['Over', 'Under', 'Away', 'Home', nan, 'Yes', 'Draw', 'No', 'Odd',
       'Even', 'Neither']
SportsbookUrl (text) - URL to the sportsbook's page for this game or bet
BetPercentage (double precision) - Percentage of bets on this outcome, a lot of these are NaN, but some are not
MoneyPercentage (double precision) - Percentage of money on this outcome, a lot of these are NaN, but some are not
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
The name of the table is props. 

There will only be a player name if the question is about a player, and a team name will only be non-null if the question is about the team. All props data is for 2024 only.
You must list all the sportsbooks (Draftkings, FanDuel, etc) and corresponding sportsbook urls for all the stats you are providing. 

When asking for multiple props, only provide at most 3 interesting props. When being asked to decide which props are the most popular, use your intuition to decide which ones most seem popular. 
Don't query on BettingBetType unless the user specifically asks for one of these. This is because not all players or games have these types of bets and you are likely to get an empty response.

</special_instructions>


<question>


Given the database schema, here is the SQL query that answers `{user_question}`:

</question>


Here is an example response for the question: "What is the average point spread for home teams in games where the over/under is greater than 45 and the game has already started but is not yet over?"

If the question cannot be answered with the data provided, return the string "cannot be answered".

This is a postgres database. Do not create any new columns or tables. Only use the columns that are in the table.


Here is an example response for the question: {matched_question}
<example_response>


```sql
{matched_sql_query}
```

</example_response>


Your response will be executed on a database of NFL Betting Prompts and the answer will be returned to the User, so make sure the query is correct and will return the correct information.
The default SeasonType is Regular Season or 1. If the question is about a different SeasonType, please specify in the query. The default season is 2024.
Use the Wins and Losses columns to determine the number of wins and losses for a team. They reset each season and each season type. Remember, they are cumulative up to the current game.


If the question cannot be answered with the data provided, please return the string "Error: Cannot answer question with data provided."


Do not use functions that are not available in SQLite. Do not use functions that are not available in SQLite. Do not create new columns, only use what is provided.
Make sure you surround columns with double quotes since it is case sensitive. An example is p."PlayerName". 
This is the current date: {current_date}
For game days, you can use the Day column, if you don't have the time of the game. Make sure your date format is consistent with the data.
This is a postgres database. Do not create any new columns or tables. Only reference columns that are in the database schema provided.
Make sure you use parentheses correctly in your queries as well as commas to make logical sense. For example AND "TeamCoach" = 'Matt LaFleur' OR "OpponentCoach" = 'Matt LaFleur' should be AND ("TeamCoach" = 'Matt LaFleur' OR "OpponentCoach" = 'Matt LaFleur') since the OR should be in parentheses.
"""

prompt_template += add_line
prompt_template += "Assistant: "


sql_prompt = PromptTemplate.from_template(prompt_template)


def props_log_get_answer(model, question):
    llm = None
    input_count = count_tokens(question)
    input_count += count_tokens(prompt_template)
    input_count += count_tokens(props_metadata)





    matched_question, matched_sql_query = get_closest_embedding(question, model="text-embedding-3-large", top_k=1)

    input_count += count_tokens(matched_question)
    input_count += count_tokens(matched_sql_query)


    if model == 'openai':
        try:
            llm = ChatOpenAI(model='gpt-4o', temperature=0.96)
        except Exception as e:
            print("key not given", e)

    elif model == 'anthropic':
        llm = ChatAnthropic(model_name='claude-3-5-sonnet-20240620', temperature=0.5)

    print(llm)
    llm_chain = sql_prompt | llm
    print(sql_prompt)
    answer = llm_chain.invoke(
        {'user_question': question, "table_metadata_string": props_metadata, 'current_date': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), 'matched_question': matched_question, 'matched_sql_query': matched_sql_query})
    output_count = count_tokens(answer.content)

    return answer.content, input_count, output_count
