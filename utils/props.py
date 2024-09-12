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

props_metadata = props_metadata = """
GlobalHomeTeamID (bigint) 
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
GlobalTeamID (double precision) - Unique identifier for a team across all leagues/sports
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
GlobalGameID (bigint) - Unique identifier for the game across all leagues/sports
GlobalAwayTeamID (bigint) - Unique identifier for the away team across all leagues/sports
SportsBook (text) - Name of the sportsbook offering the odds Could be ['BetMGM', 'Caesars', 'FanDuel', 'Consensus', 'DraftKings', nan]
BettingMarketType (text) - Could be ['Game Line', 'Player Prop', 'Team Prop', 'Game Prop']
BettingBetType (text) - Could be ['Total Points', 'Spread', 'Moneyline', 'Total Passing Yards',
'Total Rushing Yards', 'Total Receiving Yards',
'To Score First Touchdown', 'To Score a Touchdown',
'To Score a D/ST Touchdown', 'To Score 2+ Touchdowns',
'To Score 2+ D/ST Touchdowns', 'Total Field Goals Scored',
'Total Passing Touchdowns', 'Total Rushing + Receiving TDs',
'Interceptions Thrown', 'Total Fumbles Lost',
'Total Passing + Rushing Yards', 'Total Rushing & Receiving Yards',
'Extra Points Made', 'Total Kicking Points',
'Total Interceptions (DEF/ST)', 'Total Tackles (Solo)',
'Total Assists', 'Total Tackles (Solo & Assists)',
'Total Passing Attempts', 'Total Pass Completions',
'Total Receiving Touchdowns', 'Total Rushing Touchdowns',
'Longest Reception', 'Player To Score Last Touchdown',
'Longest Pass', 'To Score 3+ Touchdowns', 'Total Touchdowns',
'Moneyline (3-Way)', 'Both Teams to Score on Their 1st Drive',
'Both teams to score 1+ TD in each half', 'Total Points Odd/Even',
'Race To 20 Points', 'Race To 15 Points', 'Race To 5 Points',
'To Go To Overtime', 'Both Teams to Scor e 25 Points',
'Race to 10 Points', 'Both Teams to Score 40 Points',
'To Score First and Win', 'Both teams to score 3+ TD in each half',
'Both Teams to Score', 'Both Teams to Score 10 Points',
'Both Teams to Score 30 Points', 'Both Teams to Score 20 Points',
'Both Teams to Score 35 Points', 'Last Team To Score',
'First Team To Score', 'Both teams to score 2+ TD in each half',
'Both teams to score 4+ TD in each half',
'To Score First and Lose', 'Race To 25 Points',
'To Score First Field Goal', 'Both Teams to Score 15 Points',
'Race To 30 Points', 'Longest Rush',
'To Score A Defensive Touchdown',
'To Score 2+ Defensive Touchdowns',
'Team To Score First Touchdown',
'Either Team To Score 3 Unanswered Times',
'A Score In Final Two Minutes', 'To Record A Safety',
'First Team To Call Timeout', 'To Attempt an Onside Kick',
'Punt Returned For Touchdown', 'Punt To Be Blocked',
'Field Goal To Be Blocked', 'Both Teams To Score A Touchdown',
'Both Teams To Score 2+ Touchdowns',
'Both Teams To Score 3+ Touchdowns',
'Both Teams To Score 3+ Points', 'Both Teams To Score 7+ Points',
'First Team to Use Coach Challenge', 'Total Sacks',
'Total Receptions', 'To Record Successful Two Point Conversion',
'To Attempt 2-Point Conversion',
'Total Pass + Rush + Rec Touchdowns',
'Punt Downed Inside The 5-yard line', 'Total Rushing Attempts',
'To Complete First Pass']
BettingPeriodType (text) - Could be ['Full Game', '1st Quarter', '3rd Quarter', '4th Quarter','2nd Quarter', 'First Half', 'Second Half', 'Regular Season']
PlayerName (text) - If it is a player prop this will be the name of the player for player props, format is first name last name, ex: 'Jordan Love'
AwayTeam (text) - Name of the away team in short form, like the San Francisco 49ers are SF
HomeTeam (text) - Name of the home team in short form, like the San Francisco 49ers are SF
Channel (text) - Name of network provider, could be ['PEA', 'NBC', 'FOX', 'CBS', 'ABC', 'ESPN', 'AMZN', 'NFLN', 'NFLX', nan]
QuarterDescription (text) - Description of the current quarter or game state
Day (text) - Day of the week for the game, looks like 2024-09-06T00:00:00. You can use this when you don't know game time. 
DateTime (text) - Datetime of the game, looks like 2024-09-06T20:15:00. You can use this when you know the exact starting game time.
Status (text) - Current status of the game (e.g., scheduled, in progress, final)
DateTimeUTC (text) - Datetime of the game in UTC, looks like 2024-09-07T00:15:00. You can use this when you know the exact starting game time.
BettingOutcomeType (text) - Could be ['Over', 'Under', 'Away', 'Home', nan, 'Yes', 'Draw', 'No', 'Odd',
       'Even', 'Neither']
SportsbookUrl (text) - URL to the sportsbook's page for this game or bet
BetPercentage (double precision) - Percentage of bets on this outcome
MoneyPercentage (double precision) - Percentage of money on this outcome

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
This is the current date: {current_date}
For game days, you can use the Day column, if you don't have the time of the game.

Assistant: 


"""
sql_prompt = PromptTemplate.from_template(prompt_template)


def props_log_get_answer(model, question):
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
    print(sql_prompt)
    answer = llm_chain.invoke(
        {'user_question': question, "table_metadata_string": props_metadata, 'current_date': datetime.now().strftime('%Y-%m-%d-%H-%M-%S')})

    return answer.content
