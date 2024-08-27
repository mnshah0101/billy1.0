

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
prompt_template = ''

# Define the prompt template
prompt_template = """

User:

<instructions>
You are a data analyst for an NFL team and you have been asked to generate a SQL query to answer the following question. You do not have to completely answer the question, just generate the SQL query to answer the question, and the result will be processed. Do your best to answer the question and do not use placeholder information. The question is:
`{user_question}`

</instructions>


<database_schema>
The query will run on a database of NFL Play by Play with the following schema:
{table_metadata_string}
</database_schema>

If the question cannot be answered with the data provided, please return the string "Error: Cannot answer question with data provided." 





<special_instructions>
- Use the table `playbyplay`.
- Columns to use for scoring plays: Season, Week, HomeTeam, AwayTeam, Date, and GameKey.
- Use the `PlayTime` column for filtering all plays by date (format: `YYYY-MM-DDTHH:MM:SS` UTC).
- Identify players by inferring full names from partial mentions (e.g., "Kelce" implies "Travis Kelce").
- Use double quotes for column names (e.g., `"RushingYards"`).
- Avoid using specific columns unless the play is a scoring play: GameKey, SeasonType, ScoringPlayID, Season, Week, AwayTeam, HomeTeam, Date, Sequence_scoring, Team_scoring, Quarter, TimeRemaining, AwayScore, HomeScore, ScoreID.
- Use `DISTINCT` for `PlayID` to handle duplicate plays.
- Calculate percentages using attempted and made columns (e.g., `ExtraPointsMade` / `ExtraPointsAttempted`).
- A way to get WR1 is to look for the player with the most receiving yards in a season for a team.
- The Season, Week, HomeTeam, AwayTeam, Date, and GameKey columns are only available for scoring plays.
- To filter the data for all plays by date, you can use the PlayTime column. The PlayTime column is in the format of 2022-09-11T22:21:49 and is in UTC time.
- Scoring plays only count touchdowns, so for extra points, field goals, and safeties you must use other columns to determine if it is a scoring play.
- Since you don't have GameKey, you can use Team, nd Date to determine the game.


Remember, do not use GameKey, SeasonType, ScoringPlayID, Season, Week, AwayTeam, HomeTeam, Date, Sequence_scoring, Team_scoring, Quarter, TimeRemaining, AwayScore, HomeScore, or ScoreID in your query unless it is a scoring play.

All columns must be surrounded by double quotes, such as
In the slot means the direction is Middle.

If the question cannot be answered with the provided data, return: "Error: Cannot answer question with data provided."

Respond only with the SQL query, enclosed in:

```sql
<SQL_QUERY_HERE>
```


Note: The database is a PostgreSQL database, and the data only goes back to the 2015 season.


</special_instructions>

<question>


Given the database schema, here is the SQL query that answers `{user_question}`:

</question>


<example_response>

```sql
SELECT SUM("RushingYards") AS Yards
FROM playbyplay
WHERE "Season" = 2023 AND "Name" = 'Patrick Mahomes'
```

</example_response>




Your response will be executed on a database of NFL Play by Play logs and the answer will be returned to the User, so make sure the query is correct and will return the correct information.
Also, keep in mind that there are duplicate plays in the database, so you may need to use DISTINCT with the PlayID to get the correct answer. 

Use the attempted and made columns to calculate percentages. For example, if you want to calculate the extra point conversion percentage, you would use ExtraPointsMade and ExtraPointsAttempted.


If the question cannot be answered with the data provided, please return the string "Error: Cannot answer question with data provided." 

This is a postgres database. Do not create any new columns or tables. Only use the columns that are in the table.

Assistant: 

"""


sql_prompt = PromptTemplate.from_template(prompt_template)


testnfl_metadata = """
Columns in table 'playbyplay':
PlayID (bigint)
QuarterName (text) - 1, 2, 3, 4, OT
Sequence (bigint)
TimeRemainingMinutes (double precision)
TimeRemainingSeconds (double precision)
PlayTime (text)
Updated (text)
Created (text)
Team (text)
Opponent (text)
Down (bigint)
Distance (bigint)
YardLine (bigint)
YardLineTerritory (text)
YardsToEndZone (bigint)
Type (text) - The Type of Play that occurred (possible values: Rush, PassCompleted, PassIncomplete, PassIntercepted, TwoPointConversion, Punt, Kickoff, FieldGoal, ExtraPoint, Fumble, Penalty, Sack, Timeout, Period)
YardsGained (bigint)
IsScoringPlay (bigint) - Only counts for touchdowns.
PlayStatID (double precision)
PlayID_playstats (double precision)
Sequence_playstats (double precision)
PlayerID (double precision)
Name (text)
Team_playstats (text)
Opponent_playstats (text)
HomeOrAway (text)  - HOME or AWAY
Direction (text) - The direction of the play (possible values: Left, Middle, Right)
Updated_playstats (text)
Created_playstats (text)
PassingAttempts (double precision)
PassingCompletions (double precision)
PassingYards (double precision)
PassingTouchdowns (double precision)
PassingInterceptions (double precision)
PassingSacks (double precision)
PassingSackYards (double precision)
RushingAttempts (double precision)
RushingYards (double precision)
RushingTouchdowns (double precision)
ReceivingTargets (double precision)
Receptions (double precision)
ReceivingYards (double precision)
ReceivingTouchdowns (double precision)
Fumbles (double precision)
FumblesLost (double precision)
TwoPointConversionAttempts (double precision) - The number of times a player attempted a two point conversion
TwoPointConversionPasses (double precision) - The number of times a player passed for a two point conversion
TwoPointConversionRuns (double precision) - The number of times a player ran for a two point conversion
TwoPointConversionReceptions (double precision) - The number of times a player caught a two point conversion
TwoPointConversionReturns (double precision) - The number of times a player returned a two point conversion
SoloTackles (double precision)
AssistedTackles (double precision)
TacklesForLoss (double precision)
Sacks (double precision)
SackYards (double precision)
PassesDefended (double precision)
Safeties (double precision)
FumblesForced (double precision)
FumblesRecovered (double precision)
FumbleReturnYards (double precision)
FumbleReturnTouchdowns (double precision)
Interceptions (double precision)
InterceptionReturnYards (double precision)
InterceptionReturnTouchdowns (double precision)
PuntReturns (double precision)
PuntReturnYards (double precision)
PuntReturnTouchdowns (double precision)
KickReturns (double precision)
KickReturnYards (double precision)
KickReturnTouchdowns (double precision)
BlockedKicks (double precision)
BlockedKickReturns (double precision)
BlockedKickReturnYards (double precision)
BlockedKickReturnTouchdowns (double precision)
FieldGoalReturns (double precision)
FieldGoalReturnYards (double precision)
FieldGoalReturnTouchdowns (double precision)
Kickoffs (double precision)
KickoffYards (double precision)
KickoffTouchbacks (double precision)
Punts (double precision)
PuntYards (double precision)
PuntTouchbacks (double precision)
PuntsHadBlocked (double precision)
FieldGoalsAttempted (double precision)
FieldGoalsMade (double precision)
FieldGoalsYards (double precision)
FieldGoalsHadBlocked (double precision)
ExtraPointsAttempted (double precision)
ExtraPointsMade (double precision)
ExtraPointsHadBlocked (double precision)
Penalties (double precision)
PenaltyYards (double precision)
GameKey (double precision) - If this is a scoring play, this is the GameKey of the game
SeasonType (double precision) - If this is a scoring play, this is the SeasonType of the game
ScoringPlayID (double precision) - If this is a scoring play, this is the PlayID of the scoring play
Season (double precision) - If this is a scoring play, this is Season of the game
Week (double precision) - If this is a scoring play, this is the Week of the game
AwayTeam (text) - If this is a scoring play, this is the AwayTeam of the game
HomeTeam (text) - If this is a scoring play, this is the HomeTeam of the game
Date (text) - If this is a scoring play, this is the Date of the game
Sequence_scoring (double precision) - The order in which the scoring play happened
Team_scoring (text) - If this is a scoring play, the Team that scored
Quarter (text) - If this is a scoring play, the Quarter in which the scoring play happened
TimeRemaining (text) - If this is a scoring play, the Time Remaining in the Quarter when the scoring play happened
AwayScore (double precision) - If this is a scoring play, the AwayScore (double precision)
HomeScore (double precision) - If this is a scoring play, the HomeScore (double precision)
ScoreID (double precision) - If this is a scoring play, the ScoreID (double precision)
"""


def play_by_play_get_answer(model, question):
    llm = None
    if model == 'openai':
        llm = ChatOpenAI(model='gpt-4o', temperature=0.9)

    elif model == 'anthropic':
        llm = ChatAnthropic(model_name='claude-3-5-sonnet-20240620')

    llm_chain = sql_prompt | llm
    answer = llm_chain.invoke(
        {'user_question': question, "table_metadata_string": testnfl_metadata})

    return answer.content

