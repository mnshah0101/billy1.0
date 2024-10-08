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
import datetime
from utils.cache import get_closest_embedding
from utils.CountUtil import count_tokens
import dotenv
dotenv.load_dotenv()

add_line = os.getenv('ADD_LINE')

prompt_template = """

User:

<instructions>
You are a data analyst for an NFL team and you have been asked to generate a SQL query to answer the following question. You do not have to completely answer the question, just generate the SQL query to answer the question, and the result will be processed. Do your best to answer the question and do not use placeholder information. The question is:
`{user_question}`

</instructions>


<database_schema>
The query will run on a table of Player Game Logs with the following schema:
{player_log_table_metadata_string}
and a table of Prop data with the following schema:
{props_table_metadata_string}
</database_schema>




<special_instructions_player_log>
The Team is always short hand, such as WAS for Washington or BAL for Baltimore.
The name of one of the tables is playerlog and the other is props. 
Instead of HomeTeam and AwayTeam, reference the Team column and the HomeOrAway Column, The Opponent column will have the opposite side.
You will have to infer player names from little data from your understanding of the NFL. For example, if the user only says Kelce, you have to infer the name Travis Kelce
To find games where two players have played against each other, you can join the table on the GameKey where the Name matches the player.
To calculate "Against the Spread" (ATS), you need to determine whether a team has covered the point spread in a game. If the team is a favorite, they have a negative point spread, and if the team is an underdog, they have a positive point spread. The formula for ATS can be derived using the team score, opponent score, and point spread as follows:

Formula:
Calculate the Cover Margin:
Cover Margin=(Score+PointSpread)-OpponentScore
Determine ATS Result:

If Cover Margin > 0, the team covered the spread.
If Cover Margin < 0, the team did not cover the spread.
If Cover Margin = 0, it is a push (no winner against the spread).


You can use MIN(GameKey) to get the earliest game and MAX(GameKey) to get the latest game.

Remember, rookies have a value of 2 in the Experience column.

A player is injured if the InjuryStatus is Doubtful, Out, or Questionable.

Make sure to use the DISTINCT keyword when necessary to avoid duplicate data.

Usually, even when a player is out or injured, they will have a record in the database. However, sometimes, they might not have a record. Therefore to see how many games a player missed, you can use 17 (or whatever number) - COUNT(DISTINCT GameKey where the player played).

Be careful of periods in the player name. For example, TJ Watt is T.J. Watt in the database.

To see how many games a played missed in the regular season, you can use 17 - COUNT(DISTINCT GameKey where the player played).
Use this logic


Only respond with the sql query, no explanation or anything else. Encompass the sql query with 
```sql

```

All columns must be surrounded by double quotes, such as "Name" or "Team".

There is no weather column, so use a combination of temperature, humidity, and wind speed to determine the weather conditions of the game.

Never include the preseason in any of your responses, and make sure to include all the types of seasons that are provided in the response (regular season, postseason, or regular season and postseason). Also, try and name all the games relevant to the question. This is important.

If asked for ranking, make sure you rank everyone in that position by the criteria given and then output the rank of the player for that criteria. This is important.

You can never not include the player name in the SQL query - doing so would be catastrophic.
When asking about a player, assume that we want logs where the player has played, unless the question specifies otherwise like for injuries or missed games.

</special_instructions_player_log>



<special_instructions_props> 
The name of the table is props. 

There will only be a player name if the question is about a player, and a team name will only be non-null if the question is about the team. All props data is for 2024 only.
You must list all the sportsbooks (Draftkings, FanDuel, etc) and corresponding sportsbook urls for all the stats you are providing. 
Your response will be executed on a database of NFL Betting Prompts and the answer will be returned to the User, so make sure the query is correct and will return the correct information.
The default SeasonType is Regular Season or 1. If the question is about a different SeasonType, please specify in the query. The default season is 2024.
To calculate record, use Wins and Losses, and you're going to have to add the most recent game to the Wins and Losses columns to get the current record, as the Wins and Losses columns are cumulative up to the current game for that season and season type.

</special_instructions_props>

<question>


Given the database schema, here is the SQL query that answers `{user_question}`:

</question>


Here is an example response for the question {match_question}:

<example_response>

```sql
{matched_sql_query}
```

</example_response>


Your response will be executed on a database of NFL Player Logs and the answer will be returned to the User, so make sure the query is correct and will return the correct information.
You may have to use the "like" operator to match player names, as the user may not provide the full name of the player or the database may have a different format for the player name.

If the question cannot be answered with the data provided, please return the string "Error: Cannot answer question with data provided."

Do not use functions that are not available in SQLite. Do not use functions that are not available in SQLite. Do not create new columns, only use what is provided.
Make sure you surround columns with double quotes since it is case sensitive. An example is p."PlayerName". 
This is the current date: {current_date}
For game days, you can use the Day column, if you don't have the time of the game. Make sure your date format is consistent with the data.
This is a postgres database. Do not create any new columns or tables. Only reference columns that are in the database schema provided.
Make sure you use parentheses correctly in your queries as well as commas to make logical sense. For example AND "TeamCoach" = 'Matt LaFleur' OR "OpponentCoach" = 'Matt LaFleur' should be AND ("TeamCoach" = 'Matt LaFleur' OR "OpponentCoach" = 'Matt LaFleur') since the OR should be in parentheses.
"""

prompt_template += add_line
prompt_template +=  "Assistant: "


sql_prompt = PromptTemplate.from_template(prompt_template)


testnfl_metadata = """
GameKey (bigint)
PlayerID (bigint)
SeasonType (bigint) - (1=Regular Season, 2=Preseason, 3=Postseason, 4=Offseason, 5=AllStar).
Season (bigint)
GameDate (text)
Week (bigint) - The week resets for each season type. So the first week of the regular season is 1, the first week of the preseason is 1, etc.
Team (text)
Opponent (text)
HomeOrAway (text) - HOME or AWAY
Number (bigint)
Name (text) - First Name and Last Name
Position (text) - Player's position for this particular game or season. Possible values: C, CB, DB, DE, DE/LB, DL, DT, FB, FS, G, ILB, K, KR, LB, LS, NT, OL, OLB, OT, P, QB, RB, S, SS, T, TE, WR
PositionCategory (text) - Abbreviation of either Offense, Defense or Special Teams (OFF, DEF, ST)
Activated (bigint)
Played (bigint) - 1 if player has atleast one play, 0 otherwise
Started (bigint) - 1 is player has started
PassingAttempts (double precision)
PassingCompletions (double precision)
PassingYards (double precision)
PassingCompletionPercentage (double precision)
PassingYardsPerAttempt (double precision)
PassingYardsPerCompletion (double precision)
PassingTouchdowns (double precision)
PassingInterceptions (double precision)
PassingRating (double precision)
PassingLong (double precision)
PassingSacks (double precision)
PassingSackYards (double precision)
RushingAttempts (double precision)
RushingYards (double precision)
RushingYardsPerAttempt (double precision)
RushingTouchdowns (double precision)
RushingLong (double precision)
ReceivingTargets (double precision)
Receptions (double precision)
ReceivingYards (double precision)
ReceivingYardsPerReception (double precision)
ReceivingTouchdowns (double precision)
ReceivingLong (double precision)
Fumbles (double precision)
FumblesLost (double precision)
PuntReturns (double precision)
PuntReturnYards (double precision)
PuntReturnYardsPerAttempt (double precision)
PuntReturnTouchdowns (double precision)
PuntReturnLong (double precision)
KickReturns (double precision)
KickReturnYards (double precision)
KickReturnYardsPerAttempt (double precision)
KickReturnTouchdowns (double precision)
KickReturnLong (double precision)
SoloTackles (double precision)
AssistedTackles (double precision)
TacklesForLoss (double precision)
Sacks (double precision)
SackYards (double precision)
QuarterbackHits (double precision)
PassesDefended (double precision)
FumblesForced (double precision)
FumblesRecovered (double precision)
FumbleReturnYards (double precision)
FumbleReturnTouchdowns (double precision)
Interceptions (double precision)
InterceptionReturnYards (double precision)
InterceptionReturnTouchdowns (double precision)
BlockedKicks (double precision)
SpecialTeamsSoloTackles (double precision)
SpecialTeamsAssistedTackles (double precision)
MiscSoloTackles (double precision)
MiscAssistedTackles (double precision)
Punts (double precision)
PuntYards (double precision)
PuntAverage (double precision)
FieldGoalsAttempted (double precision)
FieldGoalsMade (double precision)
FieldGoalsLongestMade (double precision)
ExtraPointsMade (double precision)
TwoPointConversionPasses (double precision)
TwoPointConversionRuns (double precision)
TwoPointConversionReceptions (double precision)
FantasyPoints (double precision)
FantasyPointsPPR (double precision)
ReceptionPercentage (double precision)
ReceivingYardsPerTarget (double precision)
Tackles (bigint)
OffensiveTouchdowns (bigint)
DefensiveTouchdowns (bigint)
SpecialTeamsTouchdowns (bigint)
Touchdowns (bigint)
FantasyPosition (text)
FieldGoalPercentage (double precision)
PlayerGameID (bigint)
FumblesOwnRecoveries (double precision)
FumblesOutOfBounds (double precision)
KickReturnFairCatches (double precision)
PuntReturnFairCatches (double precision)
PuntTouchbacks (double precision)
PuntInside20 (double precision)
PuntNetAverage (bigint)
ExtraPointsAttempted (double precision)
BlockedKickReturnTouchdowns (double precision)
FieldGoalReturnTouchdowns (double precision)
Safeties (double precision)
FieldGoalsHadBlocked (double precision)
PuntsHadBlocked (double precision)
ExtraPointsHadBlocked (double precision)
PuntLong (double precision)
BlockedKickReturnYards (double precision)
FieldGoalReturnYards (double precision)
PuntNetYards (double precision)
SpecialTeamsFumblesForced (double precision)
SpecialTeamsFumblesRecovered (double precision)
MiscFumblesForced (double precision)
MiscFumblesRecovered (double precision)
ShortName (text)
PlayingSurface (text) - Artificial or Grass
SafetiesAllowed (double precision)
Stadium (text)
Temperature (double precision)
Humidity (double precision)
WindSpeed (double precision)
FanDuelSalary (double precision)
DraftKingsSalary (double precision)
FantasyDataSalary (double precision)
OffensiveSnapsPlayed (double precision)
DefensiveSnapsPlayed (double precision)
SpecialTeamsSnapsPlayed (double precision)
OffensiveTeamSnaps (double precision)
DefensiveTeamSnaps (double precision)
SpecialTeamsTeamSnaps (double precision)
VictivSalary (double precision)
TwoPointConversionReturns (double precision)
FantasyPointsFanDuel (double precision)
FieldGoalsMade0to19 (double precision)
FieldGoalsMade20to29 (double precision)
FieldGoalsMade30to39 (double precision)
FieldGoalsMade40to49 (double precision)
FieldGoalsMade50Plus (double precision)
FantasyPointsDraftKings (double precision)
YahooSalary (double precision)
FantasyPointsYahoo (double precision)
InjuryStatus (text) - [None, 'Questionable', 'Probable', 'Out', 'Doubtful']
InjuryBodyPart (text)
FanDuelPosition (text)
DraftKingsPosition (text)
YahooPosition (text)
OpponentRank (double precision)
OpponentPositionRank (double precision)
InjuryPractice (double precision)
InjuryPracticeDescription (double precision)
DeclaredInactive (bigint) - If the player is retired or still playing.
FantasyDraftSalary (double precision)
FantasyDraftPosition (double precision)
TeamID (bigint)
OpponentID (bigint)
Day (TEXT): This looks like 2024-10-03T00:00:00, and can be used when you don't know the exact game time. You can extract the day of the week from this, and use it to determine the game day.
DateTime (text)
GlobalGameID (bigint)
GlobalTeamID (bigint)
GlobalOpponentID (bigint)
ScoreID (bigint)
FantasyPointsFantasyDraft (double precision)
OffensiveFumbleRecoveryTouchdowns (double precision)
SnapCountsConfirmed (bigint)
Updated (text)
source (bigint))
Wins (double precision) - This is the number of wins the team had in the season up to this point
OpponentWins (double precision) - This is the number of wins the opponent had in the season up to this point
Losses (double precision)  - This is the number of losses the team had in the season up to this point
OpponentLosses (double precision)  - This is the number of losses the opponent had in the season up to this point
PointSpread (double precision) - This is the point spread of the game.
Score (double precision) - This is the score of the team
OpponentScore (double precision) - This is the score of the opponent
Status (text) - Active or Inactive
Height (text) - Height in feet and inches like 6'0"
BirthDate (text) - The birthdate of the player like 1999-08-31T00:00:00
Weight (double precision) - The weight of the player in pounds
College (text) - The college the player attended
Experience (double precision) - The number of years the player has played in the NFL. Since it is updated every spring, rookies in the 2024 season have a value of 2.
"""

props_metadata = """
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
Day (text) - Day of the week for the game, formatted like 2024-09-06T00:00:00. You can use this when you don't know game time. 
DateTime (text) - Datetime of the game, formatted like 2024-09-06T20:15:00. You can use this when you know the exact starting game time.
DateTimeUTC (text) - Datetime of the game in UTC, formatted like 2024-09-07T00:15:00. You can use this when you know the exact starting game time.
BettingOutcomeType (text) - Could be ['Over', 'Under', 'Away', 'Home', nan, 'Yes', 'Draw', 'No', 'Odd',
       'Even', 'Neither']
SportsbookUrl (text) - URL to the sportsbook's page for this game or bet
BetPercentage (double precision) - Percentage of bets on this outcome, a lot of these are NaN, but some are not
MoneyPercentage (double precision) - Percentage of money on this outcome, a lot of these are NaN, but some are not

"""


def player_log_and_props_get_answer(model, question):
    llm = None
    input_count = count_tokens(question)
    input_count += count_tokens(prompt_template)
    input_count += count_tokens(testnfl_metadata)
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
    answer = llm_chain.invoke(
        {'user_question': question, "player_log_table_metadata_string": testnfl_metadata, "props_table_metadata_string": props_metadata, "current_date": str(datetime.datetime.today()).split()[0], "match_question": matched_question, "matched_sql_query": matched_sql_query})
    return_answer = answer.content
    output_count = count_tokens(return_answer)
    return return_answer, input_count, output_count
