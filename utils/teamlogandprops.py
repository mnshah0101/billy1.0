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

add_line  = os.getenv("ADD_LINE")

prompt_template = """

User:

<instructions>
You are a data analyst for an NFL team and you have been asked to generate a SQL query to answer the following question. You do not have to completely answer the question, just generate the SQL query to answer the question, and the result will be processed. Do your best to answer the question and do not use placeholder information. The question is:
`{user_question}`

</instructions>


<database_schema>
The query will run on a table of Team Game Logs with the following schema:
{team_log_table_metadata_string}
and a table of Prop data with the following schema:
{props_table_metadata_string}
</database_schema>




<special_instructions_team_logs>
The team is always short hand, such as WAS for Washington or BAL for Baltimore.
The name of the table is teamlog. 
Instead of HomeTeam and AwayTeam, reference the Team column and the HomeOrAway Column, The Opponent column will have the opposite side.
To calculate "Against the Spread" (ATS), you need to determine whether a team has covered the point spread in a game. The formula for ATS can be derived using the team score, opponent score, and point spread as follows:

Formula:
Calculate the Cover Margin:
Cover Margin=(Score+PointSpread)-OpponentScore
Determine ATS Result:

If Cover Margin > 0, the team covered the spread.
If Cover Margin < 0, the team did not cover the spread.
If Cover Margin = 0, it is a push (no winner against the spread).

A negative point spread means the team is favored to win, and a positive point spread means the team is the underdog.


Only respond with the sql query, no explanation or anything else. Encompass the sql query with 
```sql

```


A clever way to get the last game of a team is to do MAX(GameKey), which will give you the last game of the team. 


All columns must be surrounded by double quotes, such as "Name" or "Team".

There is no weather column, so use a combination of temperature, humidity, and wind speed to determine the weather conditions of the game.
To calculate record, use WinsAfter for record after the game and Wins for record before the game. The same goes for losses.

The games are doubled counted in the TeamLog, so you will have to use DISTINCT to get the unique games for a team. They are double counted in that in one occurrence the home team is the Team and away the Opponent and in the other occurrence the away team is the Team and the home team is the Opponent. You can do this with SELECT DISTINCT ON ("GameKey")

The team in the Team column isn't always the home team, it could be the away team, so use HomeOrAway to determine if the team is the home team or the away team. This is very important for determining who is what team in the game.


</special_instructions_team_logs>

<special_instructions_props> 
The name of the table is props. 

There will only be a player name if the question is about a player, and a team name will only be non-null if the question is about the team. All props data is for 2024 only.
You must list all the sportsbooks (Draftkings, FanDuel, etc) and corresponding sportsbook urls for all the stats you are providing. 
Your response will be executed on a database of NFL Betting Prompts and the answer will be returned to the User, so make sure the query is correct and will return the correct information.
The default SeasonType is Regular Season or 1. If the question is about a different SeasonType, please specify in the query. The default season is 2024.
Don't query on BettingBetType unless the user specifically asks for one of these. This is because not all players or games have these types of bets and you are likely to get an empty response.

</special_instructions_props>

<question>


Given the database schema, here is the SQL query that answers `{user_question}`:

</question>


Here is an example response for the question: {matched_question}

<example_response>

```sql
{matched_sql_query}
```
</example_response>


Your response will be executed on a database of NFL Team Logs and the answer will be returned to the User, so make sure the query is correct and will return the correct information.

If the question cannot be answered with the data provided, please return the string "Error: Cannot answer question with data provided."

Do not use functions that are not available in SQLite. Do not use functions that are not available in SQLite. Do not create new columns, only use what is provided.
Make sure you surround columns with double quotes since it is case sensitive. An example is p."PlayerName". 
This is the current date: {current_date}
For game days, you can use the Day column, if you don't have the time of the game. Make sure your date format is consistent with the data.
This is a postgres database. Do not create any new columns or tables. Only reference columns that are in the database schema provided.
Make sure you use parentheses correctly in your queries as well as commas to make logical sense. 
"""

prompt_template += add_line
prompt_template += "Assistant:"


sql_prompt = PromptTemplate.from_template(prompt_template)


testnfl_metadata = """
GameKey (BIGINT): 
Date (TEXT): Format: 'YYYY-MM-DDTHH:MM:SS'. Remember, this is not a Date type, it is a TEXT type.
SeasonType (BIGINT): (1=Regular Season, 2=Preseason, 3=Postseason, 4=Offseason, 5=AllStar). The default season type is 1.
Season (BIGINT): The default season is 2024.
Week (BIGINT): The week resets for each season type. The default week is 1. Week 17 is the last week of the regular season.
Team (TEXT): Short form of the team name (e.g. WAS for Washington, BAL for Baltimore).
Opponent (TEXT): The name of the opponent team in shorthand.
HomeOrAway (TEXT): Could be HOME or AWAY.
Score (BIGINT):
OpponentScore (BIGINT):
TotalScore (BIGINT):
Stadium (TEXT): This is where the game was played. Games in England were played in Wembley Stadium or Tottenham Hotspur Stadium.
PlayingSurface (TEXT): Could be Artificial or Grass.
Temperature (DOUBLE PRECISION):
Humidity (DOUBLE PRECISION):
WindSpeed (DOUBLE PRECISION):
OverUnder (DOUBLE PRECISION): Estimated total points scored in the game. Divide by 2 to get the average points per team.
PointSpread (DOUBLE PRECISION):
ScoreQuarter1 (BIGINT):
ScoreQuarter2 (BIGINT):
ScoreQuarter3 (BIGINT):
ScoreQuarter4 (BIGINT):
ScoreOvertime (BIGINT):
TimeOfPossessionMinutes (BIGINT):
TimeOfPossessionSeconds (BIGINT):
TimeOfPossession (TEXT)
FirstDowns (BIGINT):
FirstDownsByRushing (DOUBLE PRECISION):
FirstDownsByPassing (DOUBLE PRECISION):
FirstDownsByPenalty (DOUBLE PRECISION):
OffensivePlays (BIGINT):
OffensiveYards (BIGINT):
OffensiveYardsPerPlay (DOUBLE PRECISION):
Touchdowns (DOUBLE PRECISION):
RushingAttempts (BIGINT):
RushingYards (BIGINT):
RushingYardsPerAttempt (DOUBLE PRECISION):
RushingTouchdowns (DOUBLE PRECISION):
PassingAttempts (BIGINT):
PassingCompletions (BIGINT):
PassingYards (BIGINT):
PassingTouchdowns (DOUBLE PRECISION):
PassingInterceptions (BIGINT):
PassingYardsPerAttempt (DOUBLE PRECISION):
PassingYardsPerCompletion (DOUBLE PRECISION):
CompletionPercentage (DOUBLE PRECISION):
PasserRating (DOUBLE PRECISION):
ThirdDownAttempts (DOUBLE PRECISION):
ThirdDownConversions (DOUBLE PRECISION):
ThirdDownPercentage (DOUBLE PRECISION):
FourthDownAttempts (DOUBLE PRECISION):
FourthDownConversions (DOUBLE PRECISION):
FourthDownPercentage (DOUBLE PRECISION):
RedZoneAttempts (DOUBLE PRECISION):
RedZoneConversions (DOUBLE PRECISION):
GoalToGoAttempts (DOUBLE PRECISION):
GoalToGoConversions (DOUBLE PRECISION):
ReturnYards (BIGINT):
Penalties (BIGINT):
PenaltyYards (BIGINT):
Fumbles (BIGINT):
FumblesLost (BIGINT):
TimesSacked (BIGINT):
TimesSackedYards (BIGINT):
QuarterbackHits (DOUBLE PRECISION):
TacklesForLoss (DOUBLE PRECISION):
Safeties (DOUBLE PRECISION):
Punts (BIGINT):
PuntYards (BIGINT):
PuntAverage (DOUBLE PRECISION):
Giveaways (BIGINT):
Takeaways (BIGINT):
TurnoverDifferential (BIGINT):
OpponentScoreQuarter1 (BIGINT):
OpponentScoreQuarter2 (BIGINT):
OpponentScoreQuarter3 (BIGINT):
OpponentScoreQuarter4 (BIGINT):
OpponentScoreOvertime (BIGINT):
OpponentTimeOfPossessionMinutes (BIGINT):
OpponentTimeOfPossessionSeconds (BIGINT):
OpponentTimeOfPossession (TEXT)
OpponentFirstDowns (BIGINT):
OpponentFirstDownsByRushing (DOUBLE PRECISION):
OpponentFirstDownsByPassing (DOUBLE PRECISION):
OpponentFirstDownsByPenalty (DOUBLE PRECISION):
OpponentOffensivePlays (BIGINT):
OpponentOffensiveYards (BIGINT):
OpponentOffensiveYardsPerPlay (DOUBLE PRECISION):
OpponentTouchdowns (DOUBLE PRECISION):
OpponentRushingAttempts (BIGINT):
OpponentRushingYards (BIGINT):
OpponentRushingYardsPerAttempt (DOUBLE PRECISION):
OpponentRushingTouchdowns (DOUBLE PRECISION):
OpponentPassingAttempts (BIGINT):
OpponentPassingCompletions (BIGINT):
OpponentPassingYards (BIGINT):
OpponentPassingTouchdowns (DOUBLE PRECISION):
OpponentPassingInterceptions (BIGINT):
OpponentPassingYardsPerAttempt (DOUBLE PRECISION):
OpponentPassingYardsPerCompletion (DOUBLE PRECISION):
OpponentCompletionPercentage (DOUBLE PRECISION):
OpponentPasserRating (DOUBLE PRECISION):
OpponentThirdDownAttempts (DOUBLE PRECISION):
OpponentThirdDownConversions (DOUBLE PRECISION):
OpponentThirdDownPercentage (DOUBLE PRECISION):
OpponentFourthDownAttempts (DOUBLE PRECISION):
OpponentFourthDownConversions (DOUBLE PRECISION):
OpponentFourthDownPercentage (DOUBLE PRECISION):
OpponentRedZoneAttempts (DOUBLE PRECISION):
OpponentRedZoneConversions (DOUBLE PRECISION):
OpponentGoalToGoAttempts (DOUBLE PRECISION):
OpponentGoalToGoConversions (DOUBLE PRECISION):
OpponentReturnYards (BIGINT):
OpponentPenalties (BIGINT):
OpponentPenaltyYards (BIGINT):
OpponentFumbles (BIGINT):
OpponentFumblesLost (BIGINT):
OpponentTimesSacked (BIGINT):
OpponentTimesSackedYards (BIGINT):
OpponentQuarterbackHits (DOUBLE PRECISION):
OpponentTacklesForLoss (DOUBLE PRECISION):
OpponentSafeties (DOUBLE PRECISION):
OpponentPunts (BIGINT):
OpponentPuntYards (BIGINT):
OpponentPuntAverage (DOUBLE PRECISION):
OpponentGiveaways (BIGINT):
OpponentTakeaways (BIGINT):
OpponentTurnoverDifferential (BIGINT):
RedZonePercentage (DOUBLE PRECISION):
GoalToGoPercentage (DOUBLE PRECISION):
QuarterbackHitsDifferential (BIGINT):
TacklesForLossDifferential (BIGINT):
QuarterbackSacksDifferential (BIGINT):
TacklesForLossPercentage (DOUBLE PRECISION):
QuarterbackHitsPercentage (DOUBLE PRECISION):
TimesSackedPercentage (DOUBLE PRECISION):
OpponentRedZonePercentage (DOUBLE PRECISION):
OpponentGoalToGoPercentage (DOUBLE PRECISION):
OpponentQuarterbackHitsDifferential (BIGINT):
OpponentTacklesForLossDifferential (BIGINT):
OpponentQuarterbackSacksDifferential (BIGINT):
OpponentTacklesForLossPercentage (DOUBLE PRECISION):
OpponentQuarterbackHitsPercentage (DOUBLE PRECISION):
OpponentTimesSackedPercentage (DOUBLE PRECISION):
Kickoffs (DOUBLE PRECISION):
KickoffsInEndZone (DOUBLE PRECISION):
KickoffTouchbacks (DOUBLE PRECISION):
PuntsHadBlocked (DOUBLE PRECISION):
PuntNetAverage (DOUBLE PRECISION):
ExtraPointKickingAttempts (DOUBLE PRECISION):
ExtraPointKickingConversions (DOUBLE PRECISION):
ExtraPointsHadBlocked (DOUBLE PRECISION):
ExtraPointPassingAttempts (DOUBLE PRECISION):
ExtraPointPassingConversions (DOUBLE PRECISION):
ExtraPointRushingAttempts (DOUBLE PRECISION):
ExtraPointRushingConversions (DOUBLE PRECISION):
FieldGoalAttempts (DOUBLE PRECISION):
FieldGoalsMade (DOUBLE PRECISION):
FieldGoalsHadBlocked (DOUBLE PRECISION):
PuntReturns (DOUBLE PRECISION):
PuntReturnYards (DOUBLE PRECISION):
KickReturns (DOUBLE PRECISION):
KickReturnYards (DOUBLE PRECISION):
InterceptionReturns (DOUBLE PRECISION):
InterceptionReturnYards (DOUBLE PRECISION):
OpponentKickoffs (DOUBLE PRECISION):
OpponentKickoffsInEndZone (DOUBLE PRECISION):
OpponentKickoffTouchbacks (DOUBLE PRECISION):
OpponentPuntsHadBlocked (DOUBLE PRECISION):
OpponentPuntNetAverage (DOUBLE PRECISION):
OpponentExtraPointKickingAttempts (DOUBLE PRECISION):
OpponentExtraPointKickingConversions (DOUBLE PRECISION):
OpponentExtraPointsHadBlocked (DOUBLE PRECISION):
OpponentExtraPointPassingAttempts (DOUBLE PRECISION):
OpponentExtraPointPassingConversions (DOUBLE PRECISION):
OpponentExtraPointRushingAttempts (DOUBLE PRECISION):
OpponentExtraPointRushingConversions (DOUBLE PRECISION):
OpponentFieldGoalAttempts (DOUBLE PRECISION):
OpponentFieldGoalsMade (DOUBLE PRECISION):
OpponentFieldGoalsHadBlocked (DOUBLE PRECISION):
OpponentPuntReturns (DOUBLE PRECISION):
OpponentPuntReturnYards (DOUBLE PRECISION):
OpponentKickReturns (DOUBLE PRECISION):
OpponentKickReturnYards (DOUBLE PRECISION):
OpponentInterceptionReturns (DOUBLE PRECISION):
OpponentInterceptionReturnYards (DOUBLE PRECISION):
SoloTackles (DOUBLE PRECISION):
AssistedTackles (DOUBLE PRECISION):
Sacks (DOUBLE PRECISION):
SackYards (DOUBLE PRECISION):
PassesDefended (DOUBLE PRECISION):
FumblesForced (DOUBLE PRECISION):
FumblesRecovered (DOUBLE PRECISION):
FumbleReturnYards (DOUBLE PRECISION):
FumbleReturnTouchdowns (DOUBLE PRECISION):
InterceptionReturnTouchdowns (DOUBLE PRECISION):
BlockedKicks (DOUBLE PRECISION):
PuntReturnTouchdowns (DOUBLE PRECISION):
PuntReturnLong (DOUBLE PRECISION):
KickReturnTouchdowns (DOUBLE PRECISION):
KickReturnLong (DOUBLE PRECISION):
BlockedKickReturnYards (DOUBLE PRECISION):
BlockedKickReturnTouchdowns (DOUBLE PRECISION):
FieldGoalReturnYards (DOUBLE PRECISION):
FieldGoalReturnTouchdowns (DOUBLE PRECISION):
PuntNetYards (DOUBLE PRECISION):
OpponentSoloTackles (DOUBLE PRECISION):
OpponentAssistedTackles (DOUBLE PRECISION):
OpponentSacks (DOUBLE PRECISION):
OpponentSackYards (DOUBLE PRECISION):
OpponentPassesDefended (DOUBLE PRECISION):
OpponentFumblesForced (DOUBLE PRECISION):
OpponentFumblesRecovered (DOUBLE PRECISION):
OpponentFumbleReturnYards (DOUBLE PRECISION):
OpponentFumbleReturnTouchdowns (DOUBLE PRECISION):
OpponentInterceptionReturnTouchdowns (DOUBLE PRECISION):
OpponentBlockedKicks (DOUBLE PRECISION):
OpponentPuntReturnTouchdowns (DOUBLE PRECISION):
OpponentPuntReturnLong (DOUBLE PRECISION):
OpponentKickReturnTouchdowns (DOUBLE PRECISION):
OpponentKickReturnLong (DOUBLE PRECISION):
OpponentBlockedKickReturnYards (DOUBLE PRECISION):
OpponentBlockedKickReturnTouchdowns (DOUBLE PRECISION):
OpponentFieldGoalReturnYards (DOUBLE PRECISION):
OpponentFieldGoalReturnTouchdowns (DOUBLE PRECISION):
OpponentPuntNetYards (DOUBLE PRECISION):
TeamName (TEXT): The full name of the team (e.g. New England Patriots)
DayOfWeek (TEXT) - The day of the week this game was played on (e.g. Sunday, Monday)
PassingDropbacks (BIGINT):
OpponentPassingDropbacks (BIGINT):
TwoPointConversionReturns (BIGINT):
OpponentTwoPointConversionReturns (BIGINT):
TeamID (BIGINT):
OpponentID (BIGINT):
Day (TEXT): This looks like 2024-10-03T00:00:00, and can be used when you don't know the exact game time. You can extract the day of the week from this, and use it to determine the game day.
DateTime (TEXT): Looks like 2024-01-15T20:15:00
HomeConference (TEXT): Can be AFC or NFC.
HomeDivision (TEXT): Can be North, East, West, South.
HomeFullName (TEXT):
HomeOffensiveScheme (TEXT): (3-4, 4-3).
HomeDefensiveScheme (TEXT): (PRO, 2TE, 3WR).
HomeCity (TEXT):
HomeStadiumDetails (TEXT): A map that looks like "{'StadiumID': 3, 'Name': 'MetLife Stadium', 'City': 'East Rutherford', 'State': 'NJ', 'Country': 'USA', 'Capacity': 82500, 'PlayingSurface': 'Artificial', 'GeoLat': 40.813528, 'GeoLong': -74.074361, 'Type': 'Outdoor'}".
TeamHeadCoach (TEXT):
OpponentCoach (TEXT):
AwayConference (TEXT): Can be AFC or NFC.
AwayDivision (TEXT): Can be North, South, East, or West.
AwayFullName (TEXT):
AwayOffensiveScheme (TEXT): (PRO, 2TE, 3WR).
AwayDefensiveScheme (TEXT): (3-4, 4-3).
AwayStadiumDetails (TEXT): A map that looks like "{'StadiumID': 3, 'Name': 'MetLife Stadium', 'City': 'East Rutherford', 'State': 'NJ', 'Country': 'USA', 'Capacity': 82500, 'PlayingSurface': 'Artificial', 'GeoLat': 40.813528, 'GeoLong': -74.074361, 'Type': 'Outdoor'}".
Wins (BIGINT): These are the wins up to the current game. They reset each season and each season type.
Losses (BIGINT): These are the losses up to the current game. They reset each season and each season type.
OpponentWins (BIGINT): These are the opponent's wins up to the current game. They reset each season and each season type.
OpponentLosses (BIGINT): These are the opponent's losses up to the current game. They reset each season and each season type.
Wins_After (BIGINT): These are the wins after the current game. They reset each season and each season type.
Losses_After (BIGINT): These are the losses after the current game. They reset each season and each season type.
OpponentWins_After (BIGINT): These are the opponent's wins after the current game. They reset each season and each season type.
OpponentLosses_After (BIGINT): These are the opponent's losses after the current game. They reset each season and each season type.
Name (TEXT): Home team Stadium Name.
Capacity (BIGINT): Home team stadium Capacity.
PlayingSurface (TEXT): Home team stadium PlayingSurface.
GeoLat (DOUBLE PRECISION): Home team Latitude.
GeoLong (DOUBLE PRECISION): Home team Longitude.
Type (TEXT): Home team type of stadium (Outdoor or Indoor).
IsShortWeek (BIGINT): 1 if the team is playing on a short week, 0 if not.
"""


props_metadata = """
PointSpreadAwayTeamMoneyLine (bigint)
PointSpreadHomeTeamMoneyLine (bigint) 
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


def team_log_and_props_get_answer(model, question):
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
        {'user_question': question, "player_log_table_metadata_string": testnfl_metadata, "props_table_metadata_string": props_metadata, "current_date": str(datetime.datetime.today()).split()[0]}, matched_question=matched_question, matched_sql_query=matched_sql_query)
    
    return_answer = answer.content

    output_count = count_tokens(return_answer)


    return return_answer, input_count, output_count
