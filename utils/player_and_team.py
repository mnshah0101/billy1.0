

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
prompt_template = """

User:

<instructions>
You are a data analyst for an NFL team and you have been asked to generate a SQL query to answer the following question. You do not have to completely answer the question, just generate the SQL query to answer the question, and the result will be processed. Do your best to answer the question and do not use placeholder information. The question is:
`{user_question}`

</instructions>


<database_schema>
The query will run on a database of Player Game Logs and Team Game Logs with the following schema:
{table_metadata_string}
</database_schema>




<special_instructions>
The Team is always short hand, such as WAS for Washington or BAL for Baltimore.
The tables to query are playerlog and teamlog.
Make sure game keys are unique.
Instead of HomeTeam and AwayTeam, reference the Team column and the HomeOrAway Column, The Opponent column will have the opposite side.
You will have to infer player names from little data from your understanding of the NFL. For example, if the user only says Kelce, you have to infer the name Travis Kelce
To calculate "Against the Spread" (ATS), you need to determine whether a team has covered the point spread in a game. The formula for ATS can be derived using the team score, opponent score, and point spread as follows:

Formula:
Calculate the Cover Margin:


Cover Margin=(Score+PointSpread)-OpponentScore
Determine ATS Result:

If Cover Margin > 0, the team covered the spread.
If Cover Margin < 0, the team did not cover the spread.
If Cover Margin = 0, it is a push (no winner against the spread).

Only respond with the sql query, no explanation or anything else. Encompass the sql query with 
```sql

```

You can use MIN(GameKey) to get the earliest game and MAX(GameKey) to get the latest game.

Remember, since the 2023 season has ended, rookies have experience of 2 or less.

A player is injured if the InjuryStatus is Doubtful, Out, or Questionable.

Make sure to use the DISTINCT keyword when necessary to avoid duplicate data.

Usually, even when a player is out or injured, they will have a record in the database. However, sometimes, they might not have a record. Therefore to see how many games a player missed, you can use 17 (or whatever number) - COUNT(DISTINCT GameKey where the player played).

Be careful of periods in the player name. For example, TJ Watt is T.J. Watt in the database.

All columns must be surrounded by double quotes, such as "Name" or "Team".


There is no weather column, so use a combination of temperature, humidity, and wind speed to determine the weather conditions of the game.




</special_instructions>

<question>


Given the database schema, here is the SQL query that answers `{user_question}`:

</question>


<example_response>

```sql
SELECT SUM("RushingYards") AS Yards
FROM playerlog
WHERE "Season" = 2023 AND "Name" = 'Patrick Mahomes'
```

</example_response>


Your response will be executed on a database of NFL Player Logs and NFL Team Logs and the answer will be returned to the User, so make sure the query is correct and will return the correct information.

If the question cannot be answered with the data provided, please return the string "Error: Cannot answer question with data provided."

This is a postgres database. Do not create any new columns or tables. Only use the columns that are in the table.




Assistant: 

"""


sql_prompt = PromptTemplate.from_template(prompt_template)


testnfl_metadata = """
Table: teamlog
GameKey (bigint)
Date (text)
SeasonType (double precision)  - (1=Regular Season, 2=Preseason, 3=Postseason, 4=Offseason, 5=AllStar). The default season type is 1.
Season (double precision) - The default season is 2023.
Week (double precision)- The week resets for each season type. The default week is 1. Week 17 is the last week of the regular season.
Team (text) 
Opponent (text)
HomeOrAway (text) - Could be HOME or AWAY
Score (double precision)
OpponentScore (double precision)
TotalScore (double precision)
Stadium (text)
PlayingSurface (text) - Could be Artificial or Grass
Temperature (double precision)
Humidity (double precision)
WindSpeed (double precision)
OverUnder (double precision)
PointSpread (double precision)
ScoreQuarter1 (double precision)
ScoreQuarter2 (double precision)
ScoreQuarter3 (double precision)
ScoreQuarter4 (double precision)
ScoreOvertime (double precision)
TimeOfPossessionMinutes (double precision)
TimeOfPossessionSeconds (double precision)
TimeOfPossession (text)
FirstDowns (double precision)
FirstDownsByRushing (double precision)
FirstDownsByPassing (double precision)
FirstDownsByPenalty (double precision)
OffensivePlays (double precision)
OffensiveYards (double precision)
OffensiveYardsPerPlay (double precision)
Touchdowns (double precision)
RushingAttempts (double precision)
RushingYards (double precision)
RushingYardsPerAttempt (double precision)
RushingTouchdowns (double precision)
PassingAttempts (double precision)
PassingCompletions (double precision)
PassingYards (double precision)
PassingTouchdowns (double precision)
PassingInterceptions (double precision)
PassingYardsPerAttempt (double precision)
PassingYardsPerCompletion (double precision)
CompletionPercentage (double precision)
PasserRating (double precision)
ThirdDownAttempts (double precision)
ThirdDownConversions (double precision)
ThirdDownPercentage (double precision)
FourthDownAttempts (double precision)
FourthDownConversions (double precision)
FourthDownPercentage (double precision)
RedZoneAttempts (double precision)
RedZoneConversions (double precision)
GoalToGoAttempts (double precision)
GoalToGoConversions (double precision)
ReturnYards (double precision)
Penalties (double precision)
PenaltyYards (double precision)
Fumbles (double precision)
FumblesLost (double precision)
TimesSacked (double precision)
TimesSackedYards (double precision)
QuarterbackHits (double precision)
TacklesForLoss (double precision)
Safeties (double precision)
Punts (double precision)
PuntYards (double precision)
PuntAverage (double precision)
Giveaways (double precision)
Takeaways (double precision)
TurnoverDifferential (double precision)
OpponentScoreQuarter1 (double precision)
OpponentScoreQuarter2 (double precision)
OpponentScoreQuarter3 (double precision)
OpponentScoreQuarter4 (double precision)
OpponentScoreOvertime (double precision)
OpponentTimeOfPossessionMinutes (double precision)
OpponentTimeOfPossessionSeconds (double precision)
OpponentTimeOfPossession (text)
OpponentFirstDowns (double precision)
OpponentFirstDownsByRushing (double precision)
OpponentFirstDownsByPassing (double precision)
OpponentFirstDownsByPenalty (double precision)
OpponentOffensivePlays (double precision)
OpponentOffensiveYards (double precision)
OpponentOffensiveYardsPerPlay (double precision)
OpponentTouchdowns (double precision)
OpponentRushingAttempts (double precision)
OpponentRushingYards (double precision)
OpponentRushingYardsPerAttempt (double precision)
OpponentRushingTouchdowns (double precision)
OpponentPassingAttempts (double precision)
OpponentPassingCompletions (double precision)
OpponentPassingYards (double precision)
OpponentPassingTouchdowns (double precision)
OpponentPassingInterceptions (double precision)
OpponentPassingYardsPerAttempt (double precision)
OpponentPassingYardsPerCompletion (double precision)
OpponentCompletionPercentage (double precision)
OpponentPasserRating (double precision)
OpponentThirdDownAttempts (double precision)
OpponentThirdDownConversions (double precision)
OpponentThirdDownPercentage (double precision)
OpponentFourthDownAttempts (double precision)
OpponentFourthDownConversions (double precision)
OpponentFourthDownPercentage (double precision)
OpponentRedZoneAttempts (double precision)
OpponentRedZoneConversions (double precision)
OpponentGoalToGoAttempts (double precision)
OpponentGoalToGoConversions (double precision)
OpponentReturnYards (double precision)
OpponentPenalties (double precision)
OpponentPenaltyYards (double precision)
OpponentFumbles (double precision)
OpponentFumblesLost (double precision)
OpponentTimesSacked (double precision)
OpponentTimesSackedYards (double precision)
OpponentQuarterbackHits (double precision)
OpponentTacklesForLoss (double precision)
OpponentSafeties (double precision)
OpponentPunts (double precision)
OpponentPuntYards (double precision)
OpponentPuntAverage (double precision)
OpponentGiveaways (double precision)
OpponentTakeaways (double precision)
OpponentTurnoverDifferential (double precision)
RedZonePercentage (double precision)
GoalToGoPercentage (double precision)
QuarterbackHitsDifferential (double precision)
TacklesForLossDifferential (double precision)
QuarterbackSacksDifferential (double precision)
TacklesForLossPercentage (double precision)
QuarterbackHitsPercentage (double precision)
TimesSackedPercentage (double precision)
OpponentRedZonePercentage (double precision)
OpponentGoalToGoPercentage (double precision)
OpponentQuarterbackHitsDifferential (double precision)
OpponentTacklesForLossDifferential (double precision)
OpponentQuarterbackSacksDifferential (double precision)
OpponentTacklesForLossPercentage (double precision)
OpponentQuarterbackHitsPercentage (double precision)
OpponentTimesSackedPercentage (double precision)
Kickoffs (double precision)
KickoffsInEndZone (double precision)
KickoffTouchbacks (double precision)
PuntsHadBlocked (double precision)
PuntNetAverage (double precision)
ExtraPointKickingAttempts (double precision)
ExtraPointKickingConversions (double precision)
ExtraPointsHadBlocked (double precision)
ExtraPointPassingAttempts (double precision)
ExtraPointPassingConversions (double precision)
ExtraPointRushingAttempts (double precision)
ExtraPointRushingConversions (double precision)
FieldGoalAttempts (double precision)
FieldGoalsMade (double precision)
FieldGoalsHadBlocked (double precision)
PuntReturns (double precision)
PuntReturnYards (double precision)
KickReturns (double precision)
KickReturnYards (double precision)
InterceptionReturns (double precision)
InterceptionReturnYards (double precision)
OpponentKickoffs (double precision)
OpponentKickoffsInEndZone (double precision)
OpponentKickoffTouchbacks (double precision)
OpponentPuntsHadBlocked (double precision)
OpponentPuntNetAverage (double precision)
OpponentextraPointKickingAttempts (double precision)
OpponentextraPointKickingConversions (double precision)
OpponentextraPointsHadBlocked (double precision)
OpponentextraPointPassingAttempts (double precision)
OpponentextraPointPassingConversions (double precision)
OpponentextraPointRushingAttempts (double precision)
OpponentextraPointRushingConversions (double precision)
OpponentFieldGoalAttempts (double precision)
OpponentFieldGoalsMade (double precision)
OpponentFieldGoalsHadBlocked (double precision)
OpponentPuntReturns (double precision)
OpponentPuntReturnYards (double precision)
OpponentKickReturns (double precision)
OpponentKickReturnYards (double precision)
OpponentInterceptionReturns (double precision)
OpponentInterceptionReturnYards (double precision)
SoloTackles (double precision)
AssistedTackles (double precision)
Sacks (double precision)
SackYards (double precision)
PassesDefended (double precision)
FumblesForced (double precision)
FumblesRecovered (double precision)
FumbleReturnYards (double precision)
FumbleReturnTouchdowns (double precision)
InterceptionReturnTouchdowns (double precision)
BlockedKicks (double precision)
PuntReturnTouchdowns (double precision)
PuntReturnLong (double precision)
KickReturnTouchdowns (double precision)
KickReturnLong (double precision)
BlockedKickReturnYards (double precision)
BlockedKickReturnTouchdowns (double precision)
FieldGoalReturnYards (double precision)
FieldGoalReturnTouchdowns (double precision)
PuntNetYards (double precision)
OpponentSoloTackles (double precision)
OpponentAssistedTackles (double precision)
OpponentSacks (double precision)
OpponentSackYards (double precision)
OpponentPassesDefended (double precision)
OpponentFumblesForced (double precision)
OpponentFumblesRecovered (double precision)
OpponentFumbleReturnYards (double precision)
OpponentFumbleReturnTouchdowns (double precision)
OpponentInterceptionReturnTouchdowns (double precision)
OpponentBlockedKicks (double precision)
OpponentPuntReturnTouchdowns (double precision)
OpponentPuntReturnLong (double precision)
OpponentKickReturnTouchdowns (double precision)
OpponentKickReturnLong (double precision)
OpponentBlockedKickReturnYards (double precision)
OpponentBlockedKickReturnTouchdowns (double precision)
OpponentFieldGoalReturnYards (double precision)
OpponentFieldGoalReturnTouchdowns (double precision)
OpponentPuntNetYards (double precision)
IsGameOver (bigint)
TeamName (text)
DayOfWeek (text)
PassingDropbacks (double precision)
OpponentPassingDropbacks (double precision)
TeamGameID (double precision)
TwoPointConversionReturns (double precision)
OpponentTwoPointConversionReturns (double precision)
TeamID (double precision)
OpponentID (double precision)
Day (text)
DateTime (text)
GlobalGameID (double precision)
GlobalTeamID (double precision)
GlobalOpponentID (double precision)
ScoreID (double precision)
outer_key (bigint)
HomeConference (text)
HomeDivision (text)
HomeFullName (text)
HomeOffensiveScheme (text)
HomeDefensiveScheme (text)
HomeCity (text)
HomeStadiumDetails (text)
HomeHeadCoach (text)
AwayConference (text)
AwayDivision (text)
AwayFullName (text)
AwayOffensiveScheme (text)
AwayDefensiveScheme (text)
AwayCity (text)
AwayStadiumDetails (text)
AwayHeadCoach (text)
HomeOffensiveCoordinator (text)
HomeDefensiveCoordinator (text)
HomeSpecialTeamsCoach (text)
AwayOffensiveCoordinator (text)
AwayDefensiveCoordinator (text)
AWaySpecialTeamsCoach (text)
Wins (double precision) - These are the wins up to the current game. They reset each season and each season type.
Losses (double precision) - These are the losses up to the current game. They reset each season and each season type.
OpponentWins (double precision) - These are the opponent's wins up to the current game. They reset each season and each season type.
OpponentLosses (double precision) - These are the opponent's losses up to the current game. They reset each season and each season type.
IsShortWeek (bigint) - 1 if the team has a short week, 0 otherwise
StadiumID (bigint)
Name (text) - Home team Stadium Name
City (text) - Home team  City
State (text) - Home team state State
Country (text) - Home team Country
Capacity (bigint) - Home team stadium Capacity
PlayingSurface.1 (text) - Home team stadium PlayingSurface
GeoLat (double precision) - Home team Latitude
GeoLong (double precision) - Home team Longitude
Type (text) - Home team type of stadium (Outdoor or Indoor)

Table: playerlog
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
IsGameOver (bigint)
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
InjuryStatus (text) = [None, 'Questionable', 'Probable', 'Out', 'Doubtful']
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
Day (text)
DateTime (text)
GlobalGameID (bigint)
GlobalTeamID (bigint)
GlobalOpponentID (bigint)
ScoreID (bigint)
FantasyPointsFantasyDraft (double precision)
OffensiveFumbleRecoveryTouchdowns (double precision)
SnapCountsConfirmed (bigint)
Updated (text)
ScoringDetails (text) - A JSON array of scoring details if the player scored any points in the game. It looks like this: 
[{'GameKey': '200230131',
  'SeasonType': 3,
  'PlayerID': 8223,
  'Team': 'SF',
  'Season': 2002,
  'Week': 1,
  'ScoringType': 'RushingTouchdown',
  'Length': 14,
  'ScoringDetailID': 292738,
  'PlayerGameID': 1797109,
  'ScoringPlayID': 78786}]

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
Experience (double precision) - The number of years the player has played in the NFL. Since it is updated every spring, rookies have a value of 2.

"""


def player_and_team_log_get_answer(model, question):
    llm = None
    if model == 'openai':
        llm = ChatOpenAI(model='gpt-4o', temperature=0.9)

    elif model == 'anthropic':
        llm = ChatAnthropic(model_name='claude-3-5-sonnet-20240620',
                           )

    llm_chain = sql_prompt | llm
    answer = llm_chain.invoke(
    {'user_question': question, "table_metadata_string": testnfl_metadata})

    return answer.content


