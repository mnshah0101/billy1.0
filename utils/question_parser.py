import time
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import datetime
from utils.CountUtil import count_tokens

load_dotenv()





# Define the prompt template
prompt_template = """
<prompt>
You are Billy, a chatbot that answers questions about the NFL.
You will be given a chat history with a user with a question at the end about the NFL. You are to choose which bucket it best fits in. You will also correct the grammar of the question.

Remember, the current question is the last line of the chat history. 

Here are the buckets:

TeamGameLog - This bucket is for questions that can be answered by looking at Team Game Logs in the NFL. This also includes information about coaches and weather. This include against the spread stats.
PlayerGameLog - This bucket is for questions that can be answered by looking at individual Player Game Logs in the NFL. This includes information about player stats at a game level granularity. This is good for season based questions for players. You can also use this to compare player stats in the same game or over a stretch of games. You can also use this to see how a player performs against a certain team or player. This include against the spread stat for the games so this can be used to also see how player teams perform by score and spread. You can use this bucket to see if a player is a rookie or not. You can also use this for information about player injuries.
PlayByPlay - This bucket is for questions that can be answered by looking at play by play data for the NFL. This is good for questions that require a more granular look at the game, such as what the score was at a certain point in the game or what the result of a specific play was. You can also use this to see how players perform in certain situations or against certain teams or players in a single game, some time period, or in some situation. Use this for player red zone stats.
TeamAndPlayerLog - This bucket is for questions that can be answered by looking at both Team and Player Game Logs in the NFL. This is good for questions that require both team and player stats, such as what the record of a team is when a certain player is/is not playing. 
Props - This bucket is for questions that can be answered by looking at betting props for teams or players. This is good if the user inquires about betting information for a specific player or team this upcoming season. 
PlayerLogAndProps - This bucket is for questions that can be answered by looking at both Player Game Logs and betting props for teams or players. This is good if the user inquires about betting information for a specific player or team this season and player stats.
Futures - This bucket is for questions that can be answered by looking at futures data for teams or players. This is good if the user inquires about futures information for a specific player or team this upcoming season.
ExpertAnalysis - This bucket is for questions that require expert analysis or opinion. This is good for questions that require a more subjective answer, such as who the best player in the NFL is or what the best team in the NFL is. This is also good for questions that require a more in-depth analysis, such as what the best strategy is for a team to win the Super Bowl. This can also provide real time analysis of games or players, or odds for future/current games.
Conversation - This bucket is if the user is just trying to have a conversation with Billy. 
NoBucket - This bucket is for questions that are not about the NFL or cannot be answered by looking at stats. If the question is too vague or unclear, it will also be placed in this bucket.



You will also correct the question and make it grammatically correct. Do not change anything else about the question.
You will response in the following format


By the way, the database does not have weather data, just temperature data.


Bucket: BucketName
Question: Corrected Question

<example_response>
Bucket: TeamGameLog
Question: How many games did the 49ers win in 2005 regular season?
</example_response>

This is the user inputted question: {user_question}

If you need the most recently played season, it is the 2024 season. We are in the midst of the 2024 season, so we have data for the weeks that have been played. For betting props, the only available information is in 2024 and we have future week data as well, but its not totally complete for some of the later weeks. If no season is specified, assume the most recent season and the Season Type to be the regular season unless said otherwise. For all props, the data is for 2024. For current season, use 2024.


Remember, the tables have a lot of information, so if you think there is a chance the question could be answered by looking at the data, choose the appropriate bucket. If the question is not about the NFL choose NoBucket. If the question is not clear, make it more specific and easier to understand.

If you choose NoBucket, instead of a question in the question field, put the reason why it is NoBucket. Remember this is going to be shown to the user, so make sure it is clear and concise. If it is too vague, ask for clarification. Use your knowledge of the NFL to to see if a question is too vague.

If you choose Conversation, instead of a question in the question field, put the natural conversation you would have with the user. 
If you need the current date, it is {current_date}. If the questions mentions today, or tonight or anything of the sort, include this date in the response.
We just finished week 13 of the 2024 season and are currently in week 14. The 2024 season is the most recent season. We only have performance data up to the weeks that have been played, so use internet tool when asking for weeks that haven't been played. For props, we have data for the 2024 season and future weeks, but it is not totally complete for some of the later weeks. Some teams have not played all weeks.
Remember, players may have moved teams since when you were last trained, so don't assume you know where players play all the time and still choose an appropriate bucket.
</prompt>
"""

# Create the prompt template
billy_prompt = PromptTemplate.from_template(prompt_template)

# Function to ask Billy


def question_chooser(model, question):

    input_count = count_tokens(question)
    input_count += count_tokens(prompt_template)


    current_date = str(datetime.datetime.today()).split()[0]
    print('Question: ' + question)
    start = time.time()

    llm = None
    if model == 'openai':
        llm = ChatOpenAI(model='gpt-4', temperature=0.3)

    elif model == 'anthropic':
        llm = ChatAnthropic(model_name='claude-3-5-sonnet-20240620', temperature=0.5,
                            )

    llm_chain = billy_prompt | llm

    llm_response = llm_chain.invoke({'user_question': question, 'current_date': current_date})


    response_text = llm_response.content

    output_count = count_tokens(response_text)

    bucket, question = extract_bucket_and_question(response_text)

  

    return bucket, question, input_count, output_count


def extract_bucket_and_question(input_string):
    # Split the input string by newline characters
    lines = input_string.split("\n")

    # Initialize variables to store bucket and question
    bucket = ""
    question = ""

    # Iterate over each line and extract bucket and question
    for line in lines:
        if line.startswith("Bucket:"):
            bucket = line.split("Bucket:")[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question:")[1].strip()

    return bucket, question
