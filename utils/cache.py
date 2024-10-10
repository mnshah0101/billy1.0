import os
import dotenv
from pinecone import Pinecone, ServerlessSpec
import pinecone
from openai import OpenAI
import time

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = "billybets"

index = pc.Index(index_name)


def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def get_closest_embedding(question, model="text-embedding-3-large", top_k=1):
    # Get the embedding for the input question
    query_embedding = get_embedding(question, model)

    # Query Pinecone for the closest match
    results = index.query(
        vector=query_embedding , top_k=top_k, include_metadata=True)
    
    print('results')
    
    print(results)
    

    # Retrieve the most relevant result's metadata (which contains the SQL query and question)
    if results['matches']:
        closest_match = results['matches'][0]
        matched_question = closest_match['metadata']['question']
        matched_sql_query = closest_match['metadata']['sql_query']
        score = closest_match['score']

        print(f"Matched Question: {matched_question}")
        print(f"SQL Query: {matched_sql_query}")
        print(f"Similarity Score: {score}")

        return matched_question, matched_sql_query
    else:
        print("No relevant embeddings found.")
        default_sql_query = """
        SELECT
            SUM(CASE WHEN "Score" > "OpponentScore" THEN 1 ELSE 0 END) AS Wins,
            SUM(CASE WHEN "Score" < "OpponentScore" THEN 1 ELSE 0 END) AS Losses
        FROM (
            SELECT DISTINCT ON ("GameKey")
                "GameKey", "Team", "Score", "OpponentScore", "PointSpread"
            FROM teamlog
            WHERE "Team" = 'Any Team'
                AND ABS("PointSpread") <= 3
                AND "SeasonType" = 1
        ) AS unique_games;

        """

        default_question = "What is the win-loss record for TeamName games where the spread closes at 3 points or fewer?"

        return default_question, default_sql_query


