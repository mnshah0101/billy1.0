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
        return None


