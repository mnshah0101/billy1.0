from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()



def ask_expert(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert sports analyst. I am a curious user who wants to know more about the NFL and betting. Your queries are strictly about the NFL. The current season is the 2024 season."
            ),
        },
        {
            "role": "user",
            "content": (
                question
            ),
        },
    ]

    client = OpenAI(api_key=os.getenv('PERPLEXITY_KEY'),
                    base_url="https://api.perplexity.ai")



    # chat completion with streaming
    response_stream = client.chat.completions.create(
        model="llama-3.1-sonar-huge-128k-online",
        messages=messages,
        stream=True,
    )
    for response in response_stream:
        yield response.choices[0].message['content']
