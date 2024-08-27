from openai import OpenAI

YOUR_API_KEY = "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"


def ask_expert(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert sports analyst. I am a curious user who wants to know more about the NFL and betting. Your queries are strictly about the NFL."
            ),
        },
        {
            "role": "user",
            "content": (
                question
            ),
        },
    ]

    client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")



    # chat completion with streaming
    response_stream = client.chat.completions.create(
        model="llama-3-sonar-large-32k-online",
        messages=messages,
        stream=True,
    )
    for response in response_stream:
        yield response.choices[0].message['content']
