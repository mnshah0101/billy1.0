from langchain_openai import ChatOpenAI
import requests

evaluation_questions = [
    "What is Derick Henry's Yards Per Carry Against the Jaguars?",
    "How many 100-yard receiving games did Amon-Ra St. Brown have in 2023?",
    "How many yards has CMC rushed for against the Seahawks in his career?"
]



def evaluate_consistency(): 
    for question in evaluation_questions: 
        responses = [] 
        try: 
            for _ in range(3): 
                response = requests.post(url, json={ 
                    "message": question
                })
                if response.status_code == 200:
                    responses.append(response)
        except: 
            break
        
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key="...",  
            )



    

    return 