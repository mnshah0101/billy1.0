from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.question_parser import question_chooser
from utils.team_log import team_log_get_answer
from utils.player_log import player_log_get_answer
from utils.playbyplay import play_by_play_get_answer
from utils.executor import execute_query, extract_sql_query
from utils.answer_parser import get_answer
from flask_socketio import SocketIO
from flask_socketio import send, emit
from utils.player_and_team import player_and_team_log_get_answer
from utils.perplexity import ask_expert
from flask import request
from supabase import create_client, Client
import dotenv
import os




dotenv.load_dotenv()




app = Flask(__name__)
CORS(app) 



# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


socketio = SocketIO(app, cors_allowed_origins='*')

global_bucket = None 



@socketio.on('billy')
def chat(data):
    if 'message' not in data:
        emit('billy', {'response': 'I am sorry, I do not have an answer for that question.',
             'type': 'query', 'status': 'done'})
        print('No message or ip or session')
        return
    print("Message:")
    print(data['message'])

    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)


    message = data['message']['message']

    session = data['message']['session']

    print(f'IP: {ip}')
    print(f'Session: {session}')

    global global_bucket

    while True:
        try:
            # Call the question_chooser function to get the bucket and question
            bucket, question = question_chooser('anthropic', message)
            global_bucket = bucket

            print(f'Bucket: {bucket}')
            print(f'Question: {question}')
            
            if bucket =='Conversation':
                emit('billy', {'response':question, 'type': 'answer', 'status': 'done'})
                return

            if bucket == 'NoBucket':
                if question == '':
                    emit('billy', {
                        'response': "I am sorry, I do not have an answer for that question.", 'type': 'answer', 'status': 'done'})
                    return

                emit('billy', {
                        'response':question, 'type': 'answer', 'status': 'done'})
                return
            
            if bucket == 'ExpertAnalysis':
                emit('billy', {'response': '',
                               'type': 'query', 'status': 'generating'})
                generator = ask_expert(question)
                answer = ''
                generating_answer = True
                while generating_answer:
                    try:
                        next_answer = next(generator)
                        answer += next_answer
                        emit('billy', {'response': next_answer,
                             'type': 'answer', 'status': 'generating'})
                    except Exception as e:
                        generating_answer = False
                        emit('billy', {'response': next_answer,
                             'type': 'answer', 'status': 'done'})
                        
                return answer

            raw_query = None

            if bucket == 'TeamGameLog':
                raw_query = team_log_get_answer('anthropic', question)
            elif bucket == 'PlayerGameLog':
                raw_query = player_log_get_answer('anthropic', question)
            elif bucket == 'PlayByPlay':
                raw_query = play_by_play_get_answer('anthropic', question)
            elif bucket == 'TeamAndPlayerLog':
                raw_query = player_and_team_log_get_answer(
                    'anthropic', question)
            
            print(f'Raw Query: {raw_query}')
                
            if 'error' and 'cannot' in raw_query.lower():
                emit('billy', {'response': '',
                               'type': 'query', 'status': 'generating'})
                generator = ask_expert(question)
                answer = ''
                generating_answer = True
                while generating_answer:
                    try:
                        next_answer = next(generator)
                        answer += next_answer
                        emit('billy', {'response': next_answer,
                             'type': 'answer', 'status': 'generating'})
                    except Exception as e:
                        generating_answer = False
                        emit('billy', {'response': next_answer,
                             'type': 'answer', 'status': 'done'})

                return answer


            # Extract the SQL query from the raw_query
            query = extract_sql_query(raw_query)

            emit('billy', {'response': query,
                    'type': 'query', 'status': 'generating'})

            # Execute the SQL query
            result = execute_query(query)

            # If execution reaches here, the query was successful, break the loop
            break

        except Exception as e:
            print(f'Error: {e}. Retrying...')

   

    answer = get_answer('openai', question, query, result)

    answerGenerating = True
    answer_string = ''

    while answerGenerating:
        try:
            next_answer = next(answer)
            answer_string += next_answer
            emit('billy', {'response': answer_string,
                 'type': 'answer', 'status': 'generating', 'bucket': bucket})
        except Exception as e:
            answerGenerating = False

            emit('billy', {'response': answer_string,
                 'type': 'answer', 'status': 'done'})
            


    return answer_string

@app.route('/store-query', methods=['POST'])
def store_query():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    required_fields = ['question', 'answer', 'correct', 'category', 'sql']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} Check not found in request data'}), 400

    question = data['question']
    bucket = global_bucket
    answer = data['answer']
    correct = data['correct']
    category = data['category']
    sql = data['sql']
    user_id = data.get('user_id', None)

    try:
        # Check if an entry with the same question exists
        existing_entry = supabase.table("store-queries").select("*").eq("question", question).execute()

        if existing_entry.data:
            # Update the existing entry
            result = supabase.table("store-queries").update({
                "answer": answer,
                "correct": correct,
                "category": category,
                "sql": sql,
                "seen": True
            }).eq("id", existing_entry.data[0]['id']).execute()

            if result.data:
                return jsonify({'message': 'Query updated successfully'}), 200
            else:
                return jsonify({'error': 'No changes made to the existing entry'}), 400
        else:
            # Insert a new entry
            new_entry = {
                "question": question,
                "answer": answer,
                "bucket": bucket,
                "correct": correct,
                "category": category,
                "sql": sql,
                "user_id": user_id,
                "seen": False,
            }

            result = supabase.table("store-queries").insert(new_entry).execute()

            if result.data:
                return jsonify({'message': 'New query stored successfully'}), 201
            else:
                return jsonify({'error': 'Failed to insert new entry'}), 500

    except Exception as e:
        print(f"Error interacting with Supabase: {e}")
        return jsonify({'error': 'Could not store/update query', 'details': str(e)}), 500

@app.route('/chat')
def chat_http(data):
    if 'message' not in data:
        emit('billy', {'response': 'I am sorry, I do not have an answer for that question.',
             'type': 'query', 'status': 'done'})
        return

    message = data['message']

    
    while True:
        try:
            # Call the question_chooser function to get the bucket and question
            bucket, question = question_chooser('openai', message)

            print(f'Bucket: {bucket}')
            print(f'Question: {question}')

            if bucket == 'NoBucket':
                emit('billy', {
                        'response': "I am sorry, I do not have an answer for that question.", 'type': 'answer', 'status': 'done'})
                return

            raw_query = None

            if bucket == 'TeamGameLog':
                raw_query = team_log_get_answer('anthropic', question)
            elif bucket == 'PlayerGameLog':
                raw_query = player_log_get_answer('anthropic', question)
            elif bucket == 'PlayByPlay':
                raw_query = play_by_play_get_answer('anthropic', question)
            elif bucket == 'TeamAndPlayerLog':
                raw_query = player_and_team_log_get_answer(
                    'anthropic', question)

            # Extract the SQL query from the raw_query
            query = extract_sql_query(raw_query)

            emit('billy', {'response': query,
                    'type': 'query', 'status': 'generating'})

            # Execute the SQL query
            result = execute_query(query)

            # If execution reaches here, the query was successful, break the loop
            break

        except Exception as e:
            print(f'Error: {e}. Retrying...')

   

    answer = get_answer('openai', question, query, result)

    answerGenerating = True
    answer_string = ''

    while answerGenerating:
        try:
            next_answer = next(answer)
            answer_string += next_answer
           
        except Exception as e:
            answerGenerating = False
            
    return answer_string





if __name__ == '__main__':

    app.run(debug=True)


    socketio.run(app)
