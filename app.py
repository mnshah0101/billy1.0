from flask import Flask, jsonify, request
import json
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
from utils.playerlogandprops import player_log_and_props_get_answer
from utils.teamlogandprops import team_log_and_props_get_answer
from utils.props import props_log_get_answer
from utils.perplexity import ask_expert
from utils.futures import futures_log_get_answer
from utils.CountUtil import count_tokens
from flask import request
import tiktoken
from supabase import create_client, Client
import dotenv
from functools import wraps

import os
import logging


dotenv.load_dotenv()


app = Flask(__name__)
CORS(app)


# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


socketio = SocketIO(app, cors_allowed_origins='*')

global_bucket = None



def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('Error from handle_errors')
            print(f"Error: {e}")
            emit('billy', {
                'response': "I'm sorry, an error occurred while processing your request.",
                'type': 'answer',
                'status': 'done'
            })
    return wrapper


def get_ip_and_session(data):
    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    session = data['message'].get('session', 'Unknown')
    return ip, session


def process_expert_analysis(question):
    emit('billy', {'response': '', 'type': 'query', 'status': 'generating'})
    generator = ask_expert(question)
    answer = ''
    for next_answer in generator:
        answer += next_answer
        emit('billy', {'response': next_answer,
             'type': 'answer', 'status': 'generating'})
    emit('billy', {'response': next_answer,
         'type': 'answer', 'status': 'done'})
    return ''


def process_database_query(bucket, question):
    bucket_to_function = {
        'TeamGameLog': team_log_get_answer,
        'PlayerGameLog': player_log_get_answer,
        'PlayByPlay': play_by_play_get_answer,
        'TeamAndPlayerLog': player_and_team_log_get_answer,
        'Props': props_log_get_answer,
        'PlayerLogAndProps': player_log_and_props_get_answer,
        'TeamLogAndProps': team_log_and_props_get_answer,
        'Futures': futures_log_get_answer
    }

    get_answer_func = bucket_to_function.get(bucket)
    if not get_answer_func:
        raise ValueError(f"Unknown bucket: {bucket}")
    


    raw_query, sql_input_tokens, sql_output_tokens = get_answer_func('openai', question)
    if 'error' in raw_query.lower() or 'cannot' in raw_query.lower():
        return process_expert_analysis(question)

    query = extract_sql_query(raw_query)
    emit('billy', {'response': query, 'type': 'query', 'status': 'generating'})
    result = execute_query(query)
    print(f"Result: {result}")

    tokens = count_tokens(str(result))
    if tokens > 5000:
        return process_expert_analysis(question)
    
    answer_input_tokens = count_tokens(question) + count_tokens(raw_query) + count_tokens(str(result)) + 700

    print("running up to get_answer")

    
    return get_answer('openai', question, query, result), sql_input_tokens, sql_output_tokens, answer_input_tokens, raw_query


@socketio.on('billy')
@handle_errors
def chat(data):
    if 'message' not in data:
        emit('billy', {'response': 'I am sorry, I do not have an answer for that question.',
                       'type': 'query', 'status': 'done'})
        print('No message or ip or session')
        return

    message = data['message']['message']
    ip, session = get_ip_and_session(data)
    print(f"Message: {message}")
    print(f"IP: {ip}")
    print(f"Session: {session}")

    bucket, question, question_chooser_input_count, question_chooser_output_count = question_chooser('openai', message)
    print(f"Bucket: {bucket}")
    print(f"Question: {question}")

    global global_bucket
    global_bucket = bucket

    if bucket == 'Conversation':
        emit('billy', {'response': question,
             'type': 'answer', 'status': 'done'})
        return

    if bucket == 'NoBucket':
        response = question if question else "I am sorry, I do not have an answer for that question."
        emit('billy', {'response': response,
             'type': 'answer', 'status': 'done'})
        return

    if bucket == 'ExpertAnalysis':
        return process_expert_analysis(question)
    
    
    try:

        answer_generator, input_sql_tokens, output_sql_tokens, answer_input_tokens, raw_sql_query = process_database_query(bucket, question)
        answer_string = ''
        for next_answer in answer_generator:
            answer_string += next_answer
            emit('billy', {'response': answer_string, 'type': 'answer',
                'status': 'generating', 'bucket': bucket})


        emit('billy', {'response': answer_string,
            'type': 'answer', 'status': 'done'})
        
        #store_query()
        print(f"Input SQL Tokens: {input_sql_tokens}")
        print(f"Output SQL Tokens: {output_sql_tokens}")
        print(f"Question Chooser Input Tokens: {question_chooser_input_count}")
        print(f"Question Chooser Output Tokens: {question_chooser_output_count}")
        print(f"Answer Input Tokens: {answer_input_tokens}")

        answer_output_tokens = count_tokens(answer_string)

        #cost is sum(inputs) * (3.50/1,000,000) + sum(outputs) * (15/1,000,000)



        cost = (question_chooser_input_count + input_sql_tokens + answer_input_tokens) * (3.50/1000000) + (question_chooser_output_count + output_sql_tokens + answer_output_tokens) * (15/1000000)
        
        #TODO: Store the chat in the db with 
        supabase.table('billy_answers').insert({
            'question': question,
            'bucket': bucket,
            'sql_query': raw_sql_query,
            'answer': answer_string,
            'question_parser_input_tokens': question_chooser_input_count,
            'question_parser_output_tokens': question_chooser_output_count,
            'sql_input_tokens': input_sql_tokens,
            'sql_output_tokens': output_sql_tokens,
            'answer_input_tokens': answer_input_tokens,
            'answer_output_tokens': answer_output_tokens,
            'cost': cost,
        }).execute()



    
        return answer_string
    except Exception as e:
        emit('billy', {
            'response': "I'm sorry, an error occurred while processing your request. Please try again.",
            'type': 'answer',
            'status': 'done'
        })
        print(f"Error: {e}")
        return 


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
        existing_entry = supabase.table(
            "store-queries").select("*").eq("question", question).execute()

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

            result = supabase.table(
                "store-queries").insert(new_entry).execute()

            if result.data:
                return jsonify({'message': 'New query stored successfully'}), 201
            else:
                return jsonify({'error': 'Failed to insert new entry'}), 500

    except Exception as e:
        print(f"Error interacting with Supabase: {e}")
        return jsonify({'error': 'Could not store/update query', 'details': str(e)}), 500





@app.route('/post-chats', methods=['POST'])
def store_chats():
    # Sample input:
    # {
    #     "user_id": "123",
    #     "messages": [
    #         {"message": "test", "sender": "user"},
    #         {"message": "test2", "sender": "billy"}
    #     ],
    #      "name": "Ronit"
    #       "sql_query": "SELECT(*) FROM *"
    #       "chat_id": "123123"
    # }

    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    required_fields = ['user_id', 'messages', 'name', 'sql_query', 'chat_id']

    # Check if user_id and chat are provided
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} not found in request data'}), 400

    user = data['user_id']
    messages = data['messages']
    name = data['name']
    sql_query = data['sql_query']
    chat_id = data['chat_id']

    try:
        # Insert into 'chats' table, automatically sets 'created_at'
        # Insert into 'chats' table
        response = supabase.table('chats').upsert({
            "messages": messages,
            "user_id": user,
            "name": name,
            "sql_query": sql_query,
            "id": chat_id,
        }).execute()

        if hasattr(response, 'error') and response.error:
            return jsonify({"error": str(response.error)}), 500

        response = supabase.table('profiles').select(
            'chats').eq('user_id', user).execute()

        if hasattr(response, 'error') and response.error:
            return jsonify({"error": str(response.error)}), 500

        if response.data:
            current_chats = response.data[0].get('chats') or []
        else:
            return jsonify({"error": "User profile not found"}), 404

        if chat_id not in current_chats:
            # Append 'chat_id' to 'current_chats'
            current_chats.append(chat_id)

            # Update the 'chats' array in 'profiles' table
            response = supabase.table('profiles').update({
                "chats": current_chats
            }).eq('user_id', user).execute()

            if hasattr(response, 'error') and response.error:
                return jsonify({"error": str(response.error)}), 500

            return jsonify({"message": "Chat added successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Chat added successfully"}), 200


# Route to retrieve all chats for a user
@app.route('/retrieve-all-chats', methods=['POST'])
def retrieve_all_chats():
    # Sample input:
    # {
    #     "user_id": "123"
    # }
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id not found in request data'}), 400

    try:
        # Check if the user exists in the profiles table
        user_check = supabase.table('profiles').select(
            'user_id').eq('user_id', user_id).execute()

        if not user_check.data:
            return jsonify({"error": "User not found"}), 404

        # Query the chats table to get chats for the given user_id
        response = supabase.table('chats').select(
            '*').eq('user_id', user_id).execute()

        if hasattr(response, 'error') and response.error:
            return jsonify({"error": str(response.error)}), 500

        chats = response.data if response.data else []

        # Return all chats associated with the user
        return jsonify({"chats": chats}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to retrieve a specific chat by chat_id


@app.route('/retrieve-chat', methods=['POST'])
def retrieve_chat():
    # Sample input:
    # {
    #     "chat_id": "your_chat_id_here"
    # }
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    chat_id = data.get('chat_id')
    if not chat_id:
        return jsonify({'error': 'chat_id not found in request data'}), 400

    try:
        # Query the chats table to retrieve the specific chat by chat_id
        response = supabase.table('chats').select(
            '*').eq('id', chat_id).execute()

        if hasattr(response, 'error') and response.error:
            return jsonify({"error": str(response.error)}), 500

        chat = response.data[0] if response.data else None
        if not chat:
            return jsonify({"error": "Chat not found"}), 404

        # Extract the 'messages' field from the chat data
        messages = chat.get('messages', [])

        return jsonify({"chat": messages}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':

    app.run(debug=True, port=5000)

    socketio.run(app, port=5000)

