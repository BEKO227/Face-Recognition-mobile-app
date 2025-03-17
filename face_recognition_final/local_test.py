import os
from dotenv import load_dotenv
from supabase import create_client, Client
import datetime
from groq import Groq
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Function to process plain language queries using Groq LLaMA
def process_nlp_query(query):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a database assistant connected to a Supabase database that contains a table named 'face_logs'. "
                        "This table has columns: 'name' (string), 'timestamp' (datetime), 'image_url' (string), and 'authorized' (boolean). "
                        "Extract the necessary information from the user's query to help retrieve data. "
                        "Provide a JSON object with a 'filters' field for querying. "
                        "The 'filters' field should contain any 'date' (in 'YYYY-MM-DD' format), 'name', and 'authorized' status extracted from the query. "
                        "Only return the JSON object without any additional text."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=150,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract the response
        response = completion.choices[0].message.content.strip()
        print("Groq API response:", response)

        # Parse JSON from Groq's output
        try:
            query_params = json.loads(response)
            return query_params  # Return the parameters to be used in querying
        except json.JSONDecodeError:
            print("Invalid JSON received from Groq.")
            return None
    except Exception as e:
        print(f"Error processing NLP query with Groq: {e}")
        return None

# Function to query face_logs with filters
def query_face_logs_with_filters(filters):
    try:
        query = supabase.table('face_logs').select('*')

        # Apply filters based on the extracted parameters
        for key, value in filters.items():
            if key == 'date':
                # Parse the date
                try:
                    start_time = datetime.datetime.strptime(value, '%Y-%m-%d')
                except ValueError:
                    # Try parsing other date formats if needed
                    start_time = datetime.datetime.strptime(value, '%d-%m-%Y')
                end_time = start_time + datetime.timedelta(days=1)
                query = query.gte('timestamp', start_time.isoformat()).lt('timestamp', end_time.isoformat())
            elif key == 'name':
                query = query.eq('name', value)
            elif key == 'authorized':
                # Convert to boolean
                authorized = value.lower() in ['true', 'yes', '1']
                query = query.eq('authorized', authorized)
            # Add more filters as needed

        # Execute the query
        response = query.execute()

        if not response.data:
            return {"status": "error", "message": "No matching entries found."}
        
        # Return results in a structured format
        results = [
            {
                "name": log['name'],
                "timestamp": log['timestamp'],
                "authorized": log['authorized'],
                "image_url": log['image_url']
            }
            for log in response.data
        ]
        
        return {"status": "success", "data": results}
    
    except Exception as e:
        print(f"Error querying face_logs with filters: {e}")
        return {"status": "error", "message": "Error retrieving data from the database."}

# Function to handle queries
def handle_query(query):
    # Use Groq LLaMA to process the natural language query
    query_params = process_nlp_query(query)

    if query_params is None:
        return {"status": "error", "message": "Sorry, I couldn't understand your query."}

    # Now, use the extracted parameters to query the database
    try:
        filters = query_params.get("filters", {})
        results = query_face_logs_with_filters(filters)
        return results
    except Exception as e:
        print(f"Error handling query: {e}")
        return {"status": "error", "message": "Sorry, an error occurred while processing your query."}

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/query', methods=['GET'])
def query():
    user_query = request.args.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    response = handle_query(user_query)
    
    # Return the response in a JSON format
    return jsonify(response)

if __name__ == "__main__":
    print("Groq-Enabled Natural Language Query Interface")
    print("Listening for queries via GET requests...")
    app.run(port=5000)
