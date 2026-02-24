import sqlite3
import os
import re
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# --- CONFIGURATION ---
# IMPORTANT: Put your API Key here
API_KEY = "REMOVED" 
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app) 

DB_PATH = "medical_insights.db"
TABLE_NAME = "validated_summaries" 

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_context(query):
    """
    Finds relevant summaries. Uses a multi-stage retrieval:
    1. Keyword match (specific)
    2. Broad match (fallback)
    3. Representative sample (safety)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Extract keywords
        keywords = [w for w in re.sub(r'[^a-zA-Z\s]', '', query).lower().split() if len(w) > 3]
        
        results = []
        if keywords:
            search_clause = " OR ".join(["summary LIKE ?" for _ in keywords])
            search_params = [f"%{w}%" for w in keywords]
            cursor.execute(f"SELECT summary FROM {TABLE_NAME} WHERE {search_clause} LIMIT 12", search_params)
            results = cursor.fetchall()

        # If specific search is too narrow, pull a broader sample
        if len(results) < 4:
            cursor.execute(f"SELECT summary FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT 8")
            results = cursor.fetchall()
        
        conn.close()
        return "\n".join([f"- {r['summary']}" for r in results])
        
    except Exception as e:
        return f"Database access error: {str(e)}"

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db_connection()
        count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        conn.close()
        return jsonify({"count": count, "status": "Online"})
    except:
        return jsonify({"count": 0, "status": "Offline"})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("query", "")
    
    context_data = get_context(user_query)
    
    # SYSTEM INSTRUCTION: Tuned for "Meaningful & Balanced" responses
    system_prompt = f"""
    You are 'MedAI Insight', an expert Medical Research Synthesis Agent.
    
    DATASET CONTEXT (Directly from your Spark Pipeline):
    {context_data}
    
    YOUR GOAL:
    Provide a meaningful, medium-length synthesis of the research found in the context above.
    
    GUIDELINES:
    1. STICK TO CONTEXT: Prioritize the findings in the provided dataset. 
    2. DEPTH: Do not just list points. Explain the "why" or "how" based on the summaries.
    3. LENGTH: Aim for 3-5 well-developed bullet points. Not a one-liner, but not a 10-page essay.
    4. STRUCTURE: Use bold headers for categories (e.g., **Pathophysiological Insights**, **Clinical Outcomes**).
    5. NO DENIALS: You HAVE access to the data. The text provided above IS the database.
    6. MEDICAL TONE: Stay professional, objective, and scholarly.
    """
    
    try:
        # Initializing with the gemini-2.5-flash model confirmed in your logs
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_prompt
        )
        
        # We wrap the user query to reinforce the synthesis goal
        response = model.generate_content(
            f"Based on the validated research corpus, please analyze: {user_query}"
        )
        
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"AI Engine Error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)