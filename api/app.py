import os
import json
import logging
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
import traceback



# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment
load_dotenv()
logger.info("Environment variables loaded")

# Flask setup
app = Flask(__name__)
from flask_cors import CORS

CORS(app, 
     resources={r"/ask": {"origins": [
         "http://localhost:8081",
         "https://edu-frontend-1.onrender.com"
     ]}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])


logger.info("Flask app initialized with CORS")

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask chatbot backend is running!"

# MongoDB
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client["test"]
user_collection = db["users"]
marks_collection = db["marks"]
attendance_collection = db["attendances"]
remarks_collection = db["professorremarks"]

# Files
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cache_file = os.path.join(BASE_DIR, "cache.json")

# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# Check for API keys
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
if not serpapi_api_key:
    logger.error("SERPAPI_API_KEY not found in environment variables")

# LLM Setup
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
logger.info("OpenAI models initialized")
classify_prompt = PromptTemplate(
    template="""
Decide whether the following question is personal (i.e., it is asking about the student's own academic records like their marks, attendance, CGPA, or remarks).
every user is defined a role whether professor,student,admin this is also a personal query to ask about role 
📘 `users`: `_id`, `email`, `name`, `cg`, `role`, `hasAccess`  
📘 `marks`: `studentId`, `courseId`, `title`, `score`, `maxScore`, `type`, `feedback`, `createdAt`  
📘 `attendances`: `studentId`, `courseId`, `date`, `status`  
📘 `professorremarks`: `studentId`, `courseId`, `text`  

Return only **true** if it is personal. Return nothing otherwise.

Question: {question}
Response:
""",
    input_variables=["question"]
)


query_prompt = PromptTemplate(
    template="""
You are an expert MongoDB query assistant.

Your job is to take a natural language question from a student and generate an accurate MongoDB aggregation pipeline query using these collections:
- `users`
- `marks`
- `attendances`
- `professorremarks`

💡 SCHEMA:

📘 `users`: `_id`, `email`, `name`, `cg`, `role`, `hasAccess`  
📘 `marks`: `studentId`, `courseId`, `title`, `score`, `maxScore`, `type`, `feedback`, `createdAt`  
📘 `attendances`: `studentId`, `courseId`, `date`, `status`  
📘 `professorremarks`: `studentId`, `courseId`, `text`  

⚠️ RULES:
- Return a VALID JSON-formatted aggregation pipeline (as Python list of dicts).
- **Keys and field names must be wrapped in double quotes**.
- Do NOT include markdown, triple backticks, or explanation text.
- Filter using the given student email.
- Always follow this format 👇

🧾 Example format:
[
  {{
    "$match": {{
      "email": "example@student.com"
    }}
  }},
  {{
    "$project": {{
      "role": 1
    }}
  }}
]

Question: {question}  
Email: {email}  
MongoDB Aggregation Query:
""",
    input_variables=["question", "email"]
)

answer_prompt = PromptTemplate(
    template="""
You are a helpful assistant. Convert the following MongoDB result into a clear human-readable answer.

Question: {question}
Raw Result: {result}

Answer:
""",
    input_variables=["question", "result"]
)
fallback_prompt = PromptTemplate(
    template="""
You are a helpful assistant. Please answer the following question briefly and clearly.

Question: {question}

Answer:
""",
    input_variables=["question"]
)

fallback_chain = fallback_prompt | llm

query_chain = query_prompt | llm
answer_chain = answer_prompt | llm
classify_chain = classify_prompt | llm

# Cache

def load_cache():
    if os.path.exists(cache_file):
        logger.info(f"Loading cache from {cache_file}")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {"questions": [], "embeddings": [], "answers": []}
    logger.info("No cache file found, creating new cache")
    return {"questions": [], "embeddings": [], "answers": []}

cache = load_cache()


def extract_main_topic(query):
    topic = query.split("on ")[-1].strip().lower()
    if topic == query.lower():  # if "on" wasn't found
        topic = query.split("about ")[-1].strip().lower()
    
    return topic
    
def find_similar_question(query, cache, embeddings_model, similarity_threshold=0.85):
    logger.info(f"Checking for semantically similar questions to: {query}")
    
    if not cache["questions"]:
        logger.info("Cache is empty")
        return None, 0
    
    query_topic = extract_main_topic(query)
    logger.info(f"Extracted topic: {query_topic}")
    
    try:
        query_embedding = embeddings_model.embed_query(query)
        
        cache_embeddings = []
        for emb in cache["embeddings"]:
            if isinstance(emb, str):
                emb = json.loads(emb)
            cache_embeddings.append(emb)
        
        best_match_idx = -1
        max_similarity = 0
        
        for i, cached_question in enumerate(cache["questions"]):
            cached_topic = extract_main_topic(cached_question)
            
            if query_topic in cached_topic or cached_topic in query_topic:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(cache_embeddings[i]).reshape(1, -1)
                )[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_idx = i
        
        if best_match_idx >= 0 and max_similarity >= similarity_threshold:
            logger.info(f"Most similar question: '{cache['questions'][best_match_idx]}' with similarity: {max_similarity:.4f}")
            return cache["answers"][best_match_idx], max_similarity
        else:
            logger.info(f"No good match found. Best similarity was {max_similarity:.4f}")
                
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        logger.error(traceback.format_exc())
    
    return None, 0
    

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        query, email = data.get("query"), data.get("email")

        if not query or not email:
            return jsonify({"error": "Query and email are required"}), 400

        classification = classify_chain.invoke({"question": query}).content.strip().lower()
        logger.info(f"Classification: {classification}")

        if classification == "true":
            generated = query_chain.invoke({"question": query, "email": email})
            try:
                mongo_query = json.loads(generated.content.strip())
            except:
                mongo_query = ast.literal_eval(generated.content.strip().replace("```json", "").replace("```", ""))

            all_results = []
            for collection in [user_collection, marks_collection, attendance_collection, remarks_collection]:
                try:
                    results = list(collection.aggregate(mongo_query))
                    if results:
                        all_results.extend(results)
                except:
                    continue

            if not all_results:
                return jsonify({"answer": "No personal records found."})

            response = answer_chain.invoke({
                "question": query,
                "result": json.dumps(all_results, default=str)
            })
            return jsonify({"answer": response.content.strip()})

        if query in cache["questions"]:
            idx = cache["questions"].index(query)
            return jsonify({"answer": cache["answers"][idx], "source": "cache_exact_match"})

        similar_answer, score = find_similar_question(query, cache, embeddings)
        if similar_answer:
            return jsonify({"answer": similar_answer, "source": "cache_similar_match", "similarity": score})

        
        
        response = fallback_chain.invoke({"question": query})
        final_answer = response.content.strip() if response else "No relevant information found."

        query_embedding = embeddings.embed_query(query)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        cache["questions"].append(query)
        cache["embeddings"].append(query_embedding)
        cache["answers"].append(final_answer)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4)

        return jsonify({"answer": final_answer, "source": "fresh_query"})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port)
