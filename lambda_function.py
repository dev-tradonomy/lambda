import os
import requests
import psycopg2
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import httpx
from itertools import islice

# Environment variables
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", 5432)
USER_IDS = os.getenv("TWITTER_USER_IDS", "").split(",")  # Comma-separated user IDs
USER_NAMES = os.getenv("TWITTER_USER_NAMES", "").split(",")  # Comma-separated user names
USER_MAP = dict(zip(USER_IDS, USER_NAMES))  # Map user_id to user_name
X_MINUTES = int(os.getenv("X_MINUTES", 30))  # Default to last 30 minutes
TWEET_COUNT = 10  # Fetch up to 10 tweets
GALLABOX_BASE_URL = os.getenv("GALLABOX_BASE_URL")
GALLABOX_API_KEY = os.getenv("GALLABOX_API_KEY")
GALLABOX_API_SECRET = os.getenv("GALLABOX_API_SECRET")
TRADONOMY_CHANNEL_ID = os.getenv("TRADONOMY_CHANNEL_ID")

# AI Variables
class AI_Variables:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-4o-mini"
    LANGCHAIN_MODEL = "gpt-4o-mini"

class ExtractInfo(BaseModel):
    stock: list[str] = Field(default=[], description="List of stock names to be searched in the database.")
    mutual_fund: list[str] = Field(default=[], description="List of mutual fund names to be searched in the database.")
    
def extracting_data(user_message:str):
    try:
        if not AI_Variables.OPENAI_API_KEY:
            raise ValueError("OpenAI API Key is missing or None.")

        print("Using Model:", AI_Variables.MODEL_NAME)
        print("Using OpenAI API Key:", AI_Variables.OPENAI_API_KEY[:5] + "****")  # Mask key for security
        
        tagging_prompt = ChatPromptTemplate.from_template(
            """
                Extract the following details from the user's input if available:
                - Stock names (as a list)
                - Mutual fund names (as a list)
                
                **Important Notes:**
                - If the user specifies a company name (e.g., "Apple," "Tesla"), classify it under **stock**.
                
                Respond **only** in JSON format:
                {{
                    "stock": ["<stock name>", "<stock name>", ...] or [],
                    "mutual_fund": ["<mutual fund name>", "<mutual fund name>", ...] or [],
                }}

                User Input:
                "{user_message}"
            """
        )
        llm = ChatOpenAI(
            temperature=0,
            model=AI_Variables.LANGCHAIN_MODEL,
            openai_api_key=AI_Variables.OPENAI_API_KEY
        ).with_structured_output(ExtractInfo, method="function_calling", include_raw=True)

        prompt_str = tagging_prompt.format(user_message=user_message)
        
        # Invoke model
        response = llm.invoke(prompt_str)

        response_dict = response['parsed'].model_dump()
        # Extract values correctly
        stock = response_dict["stock"]
        mutual_fund = response_dict["mutual_fund"]
        
        # calculate_langchain_cost(response)

        print("Extracted Data:", response_dict)
        
        return response_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extracting Data Breaked: {e}")

def get_ai_client():
    ai_model = AI_Variables.MODEL_NAME

    client_openai = OpenAI(api_key= AI_Variables.OPENAI_API_KEY) 
    client = client_openai

    return client

def generate_embedding(text: str):
    """Generates an embedding for a given stock name."""
    try:
        client = get_ai_client()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        # calculate_embedding_cost(response)
        embedding = response.data[0].embedding  # Extract embedding vector
        return embedding
    except Exception as e:
        print(f"❌ Error generating embedding for '{text}': {e}")
        return None  # Return None to handle errors gracefully
        

def get_recent_tweets():
    """Fetch recent tweets from multiple user IDs."""
    start_time = (datetime.utcnow() - timedelta(minutes=X_MINUTES)).strftime("%Y-%m-%dT%H:%M:%SZ")
    user_query = " OR ".join([f"from:{user_id}" for user_id in USER_IDS])
    url = f"https://api.twitter.com/2/tweets/search/recent?max_results={TWEET_COUNT}&query={user_query}&start_time={start_time}&tweet.fields=created_at,text,author_id"
    
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Error fetching tweets: {response.status_code}, {response.text}")
        return []

def is_tweet_stored(conn, tweet_id):
    """Check if a tweet is already in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM tweets WHERE tweet_id = %s", (tweet_id,))
    result = cursor.fetchone()
    return result is not None  # True if tweet exists, False otherwise

def fetch_similar_entities(conn, entity_name, table_name, column_name, top_n=3):
    """Find similar entities based on vector similarity."""
    cursor = conn.cursor()
    embedding = generate_embedding(entity_name)
    if not embedding:
        return []

    embedding_array = f"[{','.join(map(str, embedding))}]"
    query = f"""
        SELECT {column_name}, embedding <-> %s::vector AS distance
        FROM {table_name}
        ORDER BY distance ASC
        LIMIT %s;
    """
    cursor.execute(query, (embedding_array, top_n))
    results = cursor.fetchall()
    return [record[0] for record in results if abs(record[1] - results[0][1]) < 0.04]

def get_users_for_entity(conn, entity_id):
    """Retrieve users watching or holding a given entity."""
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT user_id, phone_number, name
        FROM (
            SELECT u.user_id, u.phone_number, u.name
            FROM watchlist_entity w
            JOIN watchlists ws ON w.watchlist_id = ws.watchlist_id
            JOIN users u ON ws.user_id = u.user_id
            WHERE w.entity_id = %s
            UNION
            SELECT u.user_id, u.phone_number, u.name
            FROM portfolio_holdings ph
            JOIN portfolios p ON ph.portfolio_id = p.portfolio_id
            JOIN users u ON p.user_id = u.user_id
            WHERE ph.entity_id = %s
        ) AS user_data;

    """
    cursor.execute(query, (entity_id, entity_id))
    return cursor.fetchall()  # Returns [(user_id, phone_number), ...]

def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

async def send_bulk_notifications(receivers_list, message):
    """
    Sends notifications in batches of 100 users.
    
    receivers_list: List of dicts -> [{"entity": entity, "user_name": user_name, "phone_number": phone_number}, ...]
    message: The notification message to be sent.
    """
    batch_size = 100  # Process 100 users per batch
    headers = {
        "Content-Type": "application/json",
        "apiKey": GALLABOX_API_KEY,
        "apiSecret": GALLABOX_API_SECRET
    }

    for batch in chunk_list(receivers_list, batch_size):
        recipient_data = [
            {
                "name": user["user_name"] if user["user_name"] else "User",
                "phone": user["phone_number"],
                "templateValues": {
                    "bodyValues": {
                        "1": user["user_name"] if user["user_name"] else "User",
                        "2": user["entity"],
                        "3": message
                    }
                }
            }
            for user in batch
        ]

        payload = {
            "channelId": TRADONOMY_CHANNEL_ID,
            "channelType": "whatsapp",
            "recipientData": recipient_data,
            "whatsapp": {
                "type": "template",
                "template": {
                    "templateName": "updates_template",
                    "bodyValues": {
                        "1": "User Name",
                        "2": "Stock Name",
                        "3": message
                    }
                }
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(GALLABOX_BASE_URL, headers=headers, data=json.dumps(payload))
                print("Response:", response.json())
        except Exception as e:
            print("Error occurred:", e)

async def process_tweets(tweets):
    """Process tweets, extract entities, find users, and send notifications."""
    if not tweets:
        print("No new tweets to process.")
        return
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        cursor = conn.cursor()
        
        print(type(tweets))
        for tweet in tweets:
            print(tweet)
            print(type(tweet))
            # Extract tweet details safely
            tweet_id = str(tweet.get("id"))  # Ensure it's a string
            tweet_text = tweet.get("text", "").strip()
            user_id = tweet.get("author_id", "")
            user_name = USER_MAP.get(user_id, "Unknown")
            created_at = tweet.get("created_at", "")

            # Skip tweets with missing crucial information
            if not tweet_id or not tweet_text or not user_id:
                print(f"Skipping invalid tweet: {tweet}")
                continue

            # Debugging: Print each tweet being processed
            print(f"Processing tweet {tweet_id} from user {user_name}")

            # Check if the tweet is already stored
            if is_tweet_stored(conn, tweet_id):
                print(f"Skipping already processed tweet: {tweet_id}")
                continue

            # Store tweet in DB
            cursor.execute(
                """
                INSERT INTO tweets (tweet_id, user_id, user_name, text, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING;
                """,
                (tweet_id, user_id, user_name, tweet_text, tweet["created_at"])
            )

            # Extract relevant entities
            extracted_data = extracting_data(tweet_text)
            entities_to_check = extracted_data["stock"] + extracted_data["mutual_fund"]
            matched_entities = set()

            for entity_name in entities_to_check:
                # Find similar stocks & mutual funds
                matched_entities.update(fetch_similar_entities(conn, entity_name, "stock_info", "stock_name"))
                matched_entities.update(fetch_similar_entities(conn, entity_name, "mutual_fund_info", "mutual_fund"))

            print(f"Matched entities: {matched_entities}")
            # Retrieve users watching these entities

            receivers = set()  # Using a set to ensure uniqueness

            for entity in matched_entities:
                cursor.execute("SELECT id FROM entity WHERE entity_name = %s", (entity,))
                entity_id = cursor.fetchone()
                
                if entity_id:
                    users = get_users_for_entity(conn, entity_id[0])
                    for user_id, phone_number, user_name in users:
                        receivers.add((entity, user_name if user_name else "User", phone_number))  # Ensure uniqueness

            # Convert set to list of dictionaries
            receivers_list = [
                {"entity": entity, "user_name": user_name, "phone_number": phone_number}
                for entity, user_name, phone_number in receivers
            ]
            
            # Send notifications to all the users at once
            await send_bulk_notifications(receivers_list, tweet_text)

        conn.commit()
        cursor.close()
        conn.close()
        print("Tweet processing complete.")
    except Exception as e:
        print(f"Error processing tweets: {e}")



async def async_lambda_handler(event, context):
    """AWS Lambda handler function."""
    tweets = get_recent_tweets()
    print(tweets)
    await process_tweets(tweets)
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Tweets fetched, processed, and notifications sent."})
    }


import asyncio

def lambda_handler(event, context):
    """AWS Lambda synchronous entry point."""
    return asyncio.run(async_lambda_handler(event, context))