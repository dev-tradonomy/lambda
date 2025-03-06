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
from neo4j import GraphDatabase
import re

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
MAX_DAILY_TWEETS = int(os.getenv("MAX_DAILY_TWEETS", 500))
MAX_MONTHLY_TWEETS = int(os.getenv("MAX_MONTHLY_TWEETS", 15000))

# Load Neo4j credentials
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")


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

def fetch_similar_entities(conn, entity_name, table_name, column_name, top_n=1):
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
    return [record[0] for record in results]

def get_users_for_entity(conn, entity_id):
    """Retrieve users watching or holding a given """
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
                        "3": message,
                        "4": user["stock_info"] if user["stock_info"] else "No additional information available."
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
                    "templateName": "updates_template_alpha",
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



class Neo4jConnector:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=300)
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")

    def close(self):
        self.driver.close()

    def query(self, cypher_query, parameters=None):
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"Error running Cypher query: {e}")
            return []


connector = Neo4jConnector(uri, user, password)


def clean_tweet(text):
    text = text.replace("\n", " ").replace("\t", " ")  # Remove newlines and tabs
    text = re.sub(r" {5,}", "    ", text)  # Replace more than 4 spaces with 4 spaces
    return text.strip()


async def process_tweets(conn, tweets):
    """Process tweets, extract entities, find users, and send notifications."""
    if not tweets:
        print("No new tweets to process.")
        return
    
    try:
        cursor = conn.cursor()
        
        for tweet in tweets:
            # Extract tweet details safely
            tweet_id = str(tweet.get("id"))  # Ensure it's a string
            tweet_text = tweet.get("text", "").strip()
            tweet_text = clean_tweet(tweet_text)
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
            matched_entities = set()
            for entity_name in extracted_data["stock"]:
                matched_entities.update(fetch_similar_entities(conn, entity_name, "stock_info", "stock_name"))
                
            for entity_name in extracted_data["mutual_fund"]:
                matched_entities.update(fetch_similar_entities(conn, entity_name, "mutual_fund_info", "mutual_fund"))

            print(f"Matched entities: {matched_entities}")
            # Retrieve users watching these entities


            receivers = set()  # Using a set to ensure uniqueness

            for entity in matched_entities:
                cursor.execute("SELECT id FROM entity WHERE entity_name = %s", (entity,))
                entity_id = cursor.fetchone()
                cursor.execute("SELECT entity_type FROM entity WHERE entity_name = %s", (entity,))
                entity_type = cursor.fetchone()
                entity_type = entity_type[0] if entity_type else None
                cypher_query = None
                if entity_type == 'stock':
                    cypher_query = f"""MATCH (s:Stock) WHERE s.stock_name CONTAINS("{entity}") RETURN s LIMIT 1"""
                if entity_type == 'mutual_fund':
                    cypher_query = f"""MATCH (s:MutualFund) WHERE s.scheme_name CONTAINS("{entity}") RETURN s LIMIT 1"""
                if entity_type == 'index':
                    cypher_query = f"""MATCH (s:Index) WHERE s.index_name CONTAINS("{entity}") RETURN s LIMIT 1"""
                
                stock_info = ""
                if cypher_query:
                    result = connector.query(cypher_query)
                    if result:
                        json_data = result[0]['s']
                        # print(json_data)
                        json_load = json.dumps(json_data)

                        if json_data:
                            stock_info += f""" {entity} has been mentioned in a recent notification!  Here are some details about {entity}: """
                            if entity_type == 'mutual_fund':
                                if json_data['mutual_fund_business_score'] and json_data['mutual_fund_valuation_score']:
                                    stock_info += f""" Biz Score: {json_data['mutual_fund_business_score']*100}% Valuation: {json_data['mutual_fund_valuation_score']*100}%"""
                            if entity_type == 'stock':
                                stock_info += f"""EOD Price: {json_data['eod_price']} Market Cap: {json_data['market_cap']} """
        
                if entity_id:
                    users = get_users_for_entity(conn, entity_id[0])
                    for user_id, phone_number, user_name in users:
                        receivers.add((entity, user_name if user_name else "User", phone_number, stock_info))  # Ensure uniqueness

            # Convert set to list of dictionaries
            receivers_list = [
                {"entity": entity, "user_name": user_name, "phone_number": phone_number, "stock_info": stock_info}
                for entity, user_name, phone_number, stock_info in receivers
            ]

            for receiver in receivers_list:
                conn.execute("SELECT user_id FROM users WHERE phone_number = %s", (phone_number,))
                user = conn.fetchone()
                if not user:
                    raise Exception(status_code=404, detail="User not found")

                user_id = user["user_id"]
                cur.execute("INSERT INTO message_history (user_id, sender, message_text, message_type) VALUES (%s, 'bot', %s, text)",(user_id, message_text),)



            # Send notifications to all the users at once
            await send_bulk_notifications(receivers_list, tweet_text)

        cursor.close()
        print("Tweet processing complete.")
    except Exception as e:
        print(f"Error processing tweets: {e}")


def get_tweet_counts(conn):
    """Fetch today's and this month's tweet counts from the counter table."""
    cursor = conn.cursor()
    today = datetime.utcnow().date()
    current_month = datetime.utcnow().strftime("%Y-%m")

    cursor.execute("""
        SELECT daily_tweet_count, monthly_tweet_count FROM counter 
        WHERE date = %s OR month = %s
    """, (today, current_month))
    
    counts = cursor.fetchall()
    
    daily_count = 0
    monthly_count = 0
    for row in counts:
        if len(row) == 2:
            daily_count, monthly_count = row

    return daily_count, monthly_count


def update_tweet_counts(conn, new_tweets):
    """Update tweet counts after fetching new tweets."""
    cursor = conn.cursor()
    today = datetime.utcnow().date()
    current_month = datetime.utcnow().strftime("%Y-%m")
    new_count = len(new_tweets)  # Number of new tweets fetched

    # Step 1: Try inserting daily entry
    cursor.execute("""
        INSERT INTO counter (date, month, daily_tweet_count, monthly_tweet_count)
        VALUES (%s, %s, %s, 0)
        ON CONFLICT (date) DO NOTHING
        RETURNING date
    """, (today, current_month, new_count))
    daily_inserted = cursor.fetchone()  # Will be None if conflict happened

    # Step 2: Calculate total monthly count if today's entry was newly inserted
    if daily_inserted:
        cursor.execute("""
            SELECT COALESCE(SUM(monthly_tweet_count), 0) 
            FROM counter 
            WHERE month = %s
        """, (current_month,))
        previous_monthly_count = cursor.fetchone()[0]  # Sum of all previous month values
        total_monthly_count = previous_monthly_count + new_count

        # Step 3: Update the newly inserted row with the correct monthly count
        cursor.execute("""
            UPDATE counter 
            SET monthly_tweet_count = %s 
            WHERE date = %s
        """, (total_monthly_count, today))
    else:
        # Step 4: If today's row already existed, just update the counts
        cursor.execute("""
            UPDATE counter 
            SET daily_tweet_count = daily_tweet_count + %s 
            WHERE date = %s
        """, (new_count, today))

        cursor.execute("""
            UPDATE counter 
            SET monthly_tweet_count = monthly_tweet_count + %s 
            WHERE month = %s
        """, (new_count, current_month))

    conn.commit()

async def async_lambda_handler(event, context):
    """AWS Lambda handler function."""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )

        # Fetch current tweet counts
        daily_count, monthly_count = get_tweet_counts(conn)
        print(f"Current tweet counts - Daily: {daily_count}, Monthly: {monthly_count}")

        # Check if limits are exceeded
        if daily_count >= MAX_DAILY_TWEETS:
            print("Daily tweet limit reached. Skipping processing.")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Daily tweet limit reached. No tweets processed."})
            }

        if monthly_count >= MAX_MONTHLY_TWEETS:
            print("Monthly tweet limit reached. Skipping processing.")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Monthly tweet limit reached. No tweets processed."})
            }

        # Fetch new tweets
        # tweets = get_recent_tweets()
        tweets =[{'text': "'I want you to make your own money and not use what I have earned over the years,' JSW Chairman Sajjan Jindal told his Harvard-educated son, who wanted to invest in an EV company. \n\nHe further added that Harsh Goenka and Uday Kotak's sons were smarter than their fathers!'… https://t.co/vKqVyrGrzE https://t.co/HydA28LBjF", 'id': '1897574058218360986', 'edit_history_tweet_ids': ['1897574058218360986'], 'author_id': '631810714', 'created_at': '2025-03-06T09:04:45.000Z'}, {'text': "#MFCorner | @vinnii_motiwala speaks with Shibani Sircar Kurian of Kotak Mahindra AMC and Priti Rathi Gupta of LXME about rising women investors in mutual fund space in this Women's Day special segment.\n#cnbctv18digital #investment #investors #market \n\nWatch here:… https://t.co/iJoNeCfdYz", 'id': '1897569974614638940', 'edit_history_tweet_ids': ['1897569974614638940'], 'author_id': '631810714', 'created_at': '2025-03-06T08:48:31.000Z'}, {'text': 'Midcap Movers | @vamakshidhoria with the big movers in the broader market today. https://t.co/qFUJp3S1hL', 'id': '1897569136181735921', 'edit_history_tweet_ids': ['1897569136181735921'], 'author_id': '631810714', 'created_at': '2025-03-06T08:45:11.000Z'}, {'text': 'Nothing has launched its latest Phone 3a series featuring Phone 3a Pro at ₹29,999. The smartphone is packed with new periscope lens, advanced AI features, and the iconic Glyph interface. @ShibaniGharat with more.  \n\n#nothingphone #3aseries #Glyphinterface #cnbctv18digital… https://t.co/hNzKHSkAwE https://t.co/jxuLcmyP09', 'id': '1897564899905298943', 'edit_history_tweet_ids': ['1897564899905298943'], 'author_id': '631810714', 'created_at': '2025-03-06T08:28:21.000Z'}, {'text': 'Company: Nestle\n\nUpdate Type: Press Release 📰 | Sentiment: Positive 🟢\n\nSummary: Nespresso opens its first boutique in New Delhi, marking a major expansion in India with significant focus on sustainability and premium coffee experience.', 'id': '1897562650802053212', 'edit_history_tweet_ids': ['1897562650802053212'], 'author_id': '1864603590201110528', 'created_at': '2025-03-06T08:19:25.000Z'}, {'text': 'Company: Maan Aluminium\n\nUpdate Type: Acquisition 🛒\n\n📦Acquired Company: Refer Filing\n\n💼Business Overview: Refer Filing\n\n📊Percentage Acquired: Refer Filing\n\n💰Total Consideration Paid: Rs. 8.75 Crs excluding stamp duty and other charges', 'id': '1897561435393401330', 'edit_history_tweet_ids': ['1897561435393401330'], 'author_id': '1864603590201110528', 'created_at': '2025-03-06T08:14:35.000Z'}, {'text': '#Samsung begins the rollout of #Android15 based One UI 7 for older Galaxy devices. Check if your Galaxy device is eligible or not.\n\n@pihuyadav05 @SamsungIndia @SamsungMobile\n\nhttps://t.co/e3CAedqoIO', 'id': '1897561379973816326', 'edit_history_tweet_ids': ['1897561379973816326'], 'author_id': '631810714', 'created_at': '2025-03-06T08:14:22.000Z'}, {'text': "#Apple's rumoured foldable #iPhone might launch in 2026 with a $2,000 price tag, says analyst @mingchikuo\n\nHere are the details | @pihuyadav05\n@apple\n\nhttps://t.co/tRq7r2paSE", 'id': '1897560748852715770', 'edit_history_tweet_ids': ['1897560748852715770'], 'author_id': '631810714', 'created_at': '2025-03-06T08:11:52.000Z'}, {'text': 'Company: Thomas Cook (India)\n\nUpdate Type: Press Release 📰 | Sentiment: Positive 🟢\n\nSummary: Thomas Cook India and SOTC report a 35% growth in demand from female travelers, emphasizing adventure, wellness, and milestone travel. This highlights potential revenue enhancement and… https://t.co/Lvz1q88TQb', 'id': '1897559419027964191', 'edit_history_tweet_ids': ['1897559419027964191'], 'author_id': '1864603590201110528', 'created_at': '2025-03-06T08:06:35.000Z'}]
        # print(tweets)
        new_tweet_count = len(tweets)
        print(f"Fetched {new_tweet_count} new tweets.")

        if new_tweet_count == 0:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No new tweets to process."})
            }

        # Update tweet counts
        update_tweet_counts(conn, tweets)

        # Process tweets
        await process_tweets(conn, tweets)
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Tweets fetched, processed, and notifications sent."})
        }
    
    except Exception as e:
        print(f"Error in Lambda Handler: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
    finally:
        conn.commit()
        conn.close()

# Run the Lambda handler locally
import asyncio
if __name__ == "__main__":
    asyncio.run(async_lambda_handler({}, {}))


# def lambda_handler(event, context):
#     """AWS Lambda synchronous entry point."""
#     return asyncio.run(async_lambda_handler(event, context))