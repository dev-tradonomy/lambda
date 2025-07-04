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
import httpx
from neo4j import GraphDatabase
import re
import asyncio
from bs4 import BeautifulSoup
import requests
from html import unescape
import time
import math
import yfinance as yf # peewee, multitasking, websockets, tzdata, pycparser, protobuf, platformdirs, frozendict, pandas, cffi, curl_cffi, yfinance

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
    OPENAI_API_KEY = ""
    MODEL_NAME = ""
    LANGCHAIN_MODEL = "gpt-4.1-mini"

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
                Extract the following financial entities from the user's input, if mentioned:
                - Stock names (as a list of company names or ticker symbols)
                - Mutual fund names (as a list)
                
                **Important Notes:**
                1. If the user specifies a company name (e.g., "Apple," "Tesla"), classify it under **stock**.

                2. If an entity isn’t clearly a stock name or fund, **leave it out**. Be conservative — prefer empty lists over incorrect guesses.

                3. Do not infer entities. Only extract names if they are explicitly mentioned.

                4. Never classify general financial adjectives or investment categories under "stock".

                5. Only extract **concrete, real-world entities** — ignore vague descriptors or categories like:
                - Market cap terms: "large-cap", "mid-cap", "small-cap"
                - Quality types: "blue-chip", "growth stock", "value stock", "penny stock"
                - Risk terms: "low-risk", "high-risk", "safe bet"
                - Investment strategies: "momentum stocks", "undervalued stocks"
                - General labels: "top stocks", "best performers", "rising stars", "ace investors", "big investors", "indices", etc.
                
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
            WHERE w.entity_id = %s AND u.profile_status = 'active'
            
            UNION
            
            SELECT u.user_id, u.phone_number, u.name
            FROM portfolio_holdings ph
            JOIN portfolios p ON ph.portfolio_id = p.portfolio_id
            JOIN users u ON p.user_id = u.user_id
            WHERE ph.entity_id = %s AND u.profile_status = 'active'
        ) AS user_data;
    """
    cursor.execute(query, (entity_id, entity_id))
    return cursor.fetchall()  # Returns [(user_id, phone_number), ...]


def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

async def send_bulk_notifications(conn, receivers_list, message, link):
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

    cursor = conn.cursor()
    cursor.execute("SELECT templates FROM variables WHERE id = 1")
    row = cursor.fetchone()
    if not row:
        template_name="latest_updates_v2"
        template_variables=['1','2','3','4']
        print("No template found.")
    else:
        template_data = row[0]
        smart_template = next((t for t in template_data['templates'] if t['type'] == 'latest_template'), None)
        if smart_template:
            template_name = smart_template['name']
            template_variables = smart_template['variables']
        else:
            template_name="latest_updates_v2"
            template_variables=['1','2','3','4']
        

    for batch in chunk_list(receivers_list, batch_size):
        recipient_data = [
            {
                "name": user["user_name"] if user["user_name"] else "User",
                "phone": user["phone_number"],
                "templateValues": {
                    "bodyValues": {
                        template_variables[0]: user["entity"],
                        template_variables[1]: message,
                        template_variables[2]: link,
                        template_variables[3]: user["stock_info"] if user["stock_info"] else "No additional information available."
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
                    "templateName": template_name,
                    "bodyValues": {
                        template_variables[0]: "Stock Name",
                        template_variables[1]: message,
                        template_variables[2]: link,
                        template_variables[3]: "Stock Info"
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


def clean_text(text):
    try:
        text = text.replace("\n", " ").replace("\t", " ")  # Remove newlines and tabs
        text = re.sub(r" {5,}", "    ", text)  # Replace more than 4 spaces with 4 spaces
        return text.strip()
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""


def storing_tweets(conn, receivers_list, tweet_text, link):
    try:
        cursor = conn.cursor()
        created_at = int(datetime.now().timestamp() * 1000)
        for receiver in receivers_list:
            cursor.execute("SELECT user_id FROM users WHERE phone_number = %s", (receiver["phone_number"],))
            user = cursor.fetchone()
            user = user[0]
            if not user:
                raise Exception(status_code=404, detail="User not found")

            entity_id = receiver["entity_id"][0]

            # Store message history
            tweet_message = f"""
                🔔 There's an update on {receiver['entity']}:, 

                ℹ️ Update: {tweet_text}

                🔗 Source: {link}

                📈 Summary: {receiver['stock_info']}

                🤖A.I. powered update by tradonomy.ai
            """
                            
            user_id = user
            cursor.execute(
                "INSERT INTO message_history (user_id, sender, message, message_type) VALUES (%s, 'bot', %s, %s)",
                (user_id, tweet_message, 'text')  
            )
            print(f"Message history stored for user {user_id}, message: {tweet_text}")

            # created_at = int(datetime.utcnow().timestamp() * 1000) 

            cursor.execute(
                "INSERT INTO notifications (user_id, entity_id, message, created_at, status, category) VALUES (%s, %s, %s, %s, 'sent', 'Latest Updates')",
                (user_id, entity_id, tweet_message, created_at)
            )
            conn.commit()
    except Exception as e:
        print(f"Error storing tweets: {e}")
        return 
    finally:
        cursor.close()


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
            print(f"\nExtracting entities from tweet: {tweet_text}\n")
            extracted_data = extracting_data(tweet_text)
            print(f"Extracted Data: {extracted_data}")
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
                if not entity_id:
                    print(f"Entity '{entity}' not found in database. Skipping.")
                    continue
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

                            stock_noti = f"""{entity} has been mentioned in a recent notification! 
                                Tweet Text: {tweet_text}"""
                            if entity_type == 'stock':
                                if json_data.get('biz_score_percent') is not None and json_data.get('valuation_score_percent') is not None and json_data.get('trend_score_percent') is not None:
                                    stock_noti += f""" Biz Score: {json_data['biz_score_percent']*100}% Valuation Score: {json_data['valuation_score_percent']*100}%  Trend Score: {json_data['trend_score_percent']*100}%"""
                            if entity_type == 'mutual_fund':
                                if json_data.get('mutual_fund_business_score') is not None and json_data.get('mutual_fund_valuation_score') is not None:
                                    stock_noti += f""" Biz Score: {json_data['mutual_fund_business_score']*100}% Valuation: {json_data['mutual_fund_valuation_score']*100}%"""
                            messages = [
                                {"role": "system", "content": """
                                    You are a concise and actionable stock-analysis assistant. You will receive:
                                    1. A Biz Score (0 - 100%).
                                    2. A Valuation Score (0 - 100%).
                                    3. A Trend Score (0 - 100%).  
                                    IMPORTANT NOTE about Valuation Score:
                                    - Higher Valuation Score means more UNDERVALUED.
                                    - Lower Valuation Score means more OVERVALUED.
                                    (For example, Valuation Score = 70% ⇒ the stock is relatively undervalued; 30% ⇒ the stock is relatively overvalued.)

                                    Trend Score:
                                    - Higher Trend Score means more trending the stock is .


                                    Your task:
                                    - Output a single short interpretation, in your own words, based on that condition. Also mention Biz Score, Valuation Score, and Trend Score in your response.
                                    
                                    Make sure your interpretation:
                                    - Is short, direct, and actionable.
                                    - Reflects that a higher Biz Score indicates stronger fundamentals, and a higher Valuation Score indicates more undervaluation.

                                    If the Biz Score or Valuation Score or Trend Score is not available, please do not mention then it in your response and do not ask for them. Do the analysis without them.

                                    Analyse the impact of the tweet text on the stocks fundamentals by also assessing the scores provided and to give the A.I. summary. 
                                    Remember to avoid any analysis when not required, such as tweet text that doesn't mention any crucial company information.

                                """
                                },
                                {"role": "user", "content": stock_noti}
                            ]
                            ai_client = get_ai_client()
                            response =  ai_client.chat.completions.create(
                                model=AI_Variables.MODEL_NAME,
                                messages=messages
                            )

                            info = response.choices[0].message.content
                            info = clean_text(info)
                            stock_info += f"""{info}"""
                            
                            if entity_type == 'stock':
                                if json_data.get('eod_price') and json_data.get('market_cap'):
                                    stock_info += f"""EOD Price: {json_data['eod_price']:.2f}, Market Cap: {json_data['market_cap']:.2f}"""
        
                print(f"\nStock info for {entity}: {stock_info}\n")
                # Store message history
                tweet_message = f"""
                    🔔 There's an update on {entity}:, 

                    ℹ️ Update: {tweet_text}

                    🔗 Source: {tweet['link']}

                    📈 Summary: {stock_info}

                    🤖A.I. powered update by tradonomy.ai
                """
                cursor.execute(
                    "INSERT INTO stock_notifications (entity_id, message, status, category) VALUES (%s, %s, 'sent', 'Latest Updates')",
                    (entity_id, tweet_message)
                )
                conn.commit()
                if entity_id:
                    users = get_users_for_entity(conn, entity_id[0])
                    for user_id, phone_number, user_name in users:
                        receivers.add((entity, user_name if user_name else "User", phone_number, stock_info, entity_id))  # Ensure uniqueness

            # Convert set to list of dictionaries
            receivers_list = [
                {"entity": entity, "user_name": user_name, "phone_number": phone_number, "stock_info": stock_info, "entity_id": entity_id}
                for entity, user_name, phone_number, stock_info, entity_id in receivers
            ]

            print(f"\nReceivers list: {receivers_list}\n") 
            # Send notifications to all the users at once
            if receivers_list != []:
                storing_tweets(conn, receivers_list, tweet_text, tweet['link'])
                await send_bulk_notifications(conn, receivers_list, tweet_text, tweet['link'])

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

def should_run_sync():
    """Run sync only at 12:00 AM and 12:00 PM UTC."""
    now = datetime.utcnow()
    # Check if current time is between 12:00:00 and 12:14:59 (AM or PM)
    return (now.hour == 0 or now.hour == 12) and now.minute < 15

def fetch_neo4j_stock_data():
    """Fetch stock data from Neo4j."""
    try: 
        # Implement the logic to connect to Neo4j and fetch stock data
     
        cypher_query = """
            MATCH (s:Stock)
            RETURN DISTINCT s.isin AS isin, s.stock_name AS stock_name, s.nse_code AS nse_code, s.bsecode AS bse_code
        """
        results = connector.query(cypher_query)
        print(len(results))

        return results
    except Exception as e:
        print(f"Error fetching Neo4j stock data: {e}")
        return []

def fetch_neo4j_mutual_fund_data():
    """Fetch mutual fund data from Neo4j."""
    try:
        # Implement the logic to connect to Neo4j and fetch mutual fund data
        cypher_query = """
            MATCH (s:MutualFund)
            WHERE NOT s.sd_scheme_isin STARTS WITH "XX"
            WITH s.scheme_name AS mf_name, collect(s.sd_scheme_isin)[0] AS isin
            RETURN DISTINCT mf_name, isin
        """
        results = connector.query(cypher_query)
        print(len(results))

        return results
    except Exception as e:
        print(f"Error fetching Neo4j mutual fund data: {e}")
        return []

def upsert_postgres_mutual_fund(conn, mutual_fund_data):    
    """Efficiently upsert mutual fund data into Postgres."""
    try:
        cur = conn.cursor()

        # Fetch existing mutual funds
        cur.execute("SELECT id, isin, mutual_fund FROM mutual_fund_info")
        rows = cur.fetchall()
        existing_isin_map = {row[1]: (row[0], row[2]) for row in rows}
        existing_name_map = {row[2]: (row[0], row[1]) for row in rows}

        to_insert = []
        to_update_isin = []
        to_update_name = []

        for d in mutual_fund_data:
            mf_name = d.get('mf_name', '')
            isin = d.get('isin', '')

            if mf_name in existing_name_map:
                db_id, db_isin = existing_name_map[mf_name]
                if db_isin != isin:
                    # Mutual fund name exists but ISIN changed, update ISIN
                    to_update_isin.append((isin, db_id))
                    print(f"🔄 Queue ISIN update for {mf_name}")

            elif isin in existing_isin_map:
                db_id, db_name = existing_isin_map[isin]
                if db_name != mf_name:
                    # ISIN exists but mutual fund name changed, update mutual fund name
                    embedding = generate_embedding(mf_name)
                    if not embedding:
                        continue
                    embedding_array = "[" + ",".join(map(str, embedding)) + "]"
                    to_update_name.append((mf_name, embedding_array, db_id))
                    print(f"🔄 Queue mutual fund name update for {isin}")

            else:
                # New ISIN, insert new row
                embedding = generate_embedding(mf_name)
                if not embedding:
                    continue
                embedding_array = "[" + ",".join(map(str, embedding)) + "]"
                to_insert.append((mf_name, embedding_array, isin))
                print(f"➕ Queue insert: {mf_name} / {isin}")

        # Batch insert
        if to_insert:
            args_str = ','.join(cur.mogrify("(%s, %s::vector, %s)", x).decode() for x in to_insert)
            cur.execute(
                f"INSERT INTO mutual_fund_info (mutual_fund, embedding, isin) VALUES {args_str}"
            )

        # Batch update (still row by row, but only for those that changed)
        for isin, db_id in to_update_isin:
            cur.execute(
            """
            UPDATE mutual_fund_info
            SET isin = %s
            WHERE id = %s
            """,
            (isin, db_id)
        )

        for mf_name, embedding_array, db_id in to_update_name:
            cur.execute(
            """
            UPDATE mutual_fund_info
            SET mutual_fund = %s, embedding = %s::vector
            WHERE id = %s       
            """,
            (mf_name, embedding_array, db_id)
        )
        conn.commit()
        print("✅ Mutual fund data batch upsert completed.")    
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def upsert_postgres_entity(conn, stock_data, mutual_fund_data):
    """Upsert stock and mutual fund entities into Postgres."""
    try:
        cur = conn.cursor()

        # Prepare stock and mutual fund entities
        stock_entities = [(d['isin'], d['stock_name'], 'stock') for d in stock_data if d.get('isin')]
        mutual_fund_entities = [(d['isin'], d['mf_name'], 'mutual_fund') for d in mutual_fund_data if d.get('isin')]
        all_entities = stock_entities + mutual_fund_entities

        # Fetch existing entities
        cur.execute("SELECT id, entity_name, isin, entity_type FROM entity")
        rows = cur.fetchall()
        name_type_map = {(row[1], row[3]): (row[0], row[2]) for row in rows if row[1] and row[3]}
        isin_map = {row[2]: (row[0], row[1], row[3]) for row in rows if row[2]}

        to_insert = []
        to_update_isin = []
        to_update_name_type = []

        for isin, name, entity_type in all_entities:
            key = (name, entity_type)
            if key in name_type_map:
                db_id, db_isin = name_type_map[key]
                if db_isin != isin:
                    to_update_isin.append((isin, db_id))
                    print(f"🔄 Queue ISIN update for {name} / {isin}")
            elif isin in isin_map:
                db_id, db_name, db_entity_type = isin_map[isin]
                if db_name != name or db_entity_type != entity_type:
                    to_update_name_type.append((name, entity_type, db_id))
                    print(f"🔄 Queue name update for {isin} / {name}")
            else:
                to_insert.append((name, isin, entity_type))
                print(f"➕ Queue insert: {name} / {isin}")

        # Batch insert
        if to_insert:
            args_str = ','.join(cur.mogrify("(%s, %s, %s)", x).decode() for x in to_insert)
            cur.execute(
                f"INSERT INTO entity (entity_name, isin, entity_type) VALUES {args_str}"
            )

        # Batch update ISIN
        for isin, db_id in to_update_isin:
            cur.execute(
                """
                UPDATE entity
                SET isin = %s
                WHERE id = %s
                """,
                (isin, db_id)
            )

        # Batch update name/type
        for name, entity_type, db_id in to_update_name_type:
            cur.execute(
                """
                UPDATE entity
                SET entity_name = %s, entity_type = %s
                WHERE id = %s
                """,
                (name, entity_type, db_id)
            )

        conn.commit()
        print("✅ Entity data batch upsert completed.")
    except Exception as e:
        print(f"❌ Database connection error: {e}")

def upsert_postgres_stock(conn, stock_data):
    """Efficiently upsert stock data into Postgres."""
    try:
        cur = conn.cursor()

        # Fetch existing stocks
        cur.execute("SELECT id, isin, stock_name FROM stock_info")
        rows = cur.fetchall()
        existing_isin_map = {row[1]: (row[0], row[2]) for row in rows}
        existing_name_map = {row[2]: (row[0], row[1]) for row in rows}

        to_insert = []
        to_update_isin = []
        to_update_name = []

        for d in stock_data:
            stock_name = d.get('stock_name', '')
            nse_code = d.get('nse_code', '')
            isin = d.get('isin', '')

            if nse_code and nse_code != 'None':
                combined_input = f"{nse_code}, {stock_name}"
            else:
                combined_input = stock_name

            # embedding = generate_embedding(combined_input)
            # if not embedding:
            #     continue
            # embedding_array = "[" + ",".join(map(str, embedding)) + "]"

            if stock_name in existing_name_map:
                db_id, db_isin = existing_name_map[stock_name]
                if db_isin != isin:
                    # Stock name exists but ISIN changed, update ISIN
                    to_update_isin.append((isin, db_id))
                    print(f"🔄 Queue ISIN update for {stock_name}")

            elif isin in existing_isin_map:
                db_id, db_name = existing_isin_map[isin]
                if db_name != stock_name:
                    # ISIN exists but stock name changed, update stock name
                    embedding = generate_embedding(combined_input)
                    if not embedding:
                        continue
                    embedding_array = "[" + ",".join(map(str, embedding)) + "]"
                    to_update_name.append((stock_name, embedding_array, db_id))
                    print(f"🔄 Queue stock name update for {isin}")

            else:
                # New ISIN, insert new row
                embedding = generate_embedding(combined_input)
                if not embedding:
                    continue
                embedding_array = "[" + ",".join(map(str, embedding)) + "]"
                to_insert.append((stock_name, embedding_array, isin))
                print(f"➕ Queue insert: {stock_name} / {isin}")

        # Batch insert
        if to_insert:
            args_str = ','.join(cur.mogrify("(%s, %s::vector, %s)", x).decode() for x in to_insert)
            cur.execute(
                f"INSERT INTO stock_info (stock_name, embedding, isin) VALUES {args_str}"
            )

        # Batch update (still row by row, but only for those that changed)
        for isin, db_id in to_update_isin:
            cur.execute(
            """
            UPDATE stock_info
            SET isin = %s
            WHERE id = %s
            """,
            (isin, db_id)
        )

        for stock_name, embedding_array, db_id in to_update_name:
            cur.execute(
            """
            UPDATE stock_info
            SET stock_name = %s, embedding = %s::vector
            WHERE id = %s
            """,
            (stock_name, embedding_array, db_id)
        )

        conn.commit()
        print("✅ Stock embedding data batch upsert completed.")

    except Exception as e:
        print(f"❌ Database connection error: {e}")

def get_ticker_list_from_neo4j():
    stocks = fetch_neo4j_stock_data()
    ticker_list = []
    for stock in stocks:
        nse_code = stock.get("nse_code")
        bse_code = stock.get("bse_code")
        if nse_code and nse_code != "None":
            ticker_list.append(f"{nse_code}.NS")
        elif bse_code and bse_code != "None":
            # Convert float bse_code to int, then to str, to avoid .0 in ticker
            try:
                bse_code_str = str(int(float(bse_code)))
            except Exception:
                bse_code_str = str(bse_code)
            ticker_list.append(f"{bse_code_str}.BO")
    return ticker_list

def fetch_current_prices_batchwise(ticker_list, batch_size=100, sleep_time=2):
    """
    Fetch current prices for a large ticker list in batches.
    :param ticker_list: List of ticker symbols.
    :param batch_size: Number of tickers per batch.
    :param sleep_time: Seconds to sleep between batches.
    :return: Dict of {ticker: price}
    """
    all_prices = {}
    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i:i+batch_size]
        data = yf.download(batch, period="1d", interval="1m", group_by='ticker', threads=True)
        if len(batch) == 1:
            last_row = data.iloc[-1]
            price = last_row['Close']
            if not math.isnan(price):
                all_prices[batch[0]] = round(float(price) * 100) / 100 # Round to 2 decimal places
        else:
            for ticker in batch:
                try:
                    last_row = data[ticker].iloc[-1]
                    price = last_row['Close']
                    if not math.isnan(price):
                        all_prices[ticker] = round(float(price) * 100) / 100 # Round to 2 decimal places
                except Exception as e:
                    pass  # Skip tickers with errors or missing data
        print(f"Fetched prices for batch {i//batch_size + 1}/{(len(ticker_list) + batch_size - 1) // batch_size}")
        time.sleep(sleep_time)  # polite delay to avoid rate limits
    return all_prices

def update_bulk_prices_in_neo4j(price_data):
    """
    Update current prices in Neo4j for all tickers in bulk.
    :param price_data: Dict of {ticker: price}
    """
    updates = []
    for ticker, price in price_data.items():
        # Remove .NS/.BO for matching with nse_code/bse_code in Neo4j
        if '.' in ticker:
            code, suffix = ticker.split(".")
        else:
            code, suffix = ticker, ""
        updates.append({
            "code": code,
            "suffix": suffix,
            "current_price": price
        })

    cypher_query = """
    UNWIND $updates AS update
    MATCH (s:Stock)
    WHERE (update.suffix = 'NS' AND s.nse_code = update.code)
       OR (update.suffix = 'BO' AND s.bsecode = update.code)
    SET s.eod_price = update.current_price,
        s.updated_at = datetime({timezone: 'Asia/Kolkata'}).epochMillis
    """
    connector.query(cypher_query, {"updates": updates})
    print(f"✅ Updated current prices in Neo4j for {len(price_data)} tickers.")

def parse_rss_to_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml-xml") 

        output = []

        for item in soup.find_all("item"):
            title = item.title.text if item.title else ""
            link = item.link.text if item.link else ""
            pub_date = item.pubDate.text if item.pubDate else ""
            author = "Unknown"
            if url == "https://rss.app/feeds/Zwny9tJJ2sbTM5Xn.xml":
                author = "CNBCTV18Live"
            elif url == "https://rss.app/feeds/yFpZIEo9p3B8H30p.xml":
                author = "marketalertsz"
            elif url == "https://rss.app/feeds/7sMGzrCLFpAQ79bh.xml":
                author = "tradonomy"

            entry_data = {
                "author_id": author,
                "created_at": pub_date,
                "id": link.split("/")[-1],
                "link": link,
                "text": unescape(title).strip(),
                "username": author
            }

            output.append(entry_data)

        return output
    except Exception as e:
        print(f"Error parsing RSS feed: {e}")
        return []

async def async_lambda_handler(event, context):
    """AWS Lambda handler function."""
    try:

        rss_urls = [
            "https://rss.app/feeds/Zwny9tJJ2sbTM5Xn.xml","https://rss.app/feeds/yFpZIEo9p3B8H30p.xml","https://rss.app/feeds/7sMGzrCLFpAQ79bh.xml"
        ]
        tweets_from_rss_feed = []
        for url in rss_urls:
            try:
                result = parse_rss_to_json(url)
                tweets_from_rss_feed.extend(result)
                if result:
                    print(f"Data from {url}:\n{result}\n")
            except Exception as e:
                raise Exception(f"An error occurred while processing {url}: {e}")
        
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )

        # Create a cursor
        cur = conn.cursor()

        # Execute query to get entry with id = 1
        cur.execute("SELECT * FROM variables WHERE id = %s;", (1,))

        # Fetch the row
        row = cur.fetchone()

        # Check and print result
        if not row:
            print("No entry found with id = 1")

        AI_Variables.MODEL_NAME = row[1]
        AI_Variables.OPENAI_API_KEY = row[2]

        conn.commit()

        # Fetch current tweet counts
        # daily_count, monthly_count = get_tweet_counts(conn)
        # print(f"Current tweet counts - Daily: {daily_count}, Monthly: {monthly_count}")

        # Check if limits are exceeded
        # if daily_count >= MAX_DAILY_TWEETS:
        #     print("Daily tweet limit reached. Skipping processing.")
        #     return {
        #         "statusCode": 200,
        #         "body": json.dumps({"message": "Daily tweet limit reached. No tweets processed."})
        #     }

        # if monthly_count >= MAX_MONTHLY_TWEETS:
        #     print("Monthly tweet limit reached. Skipping processing.")
        #     return {
        #         "statusCode": 200,
        #         "body": json.dumps({"message": "Monthly tweet limit reached. No tweets processed."})
        #     }

        # Fetch new tweets
        # tweets = get_recent_tweets()
        tweets = tweets_from_rss_feed
        # tweets =[{'text': "RT by @CNBCTV18Live: F&O Weekly Expiry Cat & Mouse - SEBI Steps In ! - Only Tuesday or Thursday allowed - SEBI approval needed to change expiry day So MSEI's Friday is gone. BSE already on Tuesday. NSE, I reckon will want Tuesday as well - makes business sense ! #StockMarket #Nifty #banknifty", 'id': '1897574058218360991', 'edit_history_tweet_ids': ['1897574058218360986'], 'author_id': '631810714', 'created_at': '2025-03-06T09:04:45.000Z'}]
        # tweets = [{
        #     "author_id": "CNBCTV18Live",
        #     "created_at": "Thu, 06 Mar 2025 09:04:45 GMT",
        #     "id": "1897974058218361103",
        #     "link": "https://twitter.com/CNBCTV18Live/status/1897574058218360991",
        #     "text": "#NTPCGreenEnergy secures 1,000 MW solar project in #UPPCL solar auction\n@ranimegha229",
        #     "username": "CNBCTV18Live"
        # }]

        print(tweets)
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

        # Fetch ticker list from Neo4j
        ticker_list = get_ticker_list_from_neo4j()
        print(f"Total tickers fetched: {len(ticker_list)}")
        print(ticker_list)

        # Fetch current prices in batches
        price_data = fetch_current_prices_batchwise(ticker_list, batch_size=100, sleep_time=2)
        print()
        print(f"Fetched current prices for {len(price_data)} tickers.")
        print(price_data)

        # Update prices in Neo4j
        update_bulk_prices_in_neo4j(price_data)

        # 🕒 Check time condition
        if should_run_sync(): # should_run_sync()
            print("Running Neo4j → Postgres sync...")
            try:
                stock_data = fetch_neo4j_stock_data()
                upsert_postgres_stock(conn, stock_data)
                print(f"Synced {len(stock_data)} records from Neo4j to Postgres.")
                mutual_fund_data = fetch_neo4j_mutual_fund_data()
                upsert_postgres_mutual_fund(conn, mutual_fund_data)
                print(f"Synced {len(mutual_fund_data)} mutual fund records from Neo4j to Postgres.")
                upsert_postgres_entity(conn, stock_data, mutual_fund_data)
                print("Synced entities from Neo4j to Postgres.")
            except Exception as e:
                print(f"Error during sync: {e}")
        
        # Close Neo4j connection
        connector.close()
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
# if __name__ == "__main__":
#     asyncio.run(async_lambda_handler({}, {}))


def lambda_handler(event, context):
    """AWS Lambda synchronous entry point."""
    return asyncio.run(async_lambda_handler(event, context))