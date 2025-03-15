import pandas as pd
from dotenv import load_dotenv
import json
import os
import sys
import psycopg2
import psycopg2.extras
from datetime import datetime

sys.path.append(os.getcwd())

from analysis_agent import CryptoTechnicalAnalyst
from key_level_agent import TradingSignalExtractor

load_dotenv()

# Database connection parameters
DB_NAME = os.getenv('DB_NAME', 'crypto_analysis_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')


def connect_to_db():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def create_table_if_not_exists(conn):
    """Create the analysis results table if it doesn't exist"""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS crypto_analysis_results (
                id SERIAL PRIMARY KEY,
                analysis_date DATE NOT NULL,
                analysis_time TIME NOT NULL,
                trading_pair VARCHAR(50) NOT NULL,
                table_analysis TEXT,
                deep_analysis TEXT,
                raw_json_result JSONB,
                processed_json_result JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
        conn.commit()
        print('Table created or already exists.')
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()


def get_todays_analysis(conn, trading_pair):
    """Retrieve today's analysis for a trading pair if it exists"""
    try:
        current_date = datetime.now().date()
        with conn.cursor() as cur:
            cur.execute(
                """
            SELECT table_analysis, deep_analysis, raw_json_result, processed_json_result
            FROM crypto_analysis_results
            WHERE analysis_date = %s AND trading_pair = %s
            ORDER BY analysis_time DESC
            LIMIT 1
            """,
                (current_date, trading_pair),
            )

            result = cur.fetchone()
            return result
    except Exception as e:
        print(f"Error retrieving today's analysis: {e}")
        return None


def insert_analysis_result(
    conn, trading_pair, table_analysis, deep_analysis, raw_json, processed_json
):
    """Insert analysis results into the database"""
    try:
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()

        # Prepare raw_json for JSONB insertion
        if isinstance(raw_json, str):
            try:
                # If raw_json is a string, parse it to ensure it's valid JSON
                raw_json_obj = json.loads(raw_json)
            except json.JSONDecodeError:
                # If it's not valid JSON, create a JSON object with the string as content
                raw_json_obj = {'content': str(raw_json)}
        else:
            # If it's already a dict or other serializable object, use as is
            raw_json_obj = raw_json

        # Ensure processed_json is a proper dict for JSONB
        if not isinstance(processed_json, dict) and isinstance(processed_json, str):
            try:
                processed_json_obj = json.loads(processed_json)
            except json.JSONDecodeError:
                processed_json_obj = {'content': processed_json}
        else:
            processed_json_obj = processed_json

        with conn.cursor() as cur:
            cur.execute(
                """
            INSERT INTO crypto_analysis_results
            (analysis_date, analysis_time, trading_pair, table_analysis, deep_analysis, raw_json_result, processed_json_result)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    current_date,
                    current_time,
                    trading_pair,
                    table_analysis,
                    deep_analysis,
                    psycopg2.extras.Json(raw_json_obj),
                    psycopg2.extras.Json(processed_json_obj),
                ),
            )
        conn.commit()
        print(f"Analysis results for {trading_pair} saved to database.")
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()


# Main execution
def main():
    # Connect to database
    conn = connect_to_db()
    if not conn:
        print('Failed to connect to database. Exiting.')
        return

    # Create table if not exists
    create_table_if_not_exists(conn)

    # Load data and configurations
    df = pd.read_csv('user_data/pairs_analysis.csv')
    with open('deploy/config.json', 'r') as f:
        config = json.load(f)
    analyzed_pairs = []
    for pair in df['交易对'][:5]:
        print(f"Analyzing pair: {pair}")

        # Check if we already have today's analysis
        existing_analysis = get_todays_analysis(conn, pair)

        if existing_analysis:
            # Use existing analysis from today
            print(f"Using existing analysis from today for {pair}")
            table_output, analysis_text, raw_json_db, processed_json_db = existing_analysis

            # Convert from database JSONB to Python objects
            if isinstance(raw_json_db, str):
                raw_json = json.loads(raw_json_db)
            else:
                raw_json = raw_json_db

            if isinstance(processed_json_db, str):
                processed_json = json.loads(processed_json_db)
            else:
                processed_json = processed_json_db

            print('Using existing deep analysis from database')
            print(analysis_text)
        else:
            # Perform new analysis
            print(f"No existing analysis found for today. Calculating new analysis for {pair}")

            # Initialize analyzer
            analyst = CryptoTechnicalAnalyst(
                api_key=config['exchange']['key'],
                api_secret=config['exchange']['secret'],
            )
            base_pair = pair.split(':')[0]

            # Analyze trading pair
            analyst.analyze_crypto(base_pair)

            # Generate formatted table
            table_output = analyst.generate_formatted_table(base_pair)

            # Get deep analysis
            analysis_text = analyst.get_llm_analysis(base_pair)

            print(analysis_text)

            # Extract trading signals
            extractor = TradingSignalExtractor()
            raw_json, processed_json = extractor.extract_to_json_string(
                analysis_text, consolidate=True
            )

            print('\nRaw JSON:')
            print(raw_json)
            print('\nProcessed JSON:')
            print(processed_json)

            analyzed_pairs.append(pair)

            # Store results in database
            insert_analysis_result(
                conn, pair, table_output, analysis_text, raw_json, processed_json
            )

    print('配置文件已更新')
    print('分析结果已存储到PostgreSQL数据库')

    # Close database connection
    conn.close()
    return analyzed_pairs


if __name__ == '__main__':
    main()
