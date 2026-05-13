import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get('DATABASE_URL')
print(f"Connecting to: {url[:50]}...")  # hide password

try:
    conn = psycopg2.connect(url)
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")