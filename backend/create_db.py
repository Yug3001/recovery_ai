import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

host = os.environ.get("MYSQL_HOST", "localhost")
port = int(os.environ.get("MYSQL_PORT", 3306))
user = os.environ.get("MYSQL_USER", "root")
password = os.environ.get("MYSQL_PASSWORD", "")
db_name = os.environ.get("MYSQL_DATABASE", "recovery_ai")

try:
    print(f"Connecting to MySQL at {host}:{port} as {user}...")
    conn = pymysql.connect(host=host, port=port, user=user, password=password)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    print(f"✅ Database '{db_name}' ensured.")
    conn.close()
except Exception as e:
    print(f"❌ Failed to create database: {e}")
