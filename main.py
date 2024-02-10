from os import environ
from dotenv import load_dotenv
from supabase.client import Client, create_client

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")

supabase = create_client(url, key)

print("Connected to Supabase")
