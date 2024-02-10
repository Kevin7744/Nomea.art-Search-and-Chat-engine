from os import environ
from dotenv import load_dotenv
from supabase.client import Client, create_client

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")

supabase = create_client(url, key)

# Function to fetch values from the documents table
def fetch_embeddings():
    # Fetching values from the 'documents' table
    response = supabase.from_("documents").select("embedding").execute()
    
    # Check if the query was successful
    if response.status_code != 200:
        print("Error fetching embeddings:", response.get("error"))
        return
    
    # Extracting embeddings from the response
    embeddings = response.get("data")
    
    # Check if embeddings are empty
    if not embeddings:
        print("Empty")
        return
    
    # Printing retrieved embeddings
    print("Retrieved embeddings:")
    for row in embeddings:
        print(row.get("embedding"))

# Call the function to fetch embeddings
fetch_embeddings()
