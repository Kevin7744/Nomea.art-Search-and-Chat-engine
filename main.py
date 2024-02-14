from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
import pandas as pd

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
mistral_api_key = environ.get("MISTRAL_API_KEY")
supabase = create_client(url, key)

# Fetch data from "documents" table
response = supabase.table('documents').select('id', 'content', 'content_two', 'content_three', 'content_four', 'content_five').execute()

# Extract descriptions and handle missing values
descriptions = pd.Series([
    " ".join([
        record["content"],
        record["content_two"],
        record["content_three"],
        record["content_four"],
        record["content_five"]
    ])
    for record in response.data
])
descriptions = descriptions.fillna("")

# Create embeddings from text 
embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
embedding.model = "mistral-embed"

# Create embeddings in batches (recommended for efficiency & API limits)
batch_size = 10
embeddings = []
for i in range(0, len(descriptions), batch_size):
    batch = descriptions[i:i+batch_size]
    batch_embeddings = embedding.embed_documents(batch)
    embeddings.extend(batch_embeddings)

# Prepare data for insertion into "documents" table
embedding_data = []
for i, doc_id in enumerate([record["id"] for record in response.data]):
    embedding_data.append({
        "id": doc_id,
        "embedding": embeddings[i]
    })

# Update the "documents" table with embeddings
for data in embedding_data:
    update_query = supabase.table('documents').update(data).eq('id', data['id']).execute()

    # # Handle errors (e.g., update failures)
    # if update_query.error:
    #     print("Error updating embeddings for ID", data['id'], ":", update_query.error)
    # else:
    #     print("Embeddings updated successfully for ID", data['id'])

