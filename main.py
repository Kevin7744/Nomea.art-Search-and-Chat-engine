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

# Fetch data from "posts" table
response = supabase.table('posts').select('id', 'title', 'description', 'width_mm', 'height_mm', 'depth_mm', 'year', 'weigth_grams', 'priceFull').execute()

# Combine columns into a single string
descriptions = pd.Series([
    " ".join([
        str(record["title"]),
        str(record["description"]),
        str(record["width_mm"]),
        str(record["height_mm"]),
        str(record["depth_mm"]),
        str(record["year"]),
        str(record["weigth_grams"]),
        str(record["priceFull"])
    ])
    for record in response.data
])

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

# Prepare data for insertion into "posts" table
embedding_data = []
for i, doc_id in enumerate([record["id"] for record in response.data]):
    embedding_data.append({
        "id": doc_id,
        "embeddings": embeddings[i]
    })

# Update the "posts" table with embeddings
for data in embedding_data:
    update_query = supabase.table('posts').update(data).eq('id', data['id']).execute()
