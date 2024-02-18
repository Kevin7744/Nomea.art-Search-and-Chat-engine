from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
mistral_api_key = environ.get("MISTRAL_API_KEY")
supabase = create_client(url, key)

def create_embeddings(supabase, table_name, columns):
    batch_size=10
    # Load Mistral API key from environment variables
    mistral_api_key = mistral_api_key

    # Create embeddings instance
    embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    embedding.model = "mistral-embed"
    
    # Fetch data from table
    response = supabase.table(table_name).select(*columns).execute()
    
    # Combine columns into a single string
    descriptions = pd.Series([
        " ".join([str(record[col]) for col in columns])
        for record in response.data
    ])

    # Create embeddings in batches
    embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        batch_embeddings = embedding.embed_documents(batch)
        embeddings.extend(batch_embeddings)

    # Prepare data for insertion into table
    embedding_data = []
    for i, doc_id in enumerate([record["id"] for record in response.data]):
        embedding_data.append({
            "id": doc_id,
            "embeddings": embeddings[i]
        })

    # Update the table with embeddings
    for data in embedding_data:
        update_query = supabase.table(table_name).update(data).eq('id', data['id']).execute()

# # Example usage
# table_name = 'posts'
# columns = ['id', 'title', 'description', 'width_mm', 'height_mm', 'depth_mm', 'year', 'weigth_grams', 'priceFull']
# create_embeddings(supabase, table_name, columns)