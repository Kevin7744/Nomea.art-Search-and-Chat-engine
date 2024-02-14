from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
mistral_api_key = environ.get("MISTRAL_API_KEY")
supabase = create_client(url, key)

# Function to convert query to vector embedding
def convert_to_embedding(query):
    embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    embedding.model = "mistral-embed"
    return embedding.embed_documents([query])[0]

# Function to find similar documents using Supabase SQL function
def find_similar_documents(query_embedding):
    response = supabase.rpc('match_documents', {
        'query_embedding': query_embedding,
        'filter': {}
    }).execute()
    
    return [record['id'] for record in response.data]

# Input query
query = input("Enter your query: ")
query_embedding = convert_to_embedding(query)

# Find similar documents
similar_documents = find_similar_documents(query_embedding)
print(similar_documents)
