from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from os import environ
from dotenv import load_dotenv

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
mistral_api_key = environ.get("MISTRAL_API_KEY")
supabase = create_client(url, key)

def convert_to_embedding(query):
    embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    embedding.model = "mistral-embed"
    return embedding.embed_documents([query])[0]

def find_similar_documents(query_embedding):
    response = supabase.rpc('match_documents', {
        'query_embedding': query_embedding,
        'filter': {}
    }).execute()
    
    return [record['id'] for record in response.data]

class SimilaritySearchInput(BaseModel):
    query: str = Field(description="The search query to find similar documents.")

class SimilaritySearchOutput(BaseModel):
    response_code: str = Field(description="Response code for the search.")
    similar_documents: list = Field(description="List of IDs of similar documents.")

class SimilaritySearchTool(BaseTool):
    name = "similarity_search"
    description = "Tool for performing a similarity search on a Supabase database."
    args_schema = SimilaritySearchInput
    result_schema = SimilaritySearchOutput

    def _run(self, inputs):
        query_embedding = convert_to_embedding(inputs.query)
        similar_documents = find_similar_documents(query_embedding)
        return {"response_code": "success", "similar_documents": similar_documents}
