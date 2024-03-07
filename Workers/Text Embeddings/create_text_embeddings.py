from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from os import environ
from dotenv import load_dotenv

app = FastAPI()

# Initialize Supabase client and MistralAIEmbeddings
load_dotenv()
url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
mistral_api_key = environ.get("MISTRAL_API_KEY")
supabase = create_client(url, key)
embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
embedding.model = "mistral-embed"

@app.post("/create-embeddings/")
async def create_embeddings(ids: List[int]):
    # Define the columns to retrieve and modify
    columns = ['id', 'title', 'description', 'width_mm', 'height_mm', 'depth_mm', 'year', 'weigth_grams', 'priceFull']
    formatted_columns = ', '.join(columns)

    # Fetch data from Supabase for the provided IDs
    response = supabase.table('posts').select(formatted_columns).in_('id', ids).execute()

    data = response.data

    # Modify and combine column data
    combined_strings = []
    for record in data:
        combined_string = " ".join([
            f"{record[col]} millimeters" if col in ['width_mm', 'height_mm', 'depth_mm'] else
            f"{record[col]} grams" if col == 'weigth_grams' else
            f"{record[col]}$" if col == 'priceFull' else
            str(record[col])
            for col in columns if record[col] is not None
        ])
        combined_strings.append(combined_string)

    # Create and update embeddings
    embeddings = embedding.embed_documents(combined_strings)
    for i, doc_id in enumerate(ids):
        update_response = supabase.table('embeddings').upsert({"id": doc_id, "embeddings": embeddings[i]}).execute()

    return {"detail": "success"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)