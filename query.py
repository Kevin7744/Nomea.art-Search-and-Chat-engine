from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from pydantic import BaseModel
from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

# Supabase setup
SUPABASE_URL = environ.get("SUPABASE_URL")
SUPABASE_KEY = environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# MistralAI setup for text embeddings
MISTRAL_API_KEY = environ.get("MISTRAL_API_KEY")
mistral_embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
mistral_embedding.model = "mistral-embed"  # Assuming 'mistral-embed' is the model name

# Image model setup
image_model = SentenceTransformer('clip-ViT-B-32')  # Example image model

app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: Union[str, None] = None

@app.post("/search/")
async def search(query: Union[Query, None] = None, file: Union[UploadFile, None] = None):
    if query and query.text:
        # Handle text input
        text_embedding = mistral_embedding.embed_documents([query.text])[0]
        return await search_by_embedding(text_embedding, is_image=False)
    elif file:
        # Handle image input
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_embedding = image_model.encode([image])[0].tolist()
        return await search_by_embedding(image_embedding, is_image=True)
    else:
        raise HTTPException(status_code=400, detail="No valid input provided")

async def search_by_embedding(embedding, is_image):
    if is_image:
        function_name = "match_images"
        embedding_param = list(embedding)  # Convert to list for FLOAT8[] compatibility
    else:
        function_name = "match_documents"
        embedding_param = np.array(embedding).astype(np.float32).tolist()  # Ensure it's a list of float32

    data = await supabase.rpc(function_name, {"query_embedding": embedding_param, "result_limit": 10}).execute()
    if data.error:
        raise HTTPException(status_code=500, detail="Error executing RPC function")
    return data.data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
