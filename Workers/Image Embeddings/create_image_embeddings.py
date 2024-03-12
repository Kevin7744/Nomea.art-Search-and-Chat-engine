from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import requests

load_dotenv()

SUPABASE_URL = environ.get("SUPABASE_URL")
SUPABASE_KEY = environ.get("SUPABASE_SERVICE_KEY")
MISTRAL_API_KEY = environ.get("MISTRAL_API_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

mistral_embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
mistral_embedding.model = "mistral-embed"

image_model = SentenceTransformer('clip-ViT-B-32')

model = SentenceTransformer('clip-ViT-B-32')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/create-image-embeddings/")
async def create_image_embeddings(ids: List[int]):
    for image_id in ids:
        image_response = supabase.table("images").select("image_url").eq("id", image_id).execute()

        if not image_response.data:
            continue  # Skip if no image found

        image_url = image_response.data[0]["image_url"]
        full_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image_url}/public"

        img_response = requests.get(full_url)
        img_response.raise_for_status()
        image = Image.open(BytesIO(img_response.content))

        image_embeddings = image_model.encode([image.convert("RGB")])[0]
        supabase.table("embeddings").upsert({
            "id": image_id,
            "new_image_embeddings": image_embeddings.tolist()
        }).execute()

    return {"detail": "Image embeddings creation process completed"}