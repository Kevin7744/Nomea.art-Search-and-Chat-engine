from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# Load environment variables
load_dotenv()

# Supabase setup
SUPABASE_URL = environ.get("SUPABASE_URL")
SUPABASE_KEY = environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# MistralAI setup for text embeddings
MISTRAL_API_KEY = environ.get("MISTRAL_API_KEY")
mistral_embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
mistral_embedding.model = "mistral-embed"

# Image model setup using CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()

@app.post("/search/")
async def search(query: Optional[str] = Form(default=None), file: Optional[UploadFile] = File(default=None)):
    if query:
        print(f"Received text query: {query}")
        # Convert text to embeddings using MistralAI
        text_embedding = mistral_embedding.embed_documents([query])[0]
        # Call Supabase function for text search
        response = supabase.rpc("match_documents", {"query_embedding": text_embedding}).execute()
    elif file:
        print(f"Received file: {file.filename}")
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        # Process image using CLIP
        inputs = clip_processor(images=image, return_tensors="pt")
        image_embedding = clip_model.get_image_features(**inputs).detach().numpy()[0].tolist()
        # Call Supabase function for image search
        response = supabase.rpc("match_images", {"query_embedding": image_embedding, "result_limit": 10}).execute()
    else:
        raise HTTPException(status_code=400, detail="No valid input provided")

    # Extract IDs from response data
    ids = [item['id'] for item in response.data]
    return ids

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)