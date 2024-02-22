from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional, List, Union
from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sentence_transformers import util  # For cosine similarity

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

# CLIP setup for image embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()

@app.post("/search/")
async def search(query: Optional[str] = Form(default=None), file: Optional[UploadFile] = File(default=None), top_k: Optional[int] = Form(3)) -> Union[List[int], List[dict]]:
    if query:
        print(f"Received text query: {query}")
        # Convert text to embeddings using MistralAI
        text_embedding = mistral_embedding.embed_documents([query])[0]
        # Call Supabase function for text search
        response = supabase.rpc("match_documents", {"query_embedding": text_embedding}).execute()
        # Return a list of IDs for text search
        return [item['id'] for item in response.data]

    elif file:
        # Image search part
        print(f"Received file: {file.filename}")
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        inputs = clip_processor(images=image, return_tensors="pt")
        query_embedding = clip_model.get_image_features(**inputs).detach().numpy()

        # Fetch embeddings from the database
        response = supabase.table("posts").select("id", "new_image_embeddings").execute()
        
        # Preprocess and validate embeddings before converting to NumPy array
        db_embeddings = []
        for record in response.data:
            embedding = record['new_image_embeddings']
            if isinstance(embedding, list) and len(embedding) == len(query_embedding):
                db_embeddings.append(embedding)
            else:
                print(f"Skipping invalid embedding for id {record['id']}")
        
        # Convert to NumPy array and compute similarities
        if db_embeddings:
            db_embeddings = np.array(db_embeddings)
            similarities = util.pytorch_cos_sim(query_embedding, db_embeddings)[0].numpy()

            # Get top k results based on similarities
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            return [{"id": response.data[idx]['id'], "similarity": float(similarities[idx])} for idx in top_k_indices]
        else:
            return []

    # Error handling for invalid input
    else:
        raise HTTPException(status_code=400, detail="No valid input provided")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
