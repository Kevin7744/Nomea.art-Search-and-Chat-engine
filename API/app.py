from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Union
from os import environ
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import requests
from pydantic import BaseModel

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


class EmbeddingRequest(BaseModel):
    ids: List[int]

@app.get("/ping")
async def ping():
    return "pong"

@app.post("/create-text-embeddings/")
async def create_text_embeddings(request_body: EmbeddingRequest):
    ids = request_body.ids
    columns = ['id', 'title', 'description', 'width_mm', 'height_mm', 'depth_mm', 'year', 'weigth_grams', 'priceFull']
    formatted_columns = ', '.join(columns)
    response = supabase.table('posts').select(formatted_columns).in_('id', ids).execute()

    data = response.data
    if not data:
        raise HTTPException(status_code=404, detail="No records found for the provided IDs.")

    combined_strings = []
    for record in data:
        combined_string = " ".join([
            f"{record[col]} millimeters" if col in ['width_mm', 'height_mm', 'depth_mm'] else
            f"{record[col]} grams" if col == 'weigth_grams' else
            f"{record[col]}$" if col == 'priceFull' else
            str(record[col])
            for col in columns if col in record and record[col] is not None
        ])
        combined_strings.append(combined_string)

    embeddings = mistral_embedding.embed_documents(combined_strings)
    for i, doc_id in enumerate(ids):
        supabase.table('embeddings').upsert({"id": doc_id, "embeddings": embeddings[i]}).execute()

    return {"detail": "Text embeddings creation process completed"}


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


@app.post("/search/")
async def search(query: Optional[str] = Form(default=None), file: Optional[UploadFile] = File(default=None), top_k: Optional[int] = Form(100)) -> Union[List[int], List[dict]]:
    if query:
        print(f"Received text query: {query}")
        text_embedding = mistral_embedding.embed_documents([query])[0]
        response = supabase.rpc("match_documents", {"query_embedding": text_embedding}).execute()
        return [item['id'] for item in response.data]

    elif file:
        print(f"Received file: {file.filename}")
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")  # Convert image to RGB

        # Process image using the Sentence Transformer model
        query_embedding = model.encode([image])[0]  # Ensure it's a list even for a single image

        # Fetch embeddings from the database
        response = supabase.table("embeddings").select("id", "new_image_embeddings").execute()
        db_embeddings = []
        ids = []

        # Filter embeddings to ensure consistent shape
        for record in response.data:
            embedding = record.get('new_image_embeddings')
            if embedding and len(embedding) == len(query_embedding):
                db_embeddings.append(embedding)
                ids.append(record['id'])

        if db_embeddings:
            # Convert filtered embeddings to a NumPy array of type float32
            db_embeddings = np.array(db_embeddings, dtype=np.float32)

            # Convert query_embedding to a PyTorch tensor of type float32
            query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32)

            # Compute cosine similarities
            similarities = util.pytorch_cos_sim(query_embedding_tensor, torch.tensor(db_embeddings))[0].numpy()

            # Get top k results based on similarities
            top_k_indices = np.argsort(similarities)[::-1][:top_k]

            # Return just a list of IDs based on the top k indices
            return [ids[idx] for idx in top_k_indices]
        else:
            return []



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)