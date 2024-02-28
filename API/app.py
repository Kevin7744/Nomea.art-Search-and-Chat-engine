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

load_dotenv()


SUPABASE_URL = environ.get("SUPABASE_URL")
SUPABASE_KEY = environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


MISTRAL_API_KEY = environ.get("MISTRAL_API_KEY")
mistral_embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
mistral_embedding.model = "mistral-embed"

model = SentenceTransformer('clip-ViT-B-32')

app = FastAPI()

# Add CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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