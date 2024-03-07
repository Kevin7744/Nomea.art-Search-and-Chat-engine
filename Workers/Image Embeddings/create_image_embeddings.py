from fastapi import FastAPI
from typing import List
from os import environ
import requests
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import logging

app = FastAPI()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and initialize Supabase client
load_dotenv()
url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(url, key)

# Load the pre-trained model for image embeddings
model = SentenceTransformer('clip-ViT-B-32')

@app.post("/create-image-embeddings/")
async def create_image_embeddings(ids: List[int]):
    logger.info(f"Received request to create image embeddings for IDs: {ids}")

    for image_id in ids:
        try:
            # Fetch image URL from 'images' table using post_id
            image_response = supabase.table("images").select("image_url").eq("post_id", image_id).execute()
            if image_response.data:
                image_url = image_response.data[0]["image_url"]
                full_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image_url}/public"

                logger.info(f"Downloading image with post_id {image_id} from {full_url}")
                img_response = requests.get(full_url)
                img_response.raise_for_status()
                image = Image.open(BytesIO(img_response.content))

                # Convert the image to embeddings
                logger.info(f"Creating embeddings for image with post_id {image_id}")
                image_embeddings = model.encode([image.convert("RGB")])[0]

                # Upsert into the 'embeddings' table in the 'new_image_embeddings' column
                logger.info(f"Upserting new image embeddings for post_id {image_id} in Supabase")
                supabase.table("embeddings").upsert({
                    "id": image_id,
                    "new_image_embeddings": image_embeddings.tolist()
                }).execute()

        except Exception as e:
            logger.error(f"An error occurred while processing image with post_id {image_id}: {e}")
            continue

    return {"detail": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
