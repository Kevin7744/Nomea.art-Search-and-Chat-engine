from os import environ
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import requests

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(url, key)

# Load the pre-trained model for image embeddings
model = SentenceTransformer('clip-ViT-B-32')

def download_and_update_images():
    response = supabase.table("images").select("id", "image_url").execute()
    for record in response["data"]:
        image_id = record["id"]
        image_url = record["image_url"]
        full_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image_url}/public"
        
        # Download the image
        response = requests.get(full_url)
        image = Image.open(BytesIO(response.content))
        
        # Convert the image to embeddings using the pre-trained model
        image_embeddings = model.encode(image)
        
        # Update the `image_embeddings` column in the `posts` table
        update_query = supabase.table("posts").update({"image_embeddings": image_embeddings}).eq("id", image_id).execute()

# Example usage
download_and_update_images()
