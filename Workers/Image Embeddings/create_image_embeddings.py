from os import environ
import requests
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Supabase setup
url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(url, key)

# Load the pre-trained model for image embeddings
model = SentenceTransformer('clip-ViT-B-32')

def download_and_update_images():
    # Fetch image records from the Supabase table
    response = supabase.table("images").select("id", "image_url").execute()
    
    for record in response.data:
        image_id = record["id"]
        image_url = record["image_url"]
        full_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image_url}/public"
        
        # Download the image
        response = requests.get(full_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            
            # Convert the image to embeddings using the pre-trained model
            # Ensure the image is converted to RGB format to avoid issues with models expecting 3 channels
            image_embeddings = model.encode([image.convert("RGB")])[0]  # Ensure it's a list even for a single image
            
            # Update the `new_image_embeddings` column in the `posts` table
            # Directly store the embeddings as an array of floats
            supabase.table("posts").update({"new_image_embeddings": image_embeddings.tolist()}).eq("id", image_id).execute()
        else:
            print(f"Failed to download image with ID {image_id} from {full_url}")

# Example usage
if __name__ == "__main__":
    download_and_update_images()
