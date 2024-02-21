import os
from os import environ
from dotenv import load_dotenv
from supabase import create_client
import requests
import base64

load_dotenv()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(url, key)

def download_and_update_images():
    response = supabase.table("images").select("id", "image_url").execute()
    
    for record in response.data:
        image_id = record["id"]
        image_url = record["image_url"]
        full_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image_url}/public"
        image_path = f"images/{image_id}.jpg"

        # Download the image
        image_response = requests.get(full_url)
        with open(image_path, "wb") as f:
            f.write(image_response.content)
        
        # Encode the image to base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Update the post table with the base64 encoded image
        supabase.table("posts").update({"images": image_base64}).eq("id", image_id).execute()

if __name__ == "__main__":
    download_and_update_images()
