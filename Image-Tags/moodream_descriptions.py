import replicate
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

# Create a Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def get_images():
    response = supabase.table("images").select("*").execute()
    return response.data

def update_description(image_id, description):
    response = supabase.table("image_descriptions").upsert({"id": image_id, "moondream_text": description}).execute()

def generate_description(image_url):
    output = replicate.run("lucataco/moondream1:ecd26482e4c9220957e22290cb616200b51217fe807f61653a8459ed7541e9d5",
                            input={
                                "image": image_url,
                                "prompt": "What is in the image?"
                            })
    if output:
        return ' '.join([str(item) for item in output])
    return "No description generated."

# Main process
images = get_images()
for image in images:
    image_id = image['id']
    # Combine the identifier with the base URL and suffix to form the full image URL
    full_image_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image['image_url']}/public"
    description = generate_description(full_image_url)
    update_description(image_id, description)
    print(f"Processed image with ID {image_id}")