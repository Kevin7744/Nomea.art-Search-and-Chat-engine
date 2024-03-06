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

def update_image_tags(image_id, tags):
    response = supabase.table("image_descriptions").upsert({"id": image_id, "image_tags": tags}).execute()

def generate_tags_with_llama(image_url):
    output = replicate.run(
        "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
        input={
            "image": image_url,
            "top_p": 1,
            "prompt": "generate me a list of tags, that describe the image as detailed and exact as possible (style, color, subject, medium), example should be like this: blue, man, raining, vintage",
            "history": [],
            "max_tokens": 1024,
            "temperature": 0.2
        }
    )

    tags = ''
    for item in output:
        tags += item
    return tags

# Main process
images = get_images()
for image in images:
    image_id = image['id']
    full_image_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image['image_url']}/public"
    image_tags = generate_tags_with_llama(full_image_url)
    update_image_tags(image_id, image_tags)
    print(f"Processed image with ID {image_id} and updated tags using LLaMA model")