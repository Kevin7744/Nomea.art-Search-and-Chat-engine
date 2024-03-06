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

def update_description(image_id, llava_description):
    response = supabase.table("image_descriptions").upsert({"id": image_id, "llava_text": llava_description}).execute()


def generate_description_with_llama(image_url):
    output = replicate.run(
        "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
        input={
            "image": image_url,
            "top_p": 1,
            "prompt": "What should I take into account when visiting this place?",
            "history": [],
            "max_tokens": 1024,
            "temperature": 0.2
        }
    )

    description = ''
    for item in output:
        description += item
    return description

# Main process
images = get_images()
for image in images:
    image_id = image['id']
    full_image_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image['image_url']}/public"
    llava_description = generate_description_with_llama(full_image_url)
    update_description(image_id, llava_description)
    print(f"Processed image with ID {image_id} using LLaMA model")