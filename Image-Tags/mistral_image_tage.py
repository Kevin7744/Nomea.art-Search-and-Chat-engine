import os
from dotenv import load_dotenv
from supabase import create_client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load environment variables
load_dotenv()

# Supabase details
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Mistral API details
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

def create_image_tags(image_description: str):
    prompt = f"""
    You are given an image description and need to evaluate it and provide tags.

    # Image descriptions:
    {image_description}

    Create short image tags from the image description.
    The image tags should be precise and up to the context of the image description.

    For example:
    ```
    blue, travel, dynamic, paint, animals, beauty, 3D, imaginative, colorful
    ```

    There should be at least 15 tags and a maximum of 20.
    Each image tag should be separated by a comma.
    """
    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    ai_response = mistral_client.chat(
        model=model,
        messages=messages,
    )
    return ai_response.choices[0].message.content.strip()

def update_image_tags():
    response = supabase.table("image_descriptions").select("id", "llava_text").execute()

    for record in response.data:
        if record["llava_text"]:
            image_tags = create_image_tags(record["llava_text"])
            
            # Update the image_tags column for the current record
            supabase.table("image_descriptions").update({"image_tags": image_tags}).eq("id", record["id"]).execute()

if __name__ == "__main__":
    update_image_tags()
