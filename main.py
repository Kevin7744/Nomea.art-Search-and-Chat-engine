import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from PIL import Image
from io import BytesIO
import requests
from sentence_transformers import SentenceTransformer
import replicate
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load environment variables
load_dotenv()

# Supabase and MistralAI configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
IMAGE_EMBEDDING_MODEL = 'clip-ViT-B-32'

# Initialize Supabase client, MistralAI embeddings, SentenceTransformer, and MistralClient
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
mistral_embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
mistral_embedding.model = "mistral-embed"
image_embedding_model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)
client = MistralClient(api_key=MISTRAL_API_KEY)
replicate.api_token = os.getenv('REPLICATE_API_TOKEN')

def fetch_new_posts():
    response = supabase.rpc('fetch_new_posts', params={}).execute()
    return response.data if response.data else []

def fetch_images_for_post(post_id):
    response = supabase.table("images").select("id", "image_url").eq("post_id", post_id).execute()
    return response.data

def create_image_embedding(image_url):
    full_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image_url}/public"
    response = requests.get(full_url)
    image = Image.open(BytesIO(response.content))
    image_embedding = image_embedding_model.encode([image.convert("RGB")])[0]
    return image_embedding.tolist()

def generate_image_description(image_url):
    full_url = f"https://imagedelivery.net/vfguozVHBGZa-6s8NQZayA/{image_url}/public"
    try:
        output = replicate.run(
            "yorickvp/llava-13b:a0fdc44e4f2e1f20f2bb4e27846899953ac8e66c5886c5878fa1d6b73ce009e5",
            input={
                "image": full_url,
                "top_p": 1,
                "prompt": "Please provide a comprehensive and precise description of the image, focusing on its style, color palette, primary and secondary subjects, and the medium used. Detail the composition and arrangement of elements within the image, including any discernible patterns or textures. Additionally, interpret the context and possible significance of the scene depicted, and suggest the emotions or atmosphere it conveys. If applicable, comment on the artistic techniques or specific movements the image's style may align with",
                "max_tokens": 1024,
                "temperature": 0.2
            }
        )
        description = ''.join(item for item in output)
        return description
    except Exception as e:
        print(f"An error occurred while generating the description: {e}")
        return None

def create_image_tags(image_description: str):
    prompt = f"""
Your task is to analyze a detailed description of an image and generate a set of concise, relevant tags that encapsulate the essence of the image as described.

# Image Description:
{image_description}

# Instructions:
Generate Tags: Based on the provided image description, create a concise set of tags that accurately capture the key elements, themes, and attributes of the image.
Tag Characteristics: The tags should be precise, contextually relevant, and reflect various aspects of the image, including but not limited to subjects, colors, emotions, artistic styles, and notable features or actions depicted.
Quantity: Ensure that you provide a minimum of 15 tags, but do not exceed 20 tags to maintain focus and relevance.
Format: Present your tags in a comma-separated list for clarity.

## Example Tags:
blue, adventure, dynamic, painted, wildlife, beauty, 3D effect, imaginative, vibrant, tranquil, modern, abstract, expressive, urban, natural, serene, contrast, texture, perspective, illuminated

Remember, the goal is to offer a rich, multifaceted snapshot of the image's content and character through your tags, aiding in its categorization and understanding.
    """
    messages = [ChatMessage(role="user", content=prompt)]

    ai_response = client.chat(
        model="mistral-large-latest",
        messages=messages,  # Pass the ChatMessage instance directly
    )
    return ai_response.choices[0].message.content.strip()

def process_posts_and_images():
    new_posts = fetch_new_posts()
    for post in new_posts:
        # Process images related to the post
        images = fetch_images_for_post(post['id'])
        all_image_descriptions = []
        for image in images:
            image_embedding = create_image_embedding(image['image_url'])
            image_description = generate_image_description(image['image_url'])
            image_tags = create_image_tags(image_description)

            # Update image descriptions and tags table
            supabase.table("image_descriptions").upsert({
                "id": image['id'],
                "llava_text": image_description,
                "image_tags": image_tags
            }).execute()
            all_image_descriptions.append(image_description)

        # Generate text embedding for the post, combined with all related image descriptions
        text_embedding = create_text_embedding_for_post(post, ' '.join(all_image_descriptions))

        # Update embeddings table with the new text embedding
        supabase.table("embeddings").upsert({"id": post['id'], "embeddings": text_embedding}).execute()

def create_text_embedding_for_post(post, image_descriptions):
    # Define the columns to include in the combined string, along with any special formatting needed
    columns = ['title', 'description', 'year', 'width_mm', 'height_mm', 'depth_mm', 'weight_grams', 'priceFull' ]

    # Combine the strings with appropriate formatting
    combined_string = " ".join([
        f"{post[col]} millimeters" if col in ['width_mm', 'height_mm', 'depth_mm'] else
        f"{post[col]} grams" if col == 'weight_grams' else
        f"{post[col]}$" if col == 'priceFull' else
        str(post.get(col, ''))
        for col in columns if col in post and post[col] is not None
    ])

    # Combine post details with all related image descriptions
    final_combined_text = combined_string + " " + image_descriptions
    
    # Generate the text embedding for the final combined text
    text_embedding = mistral_embedding.embed_documents([final_combined_text])[0]
    
    return text_embedding

if __name__ == "__main__":
    process_posts_and_images()
