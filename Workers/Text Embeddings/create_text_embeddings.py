import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_mistralai import MistralAIEmbeddings
from typing import List

# Load environment variables
load_dotenv()

# Supabase and MistralAI configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Initialize Supabase client and MistralAI embeddings
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
mistral_embedding = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
mistral_embedding.model = "mistral-embed"

def fetch_unprocessed_posts() -> List[dict]:
    response = supabase.rpc('fetch_unprocessed_posts').execute()
    return response.data if response.data else []

def create_text_embeddings(posts: List[dict]):
    columns = ['id', 'title', 'description', 'width_mm', 'height_mm', 'depth_mm', 'year', 'weigth_grams', 'priceFull']
    
    combined_strings = []
    post_ids = []
    for post in posts:
        combined_string = " ".join([
            f"{post[col]} millimeters" if col in ['width_mm', 'height_mm', 'depth_mm'] else
            f"{post[col]} grams" if col == 'weigth_grams' else
            f"{post[col]}$" if col == 'priceFull' else
            str(post[col])
            for col in columns if col in post and post[col] is not None
        ])
        combined_strings.append(combined_string)
        post_ids.append(post['id'])

    embeddings = mistral_embedding.embed_documents(combined_strings)
    for i, post_id in enumerate(post_ids):
        supabase.table('embeddings').upsert({"id": post_id, "embeddings": embeddings[i]}).execute()

def main():
    unprocessed_posts = fetch_unprocessed_posts()
    if unprocessed_posts:
        create_text_embeddings(unprocessed_posts)
        print(f"Processed {len(unprocessed_posts)} posts.")
    else:
        print("No new posts to process.")

if __name__ == "__main__":
    main()