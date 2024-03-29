import locale
import gc, json, re
import xml.etree.ElementTree as ET
from functools import partial

import os
from dotenv import load_dotenv
from os import environ

import transformers
import torch

from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from supabase import create_client

from flask import Flask, request, jsonify

import asyncio
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Load API key from .env file
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if mistral_api_key is None:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

tokenizer = transformers.AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
model = ChatMistralAI(mistral_api_key=mistral_api_key)

def delete_model(*args):
    for var in args:
        if var in globals():
            del globals()[var]

    gc.collect()
    torch.cuda.empty_cache()

url = environ.get("SUPABASE_URL")
key = environ.get("SUPABASE_SERVICE_KEY")
mistral_api_key = environ.get("MISTRAL_API_KEY")
supabase = create_client(url, key)

def convert_to_embedding(query):
    embedding = MistralAIEmbeddings(mistral_api_key=mistral_api_key)
    embedding.model = "mistral-embed"
    return embedding.embed_documents([query])[0]

def find_similar_documents(query_embedding):
    response = supabase.rpc('match_documents', {
        'query_embedding': query_embedding,
        'filter': {}
    }).execute()

    return [record['id'] for record in response.data]

class SimilarityTool(BaseModel):
    """performs a similarity search based on query on a Supabase database."""
    query: str = Field(description="The search query to use for similarity search.")
    similar_documents: list = Field(default=[], description="List of IDs of similar images.")

    @validator("query")
    def query_must_not_be_empty(cls, value):
        if not value:
            raise ValueError("query cannot be empty.")
        return value

    def find_similar_documents(self):
        query_embedding = convert_to_embedding(self.query)
        similar_documents = find_similar_documents(query_embedding)
        self.similar_documents = similar_documents

executor = ThreadPoolExecutor()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['user_input']
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(generate_hermes_async(user_input, model, tokenizer, loop, {}))
    return jsonify({'response': response})

async def generate_hermes_async(query, model, tokenizer, loop, generation_config_overrides={}):
    prompt = f"""system
You are Lila, an AI art advisory with access to the following function:
Use it only when necessary, otherwise just answer the user's questions without using the function. 
You response should as short as possible.
{convert_pydantic_to_openai_function(SimilarityTool)}


NOTE: When you have used the function, wait for the response to give the user a response.

Keep your responses as short as possible.
                               
Don't make up lies, if you can't use function, just say `there is a problem getting the response`.
user
{query}
assistant"""

    with torch.inference_mode():
        completion = await loop.run_in_executor(executor, partial(model.invoke, [{"role": "user", "content": prompt}]))

    if isinstance(completion, str):
        content = completion.strip()
    else:
        content = completion.content.strip()

    functions = extract_function_calls(content)

    if functions:
        for function in functions:
            if function['name'] == 'SimilarityTool':
                similarity_tool = SimilarityTool(**function['arguments'])
                similarity_tool.find_similar_documents()
                while not similarity_tool.similar_documents:  # Wait until the tool has found similar documents
                    await asyncio.sleep(0.1)
                response = f"Similar documents for query '{similarity_tool.query}': {similarity_tool.similar_documents}"
                return response
    else:
        return content

def extract_function_calls(completion):
    if isinstance(completion, str):
        content = completion
    else:
        content = completion.content

    pattern = r"<functions>(.*?)</functions>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None

    multiplefn = match.group(1)
    functions = []
    for fn_match in re.finditer(r"<functioncall>(.*?)</functioncall>", multiplefn, re.DOTALL):
        fn_text = fn_match.group(1)
        try:
            functions.append(json.loads(fn_text))
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON

    return functions

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)



# def test_similarity_tool():
#     # Create an instance of SimilarityTool with a sample query
#     query = "sample query"
#     similarity_tool = SimilarityTool(query=query)
    
#     # Call the find_similar_documents method to perform the embedding conversion and similarity search
#     similarity_tool.find_similar_documents()

#     # Access the similar_documents attribute
#     similar_documents = similarity_tool.similar_documents

#     # Print the result
#     print(f"Similar documents for query '{query}': {similar_documents}")

# if __name__ == "__main__":
#     test_similarity_tool()