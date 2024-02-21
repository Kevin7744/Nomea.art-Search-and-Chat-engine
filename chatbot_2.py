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
    similar_documents: list = Field(description="List of IDs of similar images.")

    @validator("query")
    def query_must_not_be_empty(cls, field):
        if not field:
            raise ValueError("query cannot be empty")
        else:
            query_embedding = convert_to_embedding(field)
            similar_documents = find_similar_documents(query_embedding)
        return similar_documents

# convert_pydantic_to_openai_function(SimilarityTool)
    
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

def generate_hermes(query, model, tokenizer, generation_config_overrides={}):
    # fn = """{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": value_2, ...}}"""
    prompt = f"""system
You are Lila, an AI art advisory with access to the following function:
Use it only when necessary, otherwise just answer the user's questions without using the function. 
You response should as short as possible.
{convert_pydantic_to_openai_function(SimilarityTool)}

when you have used the function, have you response like:

Answer: just say "Is that what you are looking for?" when you have used the function.

NOTE: When you have used the function, wait for the response to give the user a response.

Keep your responses as short as possible.
                               
Don't make up lies, if you can't use function, just say `there is a problem getting the response`.
user
{query}
assistant"""

    with torch.inference_mode():
        completion = model.invoke([{"role": "user", "content": prompt}])

    if isinstance(completion, str):
        content = completion.strip()
    else:
        content = completion.content.strip()

    functions = extract_function_calls(content)

    if functions:
        return functions
    else:
        return content



# Your existing code for setting up the model, tokenizer, etc.

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data['user_input']
    response = generate_hermes(user_input, model, tokenizer, {})
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)