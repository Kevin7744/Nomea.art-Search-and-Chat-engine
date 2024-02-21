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

triggered = False  # Variable to track if the function call should be triggered

def generate_hermes(query, model, tokenizer, generation_config_overrides={}):
    global triggered

    main_prompt = """system
You are Lila, an AI art advisory with access to the following function:

{convert_pydantic_to_openai_function(SimilarityTool)}

To use these functions respond with:
<function>
    <functioncall> {fn} </functioncall>
</function>

For example:
Question: user's Question here
Query: Query to run with similarity_search from user's question
Result: Result from the similarity_search
Answer: Don't display the "Result" to the user just say "Is that what you are looking for?"

Wait for API response to give back the response to the user.                             
Keep your responses as short as possible.
                               
Don't make up lies, if you can't use SimilaritySearchTool, just say `there is a problem getting the response`.
user
{query}
assistant"""

    if not triggered:
        prompt = main_prompt
    else:
        prompt = f"user\n{query}\nassistant"

    with torch.inference_mode():
        completion = model.invoke([{"role": "user", "content": prompt}])

    if not triggered:
        triggered = True

    if isinstance(completion, str):
        content = completion.strip()
    else:
        content = completion.content.strip()

    functions = extract_function_calls(content)

    if functions:
        print(functions)
    else:
        print(content)
    print("=" * 100)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data['input']

        # Use the MistralAI model to generate a response
        agent_response = model({"input": user_input})
        assistant_message_content = agent_response.get("output", "No response from the assistant.")

        return jsonify({'response': assistant_message_content})
    except KeyError:
        return jsonify({'error': 'Invalid request format. Please provide input in the correct format.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)