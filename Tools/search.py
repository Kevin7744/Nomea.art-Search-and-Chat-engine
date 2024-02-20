import json
import re
import torch
import transformers

from supabase import create_client

from pydantic import BaseModel, Field

from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI


from os import environ
from dotenv import load_dotenv

load_dotenv()

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

class SimilaritySearchInput(BaseModel):
    query: str = Field(description="The search query to use for similarity search.")

class SimilaritySearchOutput(BaseModel):
    response_code: str = Field(description="Response code for the search.")
    similar_documents: list = Field(description="List of IDs of similar images.")

class SimilaritySearchTool:
    """Tool for performing a similarity search based on query on a Supabase database."""

    def __init__(self):
        pass

    def run(self, query):
        """Perform similarity search based on the given query."""
        query_embedding = convert_to_embedding(query)
        similar_documents = find_similar_documents(query_embedding)
        return similar_documents

def extract_function_calls(completion):
    if isinstance(completion, str):
        content = completion
    else:
        content = completion.content

    pattern = r"<multiplefunctions>(.*?)</multiplefunctions>"
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

tokenizer = transformers.AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
model = ChatMistralAI(mistral_api_key=mistral_api_key)

def generate_hermes(prompt, similarity_search_tool, generation_config_overrides={}):
    fn = """{"name": "SimilaritySearchTool", "arguments": {"query": "value"}}"""
    prompt = f"""user\n{prompt}\n{fn}\nassistant"""

    with torch.inference_mode():
        completion = model.invoke([{"role": "user", "content": prompt}])

    if isinstance(completion, str):
        # Handle the case where completion is a string
        content = completion.strip()
    else:
        # Handle the case where completion is an AIMessage object
        content = completion.content.strip()

    functions = extract_function_calls(content)

    if functions:
        for function_call in functions:
            function_name = function_call.get("name")
            arguments = function_call.get("arguments")
            if function_name == "SimilaritySearchTool":
                response = similarity_search_tool.run(**arguments)
                print(response)
            else:
                print(f"Unknown function: {function_name}")
    else:
        print("No function calls found in prompt")
    print("="*100)


if __name__ == "__main__":
    prompt = "Tell me about books"
    similarity_search_tool = SimilaritySearchTool()
    generate_hermes(prompt, similarity_search_tool)
