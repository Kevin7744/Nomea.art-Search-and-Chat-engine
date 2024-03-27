from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import requests
import json
# Import the keys from keys.py
from keys import (MISTRAL_API_KEY, ART_SEARCH_API_ENDPOINT)

app = FastAPI()

# Use the imported keys
client = MistralClient(api_key=MISTRAL_API_KEY)

def search_art(params):
    query = params.get("query", "")
    url = ART_SEARCH_API_ENDPOINT
    data = {'query': query, 'top_k': 100}
    response = requests.post(url, data=data)

    if response.status_code == 200:
        return json.dumps({'results': response.json()})
    else:
        print("Error fetching art data:", response.text)
        return json.dumps({'error': 'Failed to fetch art data'})

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_art",
            "description": "Search for art based on user query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's search query for art.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

names_to_functions = {
    'search_art': search_art
}

message_template = """
You are Lila, an AI art advisor. 
Your task is to answer users' questions about art.
Provide art recommendations, answer questions the user might have about the history of art and art in general.
Provide advice on which art suits the space it suits to have the art.

### Function calling
You have access to a tool called 'search_art', which you can use to search for artworks based on a user's query. 
When using tools, reply with "Is that what you are looking for?" and keep your answers short and concise.
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    user_input = chat_request.user_input
    if user_input.strip():
        user_message = ChatMessage(role="user", content=user_input)
        system_message = ChatMessage(role="system", content=message_template)
        response = client.chat(
            model="mistral-large-latest",
            messages=[system_message, user_message],
            tools=tools,
            tool_choice="auto"
        )
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)
            if function_name in names_to_functions:
                function_result = names_to_functions[function_name](function_params)
                result_message = json.loads(function_result)
                if 'results' in result_message:
                    content = "Is that what you are looking for? " + str(result_message['results'])
                elif 'error' in result_message:
                    content = "Error: " + result_message['error']
                else:
                    content = "My bad, I couldn't find any art similar to what you were looking for."
            else:
                content = "Sorry, I couldn't find the information you were looking for."
        else:
            content = response.choices[0].message.content
    return {"response": content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)