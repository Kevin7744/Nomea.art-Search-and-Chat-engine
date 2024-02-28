import json
import os
import requests
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Initialize Mistral Client with your API key
api_key = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)

def search_art(params):
    """Function to call the art search API and return results in JSON format."""
    query = params.get("query", "")
    url = f"http://178.254.23.172/api/search/?query={query}"
    
    response = requests.post(url)

    if response.status_code == 200:
        return json.dumps({'results': response.json()})
    else:
        # Log or print more detailed error information for debugging
        print("Error fetching art data:", response.text)
        return json.dumps({'error': 'Failed to fetch art data'})


# Define the tool for art search
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

# Mapping function names to function objects
names_to_functions = {
    'search_art': search_art
}

# Message template providing context about the chatbot's role and tools
message_template = """
You are Lila, an AI art advisor. 
Your task is to answer users' questions about art. 
You have access to a tool called 'search_art', which you can use to search for artworks based on a user's query. 
When using tools, reply with "Is that what you are looking for?" and keep your answers short and concise.
"""

# Main chatbot function
def art_advisor_chatbot():
    while True:
        user_input = input("You: ")
        if user_input.strip():
            # Construct the message for the chat
            user_message = ChatMessage(role="user", content=user_input)
            system_message = ChatMessage(role="system", content=message_template)

            # Generate chatbot response using Mistral model with function calling
            response = client.chat(
                model="mistral-large-latest",
                messages=[system_message, user_message],  # Include the system message for context
                tools=tools,
                tool_choice="auto"
            )

            # Check if there's a tool call in the response
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                function_name = tool_call.function.name
                function_params = json.loads(tool_call.function.arguments)

                # Execute the function based on the function name
                if function_name in names_to_functions:
                    function_result = names_to_functions[function_name](function_params)
                    # Parse the function result to create a meaningful message
                    result_message = json.loads(function_result)
                    if 'results' in result_message:
                        ids = result_message['results']
                        content = "Is that what you are looking for?" + ', '.join(map(str, ids))
                    elif 'error' in result_message:
                        content = "Error: " + result_message['error']
                    else:
                        content = "My bad, I couldn't find any art similar to what you were looking for."

                    print(f"AI Art Advisor: {content}")
                else:
                    print("Sorry, I couldn't find the information you were looking for.")
            else:
                # If no tool call, use the model's response directly
                bot_response = response.choices[0].message.content
                print(f"AI Art Advisor: {bot_response}")



if __name__ == "__main__":
    art_advisor_chatbot()
