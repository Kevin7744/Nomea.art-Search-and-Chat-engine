import requests

# Update with your API key
MISTRAL_API_KEY = ""

# Chat endpoint URL
API_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

while True:
    user_input = input("> ")

    # Prepare payload
    payload = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": user_input}],
    }

    # Set headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
    }

    # Send request and handle response
    response = requests.post(API_ENDPOINT, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        if "messages" in response_data and len(response_data["messages"]) > 0:
            assistant_response = response_data["messages"][0]["content"]
            print(f"Assistant: {assistant_response}")
        else:
            print("Empty response from the API")
            assistant_response = ""
    else:
        print(f"Error calling API: {response.status_code}")
        assistant_response = ""

    # Check for ending signals (optional)
    if assistant_response.lower().endswith("goodbye") or assistant_response.lower().endswith("bye"):
        break

print("Chatbot conversation ended.")
