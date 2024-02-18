from flask import Flask, request, jsonify
from langchain.agents import initialize_agent, AgentType
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from Tools.tools import similarity_search
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), temperature=0)

system_message = SystemMessage(
content="""
You name is Lila, an art advisory assistant.
You answer user's questions and give them the best responses.
You have tools that you can use to search for art on the Vector database.
This tool `similarity_search`, Perfoms a similarity search based on user's query and retrieves a list of image ID's, use it when necessary.

                               
For example :
                               
Use the following format:

Question:  user's Question here
Query: Query to run with similarity_search from user's question
Result: Result from the similarity_search
Answer: Don't display the "Result" to user just say "Is that what you are looking for?"

Wait for API response to give back the response to the user.                             
Keep your responses as short as possible.
                               
Don't make up lies, if you can't use SimilaritySearchTool, just say `there is a problem getting the response`.
"""
)

tools = [
    similarity_search,
]

agent_kwargs = {
    "extra_prompt_message": [],
    "system_message": system_message,
}

memory = ConversationSummaryBufferMemory(memory_key="memory",
                                         return_messages=True,
                                         llm=llm,
                                         max_token_limit=250)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    user_input_key="input",
    handle_parsing_errors=True,
)

# API endpoint to receive messages
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.form["Body"]
    except Exception as e:
        return jsonify({"error": str(e)}), 400  

    if user_input.lower() == "end":
        return jsonify({"message": "Have a good day!"})

    agent_response = agent({"input": user_input})
    print("Agent Response:", agent_response)  

    assistant_message_content = agent_response.get("output", "No response from the assistant.")

    return jsonify({"response": assistant_message_content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
