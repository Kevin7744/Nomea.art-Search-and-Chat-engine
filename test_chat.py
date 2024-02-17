from flask import Flask, request, jsonify
from langchain.agents import initialize_agent, AgentType
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from Tools.tools import SimilaritySearchTool
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), temperature=0)

system_message = SystemMessage(content="""
    " You are an AI art advisory assistant"
    " You asnwer user questions and give them the best responses as possible"
    " You have this tools that you can use to search for art on the Vector database and give back the art they might be looking for"
    " This tool `SimilaritySearchTool()`, Perfoms a similarity seach, use it when necessary"
    " For example :"
    "          >user_input: I'm looking for art of mountains"
    "           Use the similarity tool and search for art in the using the following query `mountain, hill, nature`"
    " Keep your responses as short as possible"
""")

tools = [
    SimilaritySearchTool(),
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
    user_input_key="input"
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
