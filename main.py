from flask import Flask, request, jsonify
import os
from langchain.agents import initialize_agent, AgentType
# from langchain_community.chat_models
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

system_message = SystemMessage(
    content="""
    "You are lila, an AI-art advisory"
    "You help users with questions regarding art and offer advises to them also"
    "You have a search tool that you can use to search a superbase vector database and retrieve the necessary art work related to the user's query."
    "   - For example: I want to some art related to da vinci style"
    "   - Use the search tool and search Da-vinci style art-work"
    "Your responses should be as short as possible"
    """)


tools = []

agent_kwargs = {
    "extra_prompt_message": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

memory = ConversationSummaryBufferMemory(memory_key="memory",
                                         return_messages=True,
                                        #  llm=llm,
                                         max_token_limit=250)


agent = initialize_agent(
    tools,
    # llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    user_input_key="input"
)

# Continuous conversation loop
# @app.route("/chat", methods=["POST"])
# def chat():
#     try:
#         user_input = request.form["Body"]
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400  
#     if user_input.lower() == "end":
#         return jsonify({
#             "message": "Have a good day!"
#         })

#     agent_response = agent({"input": user_input})
#     print("Agent Response:", agent_response)  

#     assistant_message_content = agent_response.get(
#         "output", "No response from the assistant.")

#     twilio_resp = MessagingResponse()
#     twilio_resp.message(assistant_message_content)

#     return str(twilio_resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)